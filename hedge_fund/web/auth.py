"""Identity, as reported by the platform in front of us.

This module deliberately contains no authentication. Azure Container Apps'
built-in auth terminates the Entra ID sign-in before a request ever reaches
Python and injects the result as headers; our only job is to read them and
fail closed when they are missing. Nothing here issues, validates or stores a
token or a session cookie, so there is no hand-rolled auth to get wrong.

That also means the headers are only trustworthy because the platform strips
any client-supplied copy and is the sole ingress. If this app is ever exposed
without built-in auth in front of it, the headers become attacker-controlled
and REQUIRE_AUTH is the thing standing between that and an open dashboard.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass

# Set by Container Apps' built-in auth. The base64 one carries the full claim
# set; the other two are conveniences the platform derives from it.
PRINCIPAL_HEADER = "x-ms-client-principal"
NAME_HEADER = "x-ms-client-principal-name"
ID_HEADER = "x-ms-client-principal-id"

# Default on: an unauthenticated deploy has to be a deliberate act, not what
# you get by forgetting an environment variable.
REQUIRE_AUTH = os.environ.get("AIHF_WEB_REQUIRE_AUTH", "true").lower() != "false"


class NotAuthenticated(Exception):
    """No platform-supplied principal on a request that needed one."""


@dataclass(frozen=True)
class Principal:
    """Who the platform says is calling."""

    name: str
    object_id: str | None = None

    @property
    def display(self) -> str:
        return self.name


ANONYMOUS = Principal(name="local-dev")


def principal_from_headers(headers) -> Principal:
    """Read the signed-in user out of the platform's headers.

    Raise NotAuthenticated when REQUIRE_AUTH is set and no principal is
    present. With REQUIRE_AUTH off (local development) fall back to a fixed
    anonymous identity so the dashboard is usable without an Entra tenant.
    """
    name = headers.get(NAME_HEADER)
    object_id = headers.get(ID_HEADER)

    if not name:
        # The convenience headers are absent on some configurations; the
        # base64 claims blob is authoritative, so try it before giving up.
        claims = _decode_principal(headers.get(PRINCIPAL_HEADER))
        name = claims.get("userPrincipalName") or claims.get("preferred_username") or claims.get("name")
        object_id = object_id or claims.get("oid")

    if name:
        return Principal(name=name, object_id=object_id)
    if REQUIRE_AUTH:
        raise NotAuthenticated("no platform principal on the request")
    return ANONYMOUS


def _decode_principal(encoded: str | None) -> dict:
    """Best-effort decode of the base64 claims blob. Never raises."""
    if not encoded:
        return {}
    try:
        payload = json.loads(base64.b64decode(encoded))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    # Two shapes exist in the wild: a flat object, or {claims: [{typ, val}]}.
    claims = payload.get("claims")
    if isinstance(claims, list):
        flat = {c.get("typ"): c.get("val") for c in claims if isinstance(c, dict)}
        # Map the long-form claim URIs onto the short names used above.
        return {
            "userPrincipalName": flat.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/upn"),
            "name": flat.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name"),
            "oid": flat.get("http://schemas.microsoft.com/identity/claims/objectidentifier"),
            "preferred_username": flat.get("preferred_username"),
        }
    return payload
