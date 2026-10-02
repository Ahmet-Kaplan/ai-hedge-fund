"""Congressional stock disclosures from Financial Modeling Prep.

Members of Congress file periodic transaction reports under the STOCK Act.
The filings are public record; the API here is just a convenient reading of
them.

The entire reason this module exists as its own thing, rather than a few
lines inside an alpha model, is the gap between two dates that both appear
on every filing:

    transactionDate   when the member traded
    disclosureDate    when anyone else could know about it

The STOCK Act allows up to 45 days between them. A backtest that keys on
transactionDate is therefore trading on information that did not exist yet,
and it will produce a spectacular track record that could never have been
earned. Every read path in this module is keyed on disclosureDate for that
reason, and `Disclosure.known_by` exists so callers cannot accidentally
reach for the wrong one.

Parsing is strict. A silent schema change at the vendor would otherwise turn
into an analyst who quietly abstains on everything, which looks like a thin
market rather than a broken feed — so a row missing a field this module
depends on raises instead of being skipped.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

import requests

logger = logging.getLogger(__name__)

Chamber = Literal["house", "senate"]
CHAMBERS: tuple[Chamber, ...] = ("house", "senate")

# Disclosed amounts are filed as ranges, never exact figures. The midpoint is
# the honest summary of a range, and the ranges are wide, so nothing
# downstream should treat these as precise.
_AMOUNT = re.compile(r"\$?([\d,]+)")


class CongressDataError(Exception):
    """A disclosures request failed, or returned something unreadable.

    Distinct from "this member disclosed nothing", which is an empty list. A
    run must fail on this rather than mistake a broken feed for a quiet one.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Disclosure:
    """One disclosed transaction, as filed."""

    member: str
    chamber: Chamber
    ticker: str
    # Both dates are kept because the difference between them is the point.
    transaction_date: str
    disclosure_date: str
    kind: str                 # as filed: "Purchase", "Sale (Full)", "Sale (Partial)", ...
    amount_low: float
    amount_high: float
    district: str | None = None
    owner: str | None = None

    @property
    def known_by(self) -> str:
        """The first date this filing could have informed a decision."""
        return self.disclosure_date

    @property
    def amount_mid(self) -> float:
        return (self.amount_low + self.amount_high) / 2.0

    @property
    def direction(self) -> int:
        """+1 for a purchase, -1 for a sale, 0 for anything else.

        Exchanges and transfers are deliberately 0: they move an asset
        without expressing a view, and scoring them as conviction would read
        intent into bookkeeping.
        """
        kind = self.kind.lower()
        if "purchase" in kind:
            return 1
        if "sale" in kind or "sold" in kind:
            return -1
        return 0


class CongressClient:
    """Reads congressional disclosures from Financial Modeling Prep.

    The key is read from the environment, which on the deployed fund is
    populated from Key Vault by the container platform. It is never a
    constructor default and never a literal.

    Usage::

        with CongressClient() as congress:
            filings = congress.latest("senate", pages=3)
    """

    BASE_URL = "https://financialmodelingprep.com/stable"
    PAGE_SIZE = 100
    _RETRY_DELAYS = (5, 15, 30)

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self._api_key = api_key or os.environ.get("FMP_API_KEY", "")
        self._timeout = timeout
        self._session = requests.Session()

    def __enter__(self) -> CongressClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def latest(self, chamber: Chamber, *, pages: int = 1) -> list[Disclosure]:
        """Most recent filings for *chamber*, newest first.

        The endpoint is ordered by disclosure, so paging backwards walks
        further into the past. Paging stops early on a short page rather
        than requesting beyond the end of the data.
        """
        if chamber not in CHAMBERS:
            raise ValueError(f"unknown chamber {chamber!r}; expected one of {CHAMBERS}")

        filings: list[Disclosure] = []
        for page in range(pages):
            rows = self._get(f"/{chamber}-latest", {"page": page, "limit": self.PAGE_SIZE})
            filings.extend(parse_disclosure(row, chamber) for row in rows)
            if len(rows) < self.PAGE_SIZE:
                break
        return filings

    def _get(self, path: str, params: dict) -> list[dict]:
        if not self._api_key:
            raise CongressDataError(
                "FMP_API_KEY is not set; congressional disclosures are unavailable"
            )

        url = f"{self.BASE_URL}{path}"
        # The key travels as a query parameter because the vendor accepts no
        # other form. It is kept out of logs for exactly that reason: the
        # full URL is a credential, so only the path is ever logged.
        response = self._session.get(
            url, params={**params, "apikey": self._api_key}, timeout=self._timeout,
        )
        if response.status_code != 200:
            raise CongressDataError(
                f"GET {path} returned {response.status_code}",
                status_code=response.status_code,
            )

        payload = response.json()
        if not isinstance(payload, list):
            # FMP reports quota and auth failures as a 200 with an object
            # body, so a non-list here is an error wearing a success code.
            raise CongressDataError(f"GET {path} returned {type(payload).__name__}, expected a list")
        return payload


def parse_disclosure(row: dict, chamber: Chamber) -> Disclosure:
    """Build a Disclosure from one API row, or raise if it cannot be trusted.

    Raising beats skipping. A row quietly dropped for a renamed field turns
    a vendor schema change into an analyst who abstains on everything, and
    that reads as a quiet market rather than a broken integration.
    """
    try:
        ticker = str(row["symbol"]).strip().upper()
        disclosure_date = str(row["disclosureDate"]).strip()
        transaction_date = str(row["transactionDate"]).strip()
        kind = str(row["type"]).strip()
        member = " ".join(str(row[part]).strip() for part in ("firstName", "lastName")).strip()
    except KeyError as exc:
        raise CongressDataError(
            f"disclosure row is missing {exc.args[0]!r}; the vendor schema may have changed"
        ) from exc

    if not ticker or not disclosure_date:
        raise CongressDataError("disclosure row has no symbol or no disclosure date")

    low, high = parse_amount(row.get("amount"))
    return Disclosure(
        member=member,
        chamber=chamber,
        ticker=ticker,
        transaction_date=transaction_date,
        disclosure_date=disclosure_date,
        kind=kind,
        amount_low=low,
        amount_high=high,
        district=(str(row["district"]).strip() or None) if row.get("district") else None,
        owner=(str(row["owner"]).strip() or None) if row.get("owner") else None,
    )


def parse_amount(amount: str | None) -> tuple[float, float]:
    """Read a filed range like "$15,001 - $50,000" into its bounds.

    An unparseable or absent amount is (0, 0) rather than an error: the
    direction of a trade is disclosed reliably and the size is not, so a
    filing with a malformed range still carries its most useful information.
    """
    if not amount:
        return 0.0, 0.0
    figures = [float(match.replace(",", "")) for match in _AMOUNT.findall(str(amount))]
    if not figures:
        return 0.0, 0.0
    return min(figures), max(figures)
