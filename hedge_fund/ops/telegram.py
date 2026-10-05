"""Send the daily summary to a Telegram chat — free, one HTTPS request.

Credentials: TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID
(`aihf-daily --find-chat-id` lists it after you message the bot). Missing
credentials skip the notification; a failed send never fails the run. The
token is part of the request URL, so it is redacted from every error printed.
"""

from __future__ import annotations

import os
import sys

import requests

API = "https://api.telegram.org"
MAX_LENGTH = 4096


def send(text: str, *, token: str | None = None, chat_id: str | None = None,
         session=None, timeout: float = 20.0) -> bool:
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to get notifications", file=sys.stderr)
        return False
    http = session or requests
    try:
        resp = http.post(f"{API}/bot{token}/sendMessage", timeout=timeout,
                         json={"chat_id": chat_id, "text": text[:MAX_LENGTH], "disable_web_page_preview": True})
    except requests.RequestException as exc:
        print(f"telegram: send failed: {_redact(str(exc), token)}", file=sys.stderr)
        return False
    if resp.status_code != 200:
        print(f"telegram: HTTP {resp.status_code}: {_redact(resp.text[:200], token)}", file=sys.stderr)
        return False
    return True


def chat_ids(*, token: str | None = None, session=None, timeout: float = 20.0) -> list[tuple[str, str]]:
    """(chat id, name) for every chat that has messaged the bot recently."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("set TELEGRAM_BOT_TOKEN first")
    http = session or requests
    try:
        data = http.get(f"{API}/bot{token}/getUpdates", timeout=timeout).json()
    except requests.RequestException as exc:
        raise RuntimeError(f"telegram: {_redact(str(exc), token)}") from None
    seen: dict[str, str] = {}
    for update in data.get("result", []):
        chat = (update.get("message") or update.get("channel_post") or {}).get("chat") or {}
        if "id" in chat:
            seen.setdefault(str(chat["id"]), chat.get("title") or chat.get("first_name") or chat.get("username") or "")
    return list(seen.items())


def _redact(text: str, token: str) -> str:
    return text.replace(token, "<token>") if token else text
