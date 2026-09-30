"""Per-class cover-page share counts from a filing's rendered cover (R1.htm).

SEC's companyfacts API omits dimensioned facts, so a registrant that reports
shares outstanding per class (Berkshire A/B) has no usable total there. The
filing's own rendered cover page lists each class with its count; it is
small, immutable once filed, and fetched at most once per filing.
"""

from __future__ import annotations

import html
import re

_LABEL = "Entity Common Stock, Shares Outstanding"
_MEMBER = re.compile(r"([^\[\]]{1,80}?)\s*\[Member\]")
_CLASS = re.compile(r"\bClass\s+([A-Z])\b")
_NUMBER = re.compile(r"^\s*([\d,]+)")


def _text(document: str) -> str:
    text = re.sub(r"<[^>]+>", " ", document)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


def parse_cover_shares(document: str) -> dict[str | None, float]:
    """{class letter (or member label, or None if entity-wide): shares}."""
    text = _text(document)
    out: dict[str | None, float] = {}
    cursor = 0
    while True:
        i = text.find(_LABEL, cursor)
        if i < 0:
            break
        after = text[i + len(_LABEL):]
        number = _NUMBER.match(after)
        segment = text[cursor:i]
        cursor = i + len(_LABEL)
        if not number:
            continue
        members = _MEMBER.findall(segment)
        key: str | None = None
        if members:
            label = members[-1].strip()
            letters = _CLASS.findall(label)  # last one: nearest the [Member] tag
            key = letters[-1] if letters else label
        out[key] = out.get(key, 0.0) + float(number.group(1).replace(",", ""))
    return out
