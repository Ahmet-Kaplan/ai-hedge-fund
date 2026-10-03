"""The congressional members registered as analysts.

The roster is a file in the repo, not a live query. Two reasons.

Importing a module must not make a network call: an unreachable vendor, an
unset key or a slow response would otherwise break `aihf --help`, the test
suite and the dashboard's import, none of which need congressional data.

And the set of analysts is part of the experiment. If the roster were
refetched on every run, an analyst could appear or vanish between two
backtests and their track records would not be comparable — the leaderboard
would silently be measuring a different population each time. Keeping it in
git makes a change to the roster a reviewable event with a date on it.

Refresh it with `aihf congress roster --top 50`, which ranks members by how
many disclosures they filed in the window the vendor exposes and rewrites
roster.json. Members are public officials and the filings are public record
under the STOCK Act; nothing here is private data.

The roster currently ships empty, and that is a judgement rather than an
oversight. FMP's entry plan serves page 0 only, at a limit of 25, and 402s
every per-member and per-symbol disclosure endpoint — re-probed 2026-10-02
and unchanged. Twenty-five rows per chamber is three or four days of
filings: a probe that day returned three senators and three representatives,
with a single tickered row in the whole Senate page. Registering those as
analysts would publish track records that measure which members happened to
file last week against a ninety-day lookback, and the membership would turn
over completely within the week, so nothing on the leaderboard would be
comparable to anything else. Lifting the plan's paging cap is the thing that
unblocks this; nothing in this file does.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hedge_fund.data.congress import CHAMBERS, Chamber
from hedge_fund.signals.base import AlphaModel
from hedge_fund.signals.congress import CongressModel, slug

logger = logging.getLogger(__name__)

ROSTER_PATH = Path(__file__).resolve().parent / "roster.json"


def load_roster(path: Path | None = None) -> list[tuple[str, Chamber]]:
    """Registered members as (name, chamber), or [] if the roster is absent.

    An absent or unreadable roster registers no congressional analysts and
    logs. It must not be fatal: every other part of the fund works without
    them, and a malformed file should cost the congress analysts, not the
    whole process.
    """
    path = path or ROSTER_PATH
    try:
        entries = json.loads(path.read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        logger.warning("congressional roster at %s is unreadable: %s", path, exc)
        return []

    members: list[tuple[str, Chamber]] = []
    for entry in entries:
        try:
            member, chamber = str(entry["member"]).strip(), entry["chamber"]
        except (KeyError, TypeError):
            logger.warning("skipping malformed roster entry: %r", entry)
            continue
        if member and chamber in CHAMBERS:
            members.append((member, chamber))
    return members


def member_model(member: str, chamber: Chamber) -> type[AlphaModel]:
    """A CongressModel subclass bound to one member.

    The registry instantiates with `cls(**params)` and the params come from
    a mandate, so the member cannot be passed that way without every mandate
    repeating it. Binding it into the class keeps mandates naming an analyst
    exactly as they do for any other.

    investment_approach is set on the subclass rather than left to be
    inherited because `get_investment_approach` reads `vars(cls)` and will
    not look up the chain.
    """
    def __init__(self, **params) -> None:  # noqa: N807 - bound constructor
        CongressModel.__init__(self, member, chamber, **params)

    return type(
        f"Congress_{slug(member).replace('-', '_')}",
        (CongressModel,),
        {
            "__init__": __init__,
            "investment_approach": CongressModel.investment_approach,
            "__doc__": f"Follows {member}'s disclosed trades ({chamber}).",
        },
    )


def registry_entries(path: Path | None = None) -> dict[str, type[AlphaModel]]:
    """Registry additions for every member on the roster."""
    return {slug(member): member_model(member, chamber)
            for member, chamber in load_roster(path)}
