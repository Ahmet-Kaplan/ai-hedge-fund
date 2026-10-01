"""The agent-share review: evidence from the paper fund (and the live satellite), proposed, then applied by hand."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.settings import LiveSettings, save_settings
from hedge_fund.live.ledger import Ledger
from hedge_fund.live.report import MIN_REBALANCES_FOR_VERDICT

FIRST_SHARE, STEP, CAP = 0.2, 0.1, 0.5


class ReviewResult(BaseModel):
    window_start: str
    window_end: str
    paper_rebalances: int
    paper_excess: float              # paper fund return − benchmark return over the window
    live_excess: float | None        # satellite − core, if a satellite was held
    passed: bool
    old_share: float
    new_share: float


def propose_review(paper: Ledger, live: LiveLedger, settings: LiveSettings) -> ReviewResult:
    sessions = paper.plan_sessions()
    if not sessions:
        raise ValueError("the paper fund has no rebalances yet")
    start = live.last_review_date() or sessions[0]
    rebalances = [s for s in sessions if s >= start]
    if len(rebalances) < MIN_REBALANCES_FOR_VERDICT:
        raise ValueError(f"{len(rebalances)} of {MIN_REBALANCES_FOR_VERDICT} paper rebalances since {start}; review not due")
    rows = [r for r in paper.nav_rows() if r.date >= start]
    if len(rows) < 2:
        raise ValueError(f"the paper fund has fewer than two reconciled sessions since {start}")
    paper_excess = rows[-1].equity / rows[0].equity - rows[-1].benchmark_close / rows[0].benchmark_close
    live_excess = live.satellite_excess(since=start)
    passed = paper_excess > 0 and (live_excess is None or live_excess > 0)
    old = settings.agent_share
    if passed:
        new = FIRST_SHARE if old == 0 else min(CAP, old + STEP)
    else:
        new = max(0.0, old - STEP)
    return ReviewResult(window_start=start, window_end=rows[-1].date, paper_rebalances=len(rebalances),
                        paper_excess=round(paper_excess, 6), live_excess=None if live_excess is None else round(live_excess, 6),
                        passed=passed, old_share=old, new_share=round(new, 6))


def apply_review(result: ReviewResult, settings: LiveSettings, settings_path: Path, live: LiveLedger, *, today: str) -> None:
    save_settings(settings.model_copy(update={"agent_share": result.new_share}), settings_path)
    live.append_review(today, result.old_share, result.new_share, paper_excess=result.paper_excess,
                       live_excess=result.live_excess, passed=result.passed)
    live.clear_satellite_halt()
