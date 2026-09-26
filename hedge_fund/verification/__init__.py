"""Data-assumption verification — checks that can invalidate a backtest.

Pure judges live in `checks`; the live runner in `runner`; the CLI is
``python -m hedge_fund.verification``.
"""

from hedge_fund.verification.checks import CheckResult
from hedge_fund.verification.runner import required_env, run_checks

__all__ = ["CheckResult", "required_env", "run_checks"]
