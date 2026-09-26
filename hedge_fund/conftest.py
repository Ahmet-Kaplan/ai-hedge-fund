"""Shared pytest setup for hedge_fund.

Loads a local .env if present so optional live tests can see
FINANCIAL_DATASETS_API_KEY. CI and the offline smoke path run without
keys; tests marked ``live`` skip when the variable is unset.

User data is redirected to a throwaway directory *before* any ``hedge_fund``
module is imported, because ``hedge_fund.paths`` resolves its constants at
import time. Without this a test run would write mandates, receipts, caches,
order journals and tick keys into the developer's real ``~/.hedge-fund``
(and fail outright where that directory is not writable).
"""

import os
import tempfile

os.environ.setdefault(
    "HEDGE_FUND_HOME", tempfile.mkdtemp(prefix="hedge-fund-tests-")
)

from dotenv import load_dotenv  # noqa: E402  (must follow the redirect above)

load_dotenv()
