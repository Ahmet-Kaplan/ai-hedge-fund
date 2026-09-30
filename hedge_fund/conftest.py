"""Test session setup.

Loads .env for optional live keys. Financial Datasets stays blocked regardless:
the project policy (hedge_fund/data/policy.py) requires the explicit opt-in
AIHF_ALLOW_FINANCIAL_DATASETS=1, which the test suite never sets globally.
"""

import os

from dotenv import load_dotenv

load_dotenv()
os.environ.pop("AIHF_ALLOW_FINANCIAL_DATASETS", None)
