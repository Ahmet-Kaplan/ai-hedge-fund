"""Financial Datasets stays unreachable unless the project opts in explicitly."""

from __future__ import annotations

import pytest

from hedge_fund.data import client as client_mod
from hedge_fund.data.client import FDClient
from hedge_fund.data.factory import make_data_client
from hedge_fund.data.policy import (
    POLICY_ENV,
    FinancialDatasetsDisabled,
    financial_datasets_allowed,
    require_financial_datasets,
)

SECRET = "fd-secret-value-must-never-appear-0123456789"


class _NetworkTrap:
    """A requests.Session stand-in that records any attempted request."""

    calls: list = []

    def __init__(self):
        self.headers = {}

    def request(self, *a, **kw):
        _NetworkTrap.calls.append((a, kw))
        raise AssertionError("network request attempted")

    def close(self):
        pass


@pytest.fixture(autouse=True)
def trap(monkeypatch):
    _NetworkTrap.calls = []
    monkeypatch.setattr(client_mod.requests, "Session", _NetworkTrap)
    monkeypatch.setenv("FINANCIAL_DATASETS_API_KEY", SECRET)   # present "by accident"
    monkeypatch.delenv(POLICY_ENV, raising=False)
    monkeypatch.delenv("HEDGE_FUND_DATA_PROVIDER", raising=False)
    monkeypatch.delenv("HEDGE_FUND_DATA_SUPPLEMENT", raising=False)
    yield
    assert _NetworkTrap.calls == []


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", "TRUE", " 2 "])
def test_disabled_unless_exactly_one(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(POLICY_ENV, value)
    assert not financial_datasets_allowed()
    with pytest.raises(FinancialDatasetsDisabled):
        require_financial_datasets()


def test_key_present_is_not_enough_to_build_a_client():
    with pytest.raises(FinancialDatasetsDisabled) as exc:
        FDClient()
    assert SECRET not in str(exc.value)


def test_a_client_built_while_allowed_cannot_request_after_opt_out(monkeypatch):
    monkeypatch.setenv(POLICY_ENV, "1")
    fd = FDClient()
    monkeypatch.delenv(POLICY_ENV)
    with pytest.raises(FinancialDatasetsDisabled):
        fd.get_prices("AAPL", "2024-01-01", "2024-01-31")
    with pytest.raises(FinancialDatasetsDisabled):
        fd.get_earnings("AAPL")


@pytest.mark.parametrize("var", ["HEDGE_FUND_DATA_PROVIDER", "HEDGE_FUND_DATA_SUPPLEMENT"])
def test_factory_refuses_financial_datasets_routes(monkeypatch, var):
    monkeypatch.setenv(var, "financial-datasets")
    monkeypatch.setenv("TIINGO_API_KEY", "t")
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    with pytest.raises(FinancialDatasetsDisabled) as exc:
        make_data_client()
    assert SECRET not in str(exc.value)


def test_default_factory_has_no_financial_datasets_supplement(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "t")
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    with make_data_client() as fd:
        assert fd.supplemental is None
        assert "financial_datasets" not in fd.request_counts()
        with pytest.raises(RuntimeError):
            fd.get_news("AAPL", "2024-01-31")
