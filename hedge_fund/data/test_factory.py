import pytest

from hedge_fund.data import factory
from hedge_fund.data.cached import CachedDataClient
from hedge_fund.data.free import FreeDataClient


@pytest.fixture
def keys(monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "MARKET_DB_PATH", tmp_path / "m.db")
    for name, value in {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s", "SEC_USER_AGENT": "T t@example.org",
                        "FINANCIAL_DATASETS_API_KEY": "fd"}.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_free_is_the_default(keys):
    keys.delenv("HEDGE_FUND_DATA", raising=False)
    with factory.open_data_client() as client:
        assert isinstance(client, FreeDataClient)


def test_financial_datasets_on_request(keys):
    keys.setenv("HEDGE_FUND_DATA", "fd")
    with factory.open_data_client() as client:
        assert isinstance(client, CachedDataClient)


def test_unknown_source_raises(keys):
    keys.setenv("HEDGE_FUND_DATA", "bloomberg")
    with pytest.raises(ValueError, match="HEDGE_FUND_DATA"):
        with factory.open_data_client():
            pass


def test_missing_keys_depend_on_the_source(keys):
    keys.delenv("HEDGE_FUND_DATA", raising=False)
    keys.delenv("SEC_USER_AGENT")
    keys.delenv("FINANCIAL_DATASETS_API_KEY")
    assert factory.missing_data_keys() == ["SEC_USER_AGENT"]
    keys.setenv("HEDGE_FUND_DATA", "fd")
    assert factory.missing_data_keys() == ["FINANCIAL_DATASETS_API_KEY"]
