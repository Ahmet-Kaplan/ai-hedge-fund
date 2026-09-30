"""EdgarClient HTTP behaviour with a fake session — no network."""

import os

import pytest
import requests

from hedge_fund.data.edgar import client as client_mod
from hedge_fund.data.edgar import EdgarClient, EdgarClientError, EdgarConfigError, validate_user_agent

UA = "Test Runner test@example.com"


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _Session:
    """Stands in for requests.Session; pops scripted responses."""

    instances: list = []

    def __init__(self):
        self.headers = {}
        self.urls = []
        self.script = []
        _Session.instances.append(self)

    def get(self, url, timeout=None):
        self.urls.append(url)
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def close(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    _Session.instances = []
    monkeypatch.setattr(client_mod.requests, "Session", _Session)
    monkeypatch.setattr(EdgarClient, "_RETRY_DELAYS", (0, 0, 0))
    monkeypatch.setattr(EdgarClient, "_MIN_INTERVAL", 0.0)
    return _Session


def _client(tmp_path, **kwargs):
    kwargs.setdefault("user_agent", UA)
    return EdgarClient(cache_dir=tmp_path, **kwargs)


def _script(client, *responses):
    """Create the session through the real path, then script it."""
    client._session = None
    session = _Session()
    session.headers.update({"User-Agent": validate_user_agent(client._user_agent)})
    session.script = list(responses)
    client._session = session
    return session


@pytest.mark.parametrize("value", [None, "", "   ", "no-contact", "a@b"])
def test_user_agent_must_carry_contact(value):
    with pytest.raises(EdgarConfigError, match="SEC_USER_AGENT"):
        validate_user_agent(value)


def test_missing_user_agent_fails_before_any_request(tmp_path, fake, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    c = EdgarClient(cache_dir=tmp_path)
    with pytest.raises(EdgarConfigError):
        c.get_financial_metrics("KO", "2020-01-01")
    assert fake.instances == []  # no session was ever opened


def test_user_agent_from_environment_is_sent(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    c = EdgarClient(cache_dir=tmp_path)
    with pytest.raises(IndexError):  # first request reaches the (empty) fake session
        c._companyfacts(21344)
    session = fake.instances[0]
    assert session.headers["User-Agent"] == UA
    assert session.urls == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000021344.json"]


def test_404_is_no_data_and_is_cached(tmp_path, fake):
    c = _client(tmp_path)
    s = _script(c, _Resp(404))
    assert c._companyfacts(123) is None
    assert c._companyfacts(123) is None
    assert len(s.urls) == 1


def test_403_explains_sec_policy(tmp_path, fake):
    c = _client(tmp_path)
    _script(c, _Resp(403, text="Request Rate Threshold Exceeded"))
    with pytest.raises(EdgarClientError, match="SEC_USER_AGENT") as exc:
        c._companyfacts(123)
    assert exc.value.status_code == 403


def test_retries_throttling_then_succeeds(tmp_path, fake):
    c = _client(tmp_path)
    doc = {"cik": 7, "entityName": "X", "facts": {"us-gaap": {}}}
    s = _script(c, _Resp(429), _Resp(503), _Resp(200, payload=doc))
    assert c._companyfacts(7)["entityName"] == "X"
    assert len(s.urls) == 3


def test_retries_exhausted_raise(tmp_path, fake):
    c = _client(tmp_path)
    _script(c, *[_Resp(503)] * 4)
    with pytest.raises(EdgarClientError, match="retries"):
        c._companyfacts(7)


def test_network_error_raises(tmp_path, fake):
    c = _client(tmp_path)
    _script(c, requests.ConnectionError("boom"))
    with pytest.raises(EdgarClientError, match="boom"):
        c._companyfacts(7)


def test_cache_is_trimmed_and_refreshed_when_stale(tmp_path, fake):
    raw = {"cik": 7, "entityName": "X", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"start": "2020-01-01", "end": "2020-12-31", "val": 5, "accn": "a", "form": "10-K",
             "filed": "2021-02-01", "frame": "CY2020"},
            {"start": "2020-01-01", "end": "2020-12-31", "val": 5, "accn": "b", "form": "8-K", "filed": "2021-01-20"},
        ]}},
        "SomethingUnused": {"units": {"USD": []}},
    }}}
    c = _client(tmp_path, max_age_hours=0)
    s = _script(c, _Resp(200, payload=raw), _Resp(200, payload=raw))
    doc = c._companyfacts(7)
    assert list(doc["facts"]["us-gaap"]) == ["Revenues"]
    assert doc["facts"]["us-gaap"]["Revenues"]["units"]["USD"] == [  # 8-K row and "frame" dropped
        {"start": "2020-01-01", "end": "2020-12-31", "val": 5, "accn": "a", "form": "10-K", "filed": "2021-02-01"}]
    c._companyfacts(7)  # max_age 0: stale immediately
    assert len(s.urls) == 2
    fresh = _client(tmp_path, max_age_hours=24)
    fresh._companyfacts(7)
    assert fresh._session is None  # served from disk


def test_cover_page_fetched_once_ever(tmp_path, fake):
    c = _client(tmp_path, max_age_hours=0)
    html = "<td>Class A [Member]</td><td>Entity Common Stock, Shares Outstanding</td><td>10</td>" \
           "<td>Class B [Member]</td><td>Entity Common Stock, Shares Outstanding</td><td>20</td>"
    s = _script(c, _Resp(200, text=html))
    assert c._cover_page_shares(1067983, "0001-23-000001") == {"A": 10.0, "B": 20.0}
    assert c._cover_page_shares(1067983, "0001-23-000001") == {"A": 10.0, "B": 20.0}
    assert s.urls == ["https://www.sec.gov/Archives/edgar/data/1067983/000123000001/R1.htm"]


def test_offline_never_touches_network(tmp_path, fake):
    c = _client(tmp_path, offline=True)
    with pytest.raises(EdgarClientError, match="offline"):
        c._companyfacts(7)
    assert c._cover_page_shares(1067983, "x") is None
    assert fake.instances == []


@pytest.mark.skipif(not (os.environ.get("SEC_USER_AGENT") and os.environ.get("EDGAR_LIVE_TESTS")),
                    reason="live SEC test: set SEC_USER_AGENT and EDGAR_LIVE_TESTS=1")
def test_live_single_company(tmp_path):
    """One companyfacts + one submissions request."""
    with EdgarClient(cache_dir=tmp_path) as c:
        rows = c.get_financial_metrics("KO", "2020-02-28", limit=4)
        assert rows[0].report_period == "2019-12-31"
        assert c.requests <= 3
