"""Telegram sender: fake session, no network."""

import requests

from hedge_fund.ops.telegram import chat_ids, send


class FakeResponse:
    def __init__(self, status=200, payload=None, text="ok"):
        self.status_code, self._payload, self.text = status, payload or {}, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.calls, self._response, self._error = [], response, error

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if self._error:
            raise self._error
        return self._response

    def get(self, url, timeout=None):
        self.calls.append((url, None))
        return self._response


def test_send_posts_the_message():
    session = FakeSession(FakeResponse())
    assert send("hello", token="T0K", chat_id="42", session=session) is True
    url, body = session.calls[0]
    assert url == "https://api.telegram.org/botT0K/sendMessage"
    assert body == {"chat_id": "42", "text": "hello", "disable_web_page_preview": True}


def test_missing_credentials_skip_quietly(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert send("hi", session=FakeSession()) is False
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err


def test_errors_never_print_the_token(capsys):
    boom = requests.ConnectionError("HTTPSConnectionPool: /botSECRET123/sendMessage failed")
    assert send("hi", token="SECRET123", chat_id="1", session=FakeSession(error=boom)) is False
    err = capsys.readouterr().err
    assert "SECRET123" not in err and "<token>" in err
    assert send("hi", token="SECRET123", chat_id="1", session=FakeSession(FakeResponse(401, text="bad SECRET123"))) is False
    assert "SECRET123" not in capsys.readouterr().err


def test_long_messages_are_cut_to_telegrams_limit():
    session = FakeSession(FakeResponse())
    send("x" * 5000, token="t", chat_id="1", session=session)
    assert len(session.calls[0][1]["text"]) == 4096


def test_chat_ids_from_updates():
    updates = {"ok": True, "result": [
        {"message": {"chat": {"id": 42, "first_name": "Vivek"}}},
        {"message": {"chat": {"id": 42, "first_name": "Vivek"}}},
        {"message": {"chat": {"id": -100, "title": "Group"}}},
    ]}
    assert chat_ids(token="t", session=FakeSession(FakeResponse(payload=updates))) == [("42", "Vivek"), ("-100", "Group")]
