import pytest

from hedge_fund.live.account.settings import LiveSettings, load_settings, save_settings


def test_defaults_are_safe():
    s = LiveSettings()
    assert (s.confirm_live, s.agent_share, s.core_ticker, s.shorts_enabled) == (False, 0.0, "SPY", False)


def test_round_trip(tmp_path):
    path = tmp_path / "live.yaml"
    save_settings(LiveSettings(confirm_live=True, agent_share=0.2), path)
    assert load_settings(path) == LiveSettings(confirm_live=True, agent_share=0.2)


def test_share_capped_and_unknown_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        LiveSettings(agent_share=0.6)
    path = tmp_path / "live.yaml"
    path.write_text("confirm_live: true\nleverage: 3\n")
    with pytest.raises(ValueError, match="leverage"):
        load_settings(path)


def test_missing_file_explains(tmp_path):
    with pytest.raises(ValueError, match="aihf-live init"):
        load_settings(tmp_path / "nope.yaml")


def test_crypto_defaults_and_validation():
    s = LiveSettings()
    assert s.crypto_share == 0.0
    assert s.crypto_core == {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1}
    with pytest.raises(ValueError):
        LiveSettings(crypto_share=0.6)
    with pytest.raises(ValueError, match="sum to 1"):
        LiveSettings(crypto_core={"BTC/USD": 0.5})
    with pytest.raises(ValueError, match="pair"):
        LiveSettings(crypto_core={"BTCUSD": 1.0})
