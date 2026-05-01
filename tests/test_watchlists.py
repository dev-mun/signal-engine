import json

import pytest

import algo_backtester.watchlists as watchlists_module


def test_builtin_profiles_load():
    broad_market = watchlists_module.get_watchlist("broad_market")
    high_beta = watchlists_module.get_watchlist("high_beta")
    custom = watchlists_module.get_watchlist("custom")

    assert broad_market[:6] == ["SPY", "QQQ", "DIA", "IWM", "XLK", "SMH"]
    assert high_beta == ["NVDA", "AVGO", "META", "MSFT", "DIA", "PANW", "TSM", "GOOGL", "SPY"]
    assert custom == []


def test_strategy_default_mapping_works():
    assert watchlists_module.get_default_watchlist_for_strategy("ema-rsi") == watchlists_module.get_watchlist(
        "broad_market"
    )
    assert watchlists_module.get_default_watchlist_for_strategy(
        "four-hour-trend"
    ) == watchlists_module.get_watchlist("broad_market")
    assert watchlists_module.get_default_watchlist_for_strategy(
        "rsi-bollinger-v2"
    ) == watchlists_module.get_watchlist("high_beta")


def test_explicit_tickers_override_profile():
    resolved = watchlists_module.parse_scan_universe(
        scan_arg="NVDA,AVGO,META",
        strategy="rsi-bollinger-v2",
        profile="broad_market",
    )

    assert resolved == ["NVDA", "AVGO", "META"]


def test_explicit_profile_overrides_strategy_default():
    resolved = watchlists_module.parse_scan_universe(
        scan_arg="",
        strategy="ema-rsi",
        profile="high_beta",
    )

    assert resolved == watchlists_module.get_watchlist("high_beta")


def test_custom_config_file_merge_works(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "watchlists.json"
    config_path.write_text(
        json.dumps(
            {
                "custom": ["TSLA", "AMD", "SMCI"],
                "high_beta": ["NVDA", "AVGO", "META", "TSLA"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(watchlists_module, "DEFAULT_CONFIG_PATH", config_path)

    assert watchlists_module.get_watchlist("custom") == ["TSLA", "AMD", "SMCI"]
    assert watchlists_module.get_watchlist("high_beta") == ["NVDA", "AVGO", "META", "TSLA"]
    assert "SPY" in watchlists_module.get_watchlist("broad_market")


def test_missing_config_file_falls_back_cleanly(tmp_path, monkeypatch):
    config_path = tmp_path / "config" / "watchlists.json"
    monkeypatch.setattr(watchlists_module, "DEFAULT_CONFIG_PATH", config_path)

    assert watchlists_module.get_watchlist("custom") == []
    assert watchlists_module.get_watchlist("high_beta") == [
        "NVDA",
        "AVGO",
        "META",
        "MSFT",
        "DIA",
        "PANW",
        "TSM",
        "GOOGL",
        "SPY",
    ]


def test_invalid_profile_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown watchlist profile 'not_real'"):
        watchlists_module.get_watchlist("not_real")
