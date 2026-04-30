import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from algo_backtester.backtester import TrendPullbackBacktester
from algo_backtester.backtests.ema_rsi_backtester import EmaRsiPullbackBacktester
import algo_backtester.backtests.four_hour_trend_backtester as four_hour_module
from algo_backtester.backtests.four_hour_trend_backtester import (
    FourHourTrendBacktester,
    make_demo_intraday_data,
    prepare_four_hour_data,
    resample_to_four_hour,
)
import algo_backtester.data_loader as data_loader_module
from algo_backtester.data_loader import load_yfinance_intraday_data, normalize_intraday_index, resample_to_4h
from algo_backtester.reports.four_hour_report import latest_signal, save_reports
from algo_backtester.strategies.four_hour_trend_pullback import add_indicators


def test_resample_one_hour_to_four_hour_ohlcv():
    dates = pd.date_range("2026-01-01 09:00:00", periods=8, freq="h")
    raw_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [1, 2, 3, 4, 5, 6, 7, 8],
            "High": [2, 3, 4, 5, 6, 7, 8, 9],
            "Low": [0, 1, 2, 3, 4, 5, 6, 7],
            "Close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
            "Volume": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )

    df = resample_to_four_hour(raw_df)

    assert len(df) == 2
    assert df.iloc[0]["Open"] == 1
    assert df.iloc[0]["High"] == 5
    assert df.iloc[0]["Low"] == 0
    assert df.iloc[0]["Close"] == 4.5
    assert df.iloc[0]["Volume"] == 100


def test_intraday_loader_returns_expected_schema(monkeypatch):
    raw_df = make_demo_intraday_data(rows=20).set_index("Date")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *args, **kwargs: raw_df.copy()))

    df = load_yfinance_intraday_data(ticker="SPY")

    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.is_monotonic_increasing
    assert df.index.tz is None


def test_intraday_loader_handles_empty_after_retry(monkeypatch):
    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))

    df = load_yfinance_intraday_data(ticker="SPY")

    assert df.empty
    assert calls["count"] == 2


def test_timezone_normalization_keeps_unique_sorted_index():
    dates = pd.date_range("2026-01-01 09:00:00", periods=4, freq="h", tz="America/New_York")
    raw_df = pd.DataFrame(
        {
            "Date": [dates[1], dates[0], dates[1], dates[3]],
            "Open": [2, 1, 2.2, 4],
            "High": [3, 2, 3.2, 5],
            "Low": [1, 0, 1.2, 3],
            "Close": [2.5, 1.5, 2.7, 4.5],
            "Volume": [20, 10, 22, 40],
        }
    )

    df = normalize_intraday_index(raw_df)

    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert df.index.tz is None


def test_resample_to_4h_alias_matches_shared_helper():
    raw_df = make_demo_intraday_data(rows=24)

    assert resample_to_four_hour(raw_df).equals(resample_to_4h(raw_df))


def test_four_hour_indicators_are_created():
    raw_df = make_demo_intraday_data(rows=1400)
    df = resample_to_four_hour(raw_df)
    df = add_indicators(df)

    assert not df.empty
    assert "EMA20" in df.columns
    assert "EMA50" in df.columns
    assert "EMA200" in df.columns
    assert "RSI14" in df.columns
    assert "ATR14" in df.columns
    assert "AVG_VOL20" in df.columns


def test_four_hour_backtest_returns_expected_frames():
    raw_df = make_demo_intraday_data(rows=1400)
    bt = FourHourTrendBacktester()

    _, equity_df, trades_df, signals_df = bt.run(raw_df)

    assert not equity_df.empty
    assert not signals_df.empty
    assert "Equity" in equity_df.columns
    assert "Signal" in signals_df.columns
    assert trades_df is not None


def test_four_hour_latest_signal_has_expected_fields():
    raw_df = make_demo_intraday_data(rows=1400)
    bt = FourHourTrendBacktester()

    _, _, _, signals_df = bt.run(raw_df)
    signal = latest_signal(signals_df)

    assert "Signal" in signal
    assert "Reason" in signal
    assert "Close" in signal
    assert "RSI14" in signal
    assert signal["Signal"] in {"BUY", "SELL", "SHORT_SETUP", "HOLD", "HOLD_POSITION"}


def test_four_hour_scan_result_includes_strategy(monkeypatch):
    raw_df = make_demo_intraday_data(rows=1400)

    monkeypatch.setattr(four_hour_module, "prepare_four_hour_data", lambda **kwargs: raw_df)

    result = four_hour_module.scan_ticker(ticker="SPY")

    assert result["Ticker"] == "SPY"
    assert result["Strategy"] == "four-hour-trend"
    assert "Setup" in result
    assert "Reason" in result


def test_existing_strategy_imports_still_work():
    raw_df = make_demo_intraday_data(rows=1400)

    daily_demo = data_loader_module.make_demo_data(rows=500)
    _, trend_equity_df, _, _ = TrendPullbackBacktester().run(daily_demo)
    _, ema_equity_df, _, _ = EmaRsiPullbackBacktester().run(daily_demo)

    assert not trend_equity_df.empty
    assert not ema_equity_df.empty
    assert raw_df is not None


def test_four_hour_strategy_uses_intraday_path(monkeypatch):
    raw_intraday = make_demo_intraday_data(rows=1400)
    calls = {"intraday": 0}

    def fake_loader(**kwargs):
        calls["intraday"] += 1
        return raw_intraday.set_index("Date")

    monkeypatch.setattr(four_hour_module, "load_yfinance_intraday_data", fake_loader)
    prepared = prepare_four_hour_data(ticker="SPY", interval="1h")

    assert calls["intraday"] == 1
    assert not prepared.empty


def test_existing_daily_loader_path_unchanged(monkeypatch):
    daily_df = data_loader_module.make_demo_data(rows=300).set_index("Date")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *args, **kwargs: daily_df.copy()))

    df = data_loader_module.load_yfinance_data(ticker="SPY", start="2020-01-01")

    assert not df.empty
    assert "Close" in df.columns


def test_four_hour_reports_save_to_separate_directory(tmp_path: Path):
    raw_df = make_demo_intraday_data(rows=1400)
    bt = FourHourTrendBacktester()
    _, equity_df, trades_df, signals_df = bt.run(raw_df)

    save_reports(
        label="SPY",
        equity_df=equity_df,
        trades_df=trades_df,
        signals_df=signals_df,
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "four_hour" / "SPY_equity.csv").exists()
    assert (tmp_path / "four_hour" / "SPY_trades.csv").exists()
    assert not (tmp_path / "SPY_equity.csv").exists()
