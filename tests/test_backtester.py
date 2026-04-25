import pandas as pd
import pytest

import src.algo_backtester.backtester as backtester_module
from src.algo_backtester.backtester import TrendPullbackBacktester
from src.algo_backtester.data_loader import make_demo_data, validate_ohlcv
from src.algo_backtester.metrics import performance_summary
from src.algo_backtester.options_engine import recommend_options_trade
from src.algo_backtester.reporting import latest_signal


def test_backtest_runs_with_demo_data():
    raw_df = make_demo_data(rows=500)
    bt = TrendPullbackBacktester()

    _, equity_df, trades_df, signals_df = bt.run(raw_df)

    assert not equity_df.empty
    assert not signals_df.empty
    assert "Equity" in equity_df.columns
    assert "Signal" in signals_df.columns
    assert equity_df["Equity"].iloc[-1] > 0

    summary = performance_summary(equity_df, trades_df)
    assert "Sharpe Ratio" in summary
    assert "Max Drawdown %" in summary


def test_latest_signal_has_expected_fields():
    raw_df = make_demo_data(rows=500)
    bt = TrendPullbackBacktester()

    _, _, _, signals_df = bt.run(raw_df)
    signal = latest_signal(signals_df)

    assert "Signal" in signal
    assert "Reason" in signal
    assert "Close" in signal
    assert signal["Signal"] in {
        "BUY",
        "BEARISH_ENTRY",
        "EXIT_LONG",
        "HOLD",
        "HOLD_POSITION",
    }


def test_validate_rejects_missing_columns():
    invalid_df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=250),
            "Close": range(250),
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_ohlcv(invalid_df)


def test_backtest_uses_next_open_fills_and_stop_based_position_sizing(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [99.0, 100.0, 110.0, 112.0, 90.0, 88.0],
            "High": [100.0, 103.0, 111.0, 113.0, 91.0, 89.0],
            "Low": [98.0, 99.0, 109.0, 111.0, 89.0, 87.0],
            "Close": [99.0, 102.0, 111.0, 112.0, 90.0, 88.0],
            "Volume": [100.0, 200.0, 150.0, 150.0, 150.0, 150.0],
            "EMA20": [100.0, 101.0, 105.0, 108.0, 104.0, 100.0],
            "EMA50": [101.0, 104.0, 106.0, 108.0, 107.0, 105.0],
            "EMA200": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "RSI": [35.0, 50.0, 65.0, 65.0, 45.0, 45.0],
            "ATR": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "AVG_VOL20": [120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
        }
    ).set_index("Date")

    monkeypatch.setattr(backtester_module, "validate_ohlcv", lambda raw_df: raw_df)
    monkeypatch.setattr(backtester_module, "add_indicators", lambda raw_df: raw_df)

    bt = TrendPullbackBacktester()
    bt.config = bt.config.__class__(
        initial_cash=10_000.0,
        stop_loss=0.10,
        take_profit=0.20,
        trailing_stop=0.08,
        max_hold_days=60,
        risk_per_trade=0.02,
        atr_multiple=2.0,
    )

    _, _, trades_df, signals_df = bt.run(df)

    assert list(trades_df["Action"]) == ["BUY", "SELL"]
    assert trades_df.iloc[0]["Date"] == dates[2]
    assert trades_df.iloc[0]["Price"] == 110.0
    assert trades_df.iloc[0]["Shares"] == 18
    assert trades_df.iloc[0]["InitialStop"] == 99.0
    assert trades_df.iloc[1]["Date"] == dates[5]
    assert trades_df.iloc[1]["Price"] == 88.0

    assert signals_df.loc[dates[1], "Signal"] == "BUY"
    assert not bool(signals_df.loc[dates[1], "InPositionAfterSignal"])
    assert signals_df.loc[dates[1], "PendingOrderAction"] == "BUY"
    assert signals_df.loc[dates[4], "Signal"] == "EXIT_LONG"
    assert signals_df.loc[dates[4], "PendingOrderAction"] == "EXIT_LONG"


def test_exit_long_does_not_trigger_bearish_options_trade():
    rec = recommend_options_trade(
        stock_signal="EXIT_LONG",
        ticker="AAPL",
        account_equity=10_000.0,
        risk_per_trade=0.015,
        price=200.0,
        atr=5.0,
    )

    assert rec.options_action == "NO_OPTIONS_TRADE"
    assert rec.trade_quality == "NO_TRADE"
