from algo_backtester.backtester import TrendPullbackBacktester
from algo_backtester.backtests.ema_rsi_backtester import EmaRsiPullbackBacktester
from algo_backtester.data_loader import make_demo_data, validate_ohlcv
from algo_backtester.reports.ema_rsi_report import latest_signal
from algo_backtester.strategies.ema_rsi_pullback import add_indicators, classify_setup, distance_to_setup


def test_ema_rsi_indicators_are_created():
    raw_df = make_demo_data(rows=500)
    df = add_indicators(validate_ohlcv(raw_df))

    assert not df.empty
    assert "EMA20" in df.columns
    assert "EMA50" in df.columns
    assert "EMA200" in df.columns
    assert "RSI14" in df.columns
    assert "ATR14" in df.columns
    assert "AVG_VOL20" in df.columns


def test_ema_rsi_backtest_returns_expected_frames():
    raw_df = make_demo_data(rows=500)
    bt = EmaRsiPullbackBacktester()

    _, equity_df, trades_df, signals_df = bt.run(raw_df)

    assert not equity_df.empty
    assert not signals_df.empty
    assert "Equity" in equity_df.columns
    assert "Signal" in signals_df.columns
    assert trades_df is not None


def test_ema_rsi_latest_signal_has_expected_fields():
    raw_df = make_demo_data(rows=500)
    bt = EmaRsiPullbackBacktester()

    _, _, _, signals_df = bt.run(raw_df)
    signal = latest_signal(signals_df)

    assert "Signal" in signal
    assert "Reason" in signal
    assert "Close" in signal
    assert "RSI14" in signal
    assert signal["Signal"] in {"BUY", "SELL", "HOLD", "HOLD_POSITION"}


def test_existing_trend_pullback_backtester_still_runs():
    raw_df = make_demo_data(rows=500)
    bt = TrendPullbackBacktester()

    _, equity_df, _, signals_df = bt.run(raw_df)

    assert not equity_df.empty
    assert not signals_df.empty


def test_ema_rsi_setup_classification_buckets():
    assert classify_setup("HOLD", 75.0) == "EXTENDED"
    assert classify_setup("HOLD", 65.0) == "NEEDS_PULLBACK"
    assert classify_setup("HOLD", 57.0) == "NEAR_SETUP"
    assert classify_setup("HOLD", 45.0) == "WAIT"
    assert classify_setup("HOLD", 35.0) == "WEAK"


def test_ema_rsi_actionable_signals_override_rsi_bucket():
    assert classify_setup("BUY", 35.0) == "ACTIONABLE"
    assert classify_setup("SELL", 75.0) == "ACTIONABLE"


def test_ema_rsi_distance_to_setup_buckets():
    assert distance_to_setup("HOLD", 75.0) == "Too hot"
    assert distance_to_setup("HOLD", 65.0) == "Needs pullback (-10.0 RSI)"
    assert distance_to_setup("HOLD", 57.0) == "Near setup"
    assert distance_to_setup("HOLD", 45.0) == "Waiting for EMA20 reclaim"
    assert distance_to_setup("HOLD", 35.0) == "Too weak"
    assert distance_to_setup("BUY", 65.0) == "Actionable now"
