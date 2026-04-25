from src.algo_backtester.data_loader import make_demo_data, validate_ohlcv
from src.algo_backtester.indicators import add_indicators


def test_indicators_are_created():
    raw_df = make_demo_data(rows=500)
    df = add_indicators(validate_ohlcv(raw_df))

    assert not df.empty
    assert "EMA20" in df.columns
    assert "EMA50" in df.columns
    assert "EMA200" in df.columns
    assert "RSI" in df.columns
    assert "AVG_VOL20" in df.columns


def test_rsi_bounds():
    raw_df = make_demo_data(rows=500)
    df = add_indicators(validate_ohlcv(raw_df))

    assert df["RSI"].between(0, 100).all()
