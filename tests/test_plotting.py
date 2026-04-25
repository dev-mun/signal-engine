import pandas as pd
import pytest

from algo_backtester.plotting import build_signal_overlay_data


def test_build_signal_overlay_data_creates_markers_and_trade_lines():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price_df = pd.DataFrame(
        {
            "Close": [100.0, 102.0, 99.0, 97.0],
            "EMA20": [99.0, 100.0, 100.0, 99.0],
            "EMA50": [98.0, 99.0, 99.0, 98.0],
            "EMA200": [95.0, 95.0, 95.0, 95.0],
            "RSI": [45.0, 55.0, 48.0, 42.0],
            "Volume": [1_000.0, 1_500.0, 1_400.0, 1_300.0],
            "AVG_VOL20": [900.0, 950.0, 980.0, 1_000.0],
        },
        index=dates,
    )
    signals_df = pd.DataFrame(
        {
            "Signal": ["BUY", "HOLD_POSITION", "EXIT_LONG", "BEARISH_ENTRY"],
            "InPositionAfterSignal": [False, True, True, False],
            "EntryPrice": [0.0, 101.0, 101.0, 0.0],
            "InitialStopPrice": [0.0, 92.0, 92.0, 0.0],
        },
        index=dates,
    )

    overlays = build_signal_overlay_data(
        price_df=price_df,
        signals_df=signals_df,
        take_profit_pct=0.20,
    )

    assert list(overlays["buy_markers"].index) == [dates[0]]
    assert list(overlays["exit_markers"].index) == [dates[2]]
    assert list(overlays["bearish_markers"].index) == [dates[3]]
    assert list(overlays["entry_line"].index) == [dates[1], dates[2]]
    assert list(overlays["stop_line"].values) == [92.0, 92.0]
    assert list(overlays["target_line"].values) == pytest.approx([121.2, 121.2])
