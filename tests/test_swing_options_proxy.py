from pathlib import Path

import pandas as pd

from algo_backtester.backtests.swing_options_proxy_backtester import (
    PROXY_VALIDATION_LABEL,
    _monthly_summary,
    _summary_report,
    classify_proxy_move_quality,
    evaluate_proxy_trade,
    save_proxy_backtest_outputs,
)


def _daily_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-05", periods=8),
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "High": [101.0, 102.0, 105.0, 108.0, 110.5, 109.0, 108.0, 107.5],
            "Low": [99.0, 100.0, 101.5, 102.0, 103.0, 102.0, 101.0, 100.0],
            "Close": [100.5, 101.5, 104.5, 107.0, 109.5, 108.0, 107.0, 106.0],
            "Volume": [1_000_000] * 8,
        }
    )


def test_classify_proxy_move_quality():
    quality, reason, hold_days, failed = classify_proxy_move_quality(
        days_to_1r=2,
        days_to_2r=4,
        days_to_3r=5,
        days_to_stop=None,
        available_days=10,
    )
    assert quality == "EXCELLENT"
    assert reason == "TARGET_3R"
    assert hold_days == 5
    assert failed is False

    quality, reason, hold_days, failed = classify_proxy_move_quality(
        days_to_1r=4,
        days_to_2r=None,
        days_to_3r=None,
        days_to_stop=2,
        available_days=10,
    )
    assert quality == "FAILED"
    assert reason == "STOP_1R"
    assert hold_days == 2
    assert failed is True

    quality, reason, hold_days, failed = classify_proxy_move_quality(
        days_to_1r=7,
        days_to_2r=None,
        days_to_3r=None,
        days_to_stop=None,
        available_days=10,
    )
    assert quality == "WEAK"
    assert reason == "TIME_STOP"
    assert hold_days == 5
    assert failed is False


def test_evaluate_proxy_trade_excellent_move():
    trade = evaluate_proxy_trade(
        ticker="NVDA",
        signal_date="2026-01-05",
        atr=2.0,
        daily_df=_daily_prices(),
        max_hold_days=6,
        time_stop_days=5,
    )

    assert trade is not None
    assert trade["EntryDate"] == "2026-01-06"
    assert trade["MoveQuality"] == "EXCELLENT"
    assert trade["DaysTo1R"] == 2
    assert trade["DaysTo2R"] == 2
    assert trade["DaysTo3R"] == 3
    assert trade["ValidationType"] == PROXY_VALIDATION_LABEL


def test_evaluate_proxy_trade_failed_move():
    daily_df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-02-02", periods=6),
            "Open": [100.0, 100.5, 99.5, 99.0, 98.5, 98.0],
            "High": [100.5, 101.0, 100.0, 99.5, 99.0, 98.5],
            "Low": [99.0, 97.5, 97.0, 96.5, 96.0, 95.5],
            "Close": [100.0, 98.0, 97.5, 97.0, 96.5, 96.0],
            "Volume": [1_000_000] * 6,
        }
    )
    trade = evaluate_proxy_trade(
        ticker="AVGO",
        signal_date="2026-02-02",
        atr=2.0,
        daily_df=daily_df,
        max_hold_days=5,
        time_stop_days=5,
    )

    assert trade is not None
    assert trade["MoveQuality"] == "FAILED"
    assert trade["ExitReason"] == "STOP_1R"
    assert trade["Failed"] is True


def test_proxy_summary_and_monthly_reports():
    signal_df = pd.DataFrame(
        [
            {"SignalDate": "2026-01-05", "Signal": "BUY", "RawSetup": "ACTIONABLE"},
            {"SignalDate": "2026-01-20", "Signal": "BUY", "RawSetup": "ACTIONABLE"},
            {"SignalDate": "2026-02-10", "Signal": "HOLD", "RawSetup": "WATCHLIST"},
        ]
    )
    trades_df = pd.DataFrame(
        [
            {"Ticker": "NVDA", "SignalDate": "2026-01-05", "MoveQuality": "EXCELLENT", "HoldDays": 4, "MFE_R": 3.2, "MAE_R": -0.4},
            {"Ticker": "AVGO", "SignalDate": "2026-01-20", "MoveQuality": "FAILED", "HoldDays": 2, "MFE_R": 0.4, "MAE_R": -1.2},
        ]
    )

    monthly_df = _monthly_summary(signal_df=signal_df, trades_df=trades_df)
    summary_df = _summary_report(signal_df=signal_df, trades_df=trades_df)

    assert list(monthly_df["Month"]) == ["2026-01", "2026-02"]
    assert int(monthly_df.loc[monthly_df["Month"] == "2026-01", "BUYSignals"].iloc[0]) == 2
    assert int(monthly_df.loc[monthly_df["Month"] == "2026-01", "EXCELLENT"].iloc[0]) == 1
    assert int(monthly_df.loc[monthly_df["Month"] == "2026-01", "FAILED"].iloc[0]) == 1
    assert float(summary_df.iloc[0]["PctReached3RWithin15D"]) == 50.0
    assert float(summary_df.iloc[0]["PctFailed"]) == 50.0


def test_proxy_output_paths(tmp_path: Path):
    summary_df = pd.DataFrame([{"ValidationType": PROXY_VALIDATION_LABEL, "TotalBUYSignals": 1}])
    trades_df = pd.DataFrame([{"ValidationType": PROXY_VALIDATION_LABEL, "Ticker": "NVDA"}])
    monthly_df = pd.DataFrame([{"ValidationType": PROXY_VALIDATION_LABEL, "Month": "2026-01"}])
    audit_df = pd.DataFrame([{"ValidationType": PROXY_VALIDATION_LABEL, "Ticker": "NVDA", "BlockReason": "PREMIUM_OVER_BUDGET"}])

    paths = save_proxy_backtest_outputs(
        summary_df=summary_df,
        trades_df=trades_df,
        monthly_df=monthly_df,
        audit_df=audit_df,
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "swing_options" / "proxy_backtest_summary.csv").exists()
    assert (tmp_path / "swing_options" / "proxy_backtest_trades.csv").exists()
    assert (tmp_path / "swing_options" / "proxy_backtest_monthly.csv").exists()
    assert (tmp_path / "swing_options" / "actionable_audit.csv").exists()
    assert paths["summary"].name == "proxy_backtest_summary.csv"
