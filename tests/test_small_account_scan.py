import pandas as pd

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_small_account_scan.py"
SPEC = importlib.util.spec_from_file_location("backtest_small_account_scan", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

affordable_small_account_buys = MODULE.affordable_small_account_buys
build_small_account_monthly_report = MODULE.build_small_account_monthly_report
build_small_account_summary = MODULE.build_small_account_summary
replay_small_account_candidates = MODULE.replay_small_account_candidates
select_daily_candidates = MODULE.select_daily_candidates


def test_affordable_small_account_buys_filters_exact_workflow():
    signal_df = pd.DataFrame(
        [
            {"Ticker": "SPY", "SignalDate": "2026-01-05", "Signal": "BUY", "SmallAccountEligible": "YES", "PremiumStatus": "OK", "Score": 82.0, "EstPremium": 1.1, "ATR": 5.0},
            {"Ticker": "QQQ", "SignalDate": "2026-01-05", "Signal": "BUY", "SmallAccountEligible": "YES", "PremiumStatus": "ACCEPTABLE", "Score": 85.0, "EstPremium": 1.4, "ATR": 6.0},
            {"Ticker": "AAPL", "SignalDate": "2026-01-06", "Signal": "HOLD", "SmallAccountEligible": "NO", "PremiumStatus": "N/A", "Score": 50.0, "EstPremium": 0.0, "ATR": 3.0},
            {"Ticker": "AMD", "SignalDate": "2026-01-06", "Signal": "BUY", "SmallAccountEligible": "NO", "PremiumStatus": "TOO_EXPENSIVE", "Score": 88.0, "EstPremium": 1.8, "ATR": 7.0},
        ]
    )

    filtered = affordable_small_account_buys(signal_df)

    assert list(filtered["Ticker"]) == ["SPY", "QQQ"]


def test_select_daily_candidates_prefers_ok_then_score():
    affordable_df = pd.DataFrame(
        [
            {"Ticker": "QQQ", "SignalDate": "2026-01-05", "PremiumStatus": "ACCEPTABLE", "PremiumStatusRank": 1, "Score": 90.0, "EstPremium": 1.45, "ATR": 6.0},
            {"Ticker": "SPY", "SignalDate": "2026-01-05", "PremiumStatus": "OK", "PremiumStatusRank": 0, "Score": 83.0, "EstPremium": 1.10, "ATR": 5.0},
            {"Ticker": "AAPL", "SignalDate": "2026-01-06", "PremiumStatus": "OK", "PremiumStatusRank": 0, "Score": 80.0, "EstPremium": 0.95, "ATR": 3.0},
        ]
    )

    selected = select_daily_candidates(affordable_df)

    assert list(selected["Ticker"]) == ["SPY", "AAPL"]


def test_replay_small_account_candidates_skips_overlapping_positions():
    candidate_df = pd.DataFrame(
        [
            {"Ticker": "SPY", "SignalDate": "2026-01-05", "EntryDate": "2026-01-06", "EstPremium": 1.10, "PremiumStatus": "OK", "SmallAccountEligible": "YES", "Score": 83.0, "Setup": "WATCHLIST", "ATR": 5.0},
            {"Ticker": "QQQ", "SignalDate": "2026-01-06", "EntryDate": "2026-01-07", "EstPremium": 1.15, "PremiumStatus": "OK", "SmallAccountEligible": "YES", "Score": 81.0, "Setup": "WATCHLIST", "ATR": 6.0},
            {"Ticker": "AAPL", "SignalDate": "2026-01-09", "EntryDate": "2026-01-12", "EstPremium": 0.95, "PremiumStatus": "OK", "SmallAccountEligible": "YES", "Score": 80.0, "Setup": "WATCHLIST", "ATR": 3.0},
        ]
    )
    daily_data_by_ticker = {
        "SPY": pd.DataFrame(),
        "QQQ": pd.DataFrame(),
        "AAPL": pd.DataFrame(),
    }

    def fake_trade_evaluator(**kwargs):
        ticker = kwargs["ticker"]
        if ticker == "SPY":
            return {
                "Ticker": "SPY",
                "SignalDate": "2026-01-05",
                "EntryDate": "2026-01-06",
                "ExitDate": "2026-01-08",
                "MoveQuality": "SUITABLE",
                "HoldDays": 3,
                "ExitReason": "TARGET_1R",
                "MFE_R": 1.2,
                "MAE_R": -0.4,
            }
        if ticker == "QQQ":
            return {
                "Ticker": "QQQ",
                "SignalDate": "2026-01-06",
                "EntryDate": "2026-01-07",
                "ExitDate": "2026-01-09",
                "MoveQuality": "FAILED",
                "HoldDays": 2,
                "ExitReason": "STOP_1R",
                "MFE_R": 0.3,
                "MAE_R": -1.1,
            }
        return {
            "Ticker": "AAPL",
            "SignalDate": "2026-01-09",
            "EntryDate": "2026-01-12",
            "ExitDate": "2026-01-15",
            "MoveQuality": "STRONG",
            "HoldDays": 4,
            "ExitReason": "TARGET_2R",
            "MFE_R": 2.2,
            "MAE_R": -0.5,
        }

    trades_df = replay_small_account_candidates(
        candidate_df=candidate_df,
        daily_data_by_ticker=daily_data_by_ticker,
        trade_evaluator=fake_trade_evaluator,
    )

    assert list(trades_df["Ticker"]) == ["SPY", "AAPL"]
    assert list(trades_df["IdleDaysBeforeTrade"]) == [0, 1]


def test_small_account_summary_and_monthly_reports():
    signal_df = pd.DataFrame(
        [
            {"SignalDate": "2026-01-05", "Signal": "BUY"},
            {"SignalDate": "2026-01-06", "Signal": "BUY"},
            {"SignalDate": "2026-02-10", "Signal": "HOLD"},
        ]
    )
    affordable_df = pd.DataFrame(
        [
            {"SignalDate": "2026-01-05"},
        ]
    )
    trades_df = pd.DataFrame(
        [
            {"SignalDate": "2026-01-05", "MoveQuality": "SUITABLE", "HoldDays": 3, "MFE_R": 1.3, "MAE_R": -0.5, "IdleDaysBeforeTrade": 0},
            {"SignalDate": "2026-01-20", "MoveQuality": "FAILED", "HoldDays": 2, "MFE_R": 0.4, "MAE_R": -1.1, "IdleDaysBeforeTrade": 4},
        ]
    )

    summary_df = build_small_account_summary(signal_df=signal_df, affordable_df=affordable_df, trades_df=trades_df)
    monthly_df = build_small_account_monthly_report(signal_df=signal_df, affordable_df=affordable_df, trades_df=trades_df)

    assert int(summary_df.iloc[0]["TotalSmallAccountBUYs"]) == 2
    assert float(summary_df.iloc[0]["AverageAffordableBUYsPerMonth"]) == 0.5
    assert float(summary_df.iloc[0]["PctSuitablePlus"]) == 50.0
    assert float(summary_df.iloc[0]["PctFailed"]) == 50.0
    assert list(monthly_df["Month"]) == ["2026-01", "2026-02"]
    assert int(monthly_df.loc[monthly_df["Month"] == "2026-01", "TradesTriggered"].iloc[0]) == 2
