from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_small_account_growth.py"
SPEC = importlib.util.spec_from_file_location("backtest_small_account_growth", SCRIPT_PATH)
small_account_growth = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(small_account_growth)


def test_load_small_account_growth_universe():
    assert small_account_growth.load_small_account_growth_universe() == [
        "PLTR",
        "UBER",
        "SOFI",
        "HOOD",
        "PYPL",
        "RIVN",
        "DKNG",
        "AFRM",
        "HIMS",
        "CLSK",
        "IONQ",
        "TOST",
        "SNAP",
        "SHOP",
        "AMD",
    ]


def test_build_growth_summary_and_monthly_report():
    payload = {
        "signals": pd.DataFrame(
            [
                {
                    "Ticker": "PLTR",
                    "SignalDate": "2026-01-05",
                    "Signal": "BUY",
                    "EstDebit": 1.15,
                    "ApproximationConfidence": "MEDIUM",
                    "ApproximationWarning": "",
                },
                {
                    "Ticker": "UBER",
                    "SignalDate": "2026-01-12",
                    "Signal": "BUY",
                    "EstDebit": 1.05,
                    "ApproximationConfidence": "LOW",
                    "ApproximationWarning": "Verify live chain.",
                },
                {
                    "Ticker": "AMD",
                    "SignalDate": "2026-02-03",
                    "Signal": "HOLD",
                    "EstDebit": 0.0,
                    "ApproximationConfidence": "N/A",
                    "ApproximationWarning": "",
                },
            ]
        ),
        "affordable_candidates": pd.DataFrame(
            [
                {
                    "Ticker": "PLTR",
                    "SignalDate": "2026-01-05",
                    "Signal": "BUY",
                    "EstDebit": 1.15,
                    "ApproximationConfidence": "MEDIUM",
                    "ApproximationWarning": "",
                },
                {
                    "Ticker": "UBER",
                    "SignalDate": "2026-01-12",
                    "Signal": "BUY",
                    "EstDebit": 1.05,
                    "ApproximationConfidence": "LOW",
                    "ApproximationWarning": "Verify live chain.",
                },
            ]
        ),
        "selected_candidates": pd.DataFrame(
            [
                {
                    "Ticker": "PLTR",
                    "SignalDate": "2026-01-05",
                    "ApproximationConfidence": "MEDIUM",
                    "ApproximationWarning": "",
                }
            ]
        ),
        "trades": pd.DataFrame(
            [
                {
                    "Ticker": "PLTR",
                    "SignalDate": "2026-01-05",
                    "EstDebit": 1.15,
                    "ProxyWinner": "YES",
                }
            ]
        ),
    }

    summary_df = small_account_growth.build_growth_summary(payload, universe=["PLTR", "UBER"])
    monthly_df = small_account_growth.build_growth_monthly_report(payload)

    assert float(summary_df.iloc[0]["ActionableFrequencyPerMonth"]) == 1.0
    assert float(summary_df.iloc[0]["AffordableTradesPerMonth"]) == 1.0
    assert float(summary_df.iloc[0]["TradesPerMonth"]) == 0.5
    assert float(summary_df.iloc[0]["ProxyWinRate"]) == 100.0
    assert float(summary_df.iloc[0]["AverageDebit"]) == 1.15
    assert float(summary_df.iloc[0]["PlannerMismatchFrequencyProxy"]) == 50.0

    assert list(monthly_df["Month"]) == ["2026-01"]
    assert int(monthly_df.iloc[0]["ActionableSignals"]) == 2
    assert int(monthly_df.iloc[0]["AffordableCandidates"]) == 2
    assert int(monthly_df.iloc[0]["TradesTriggered"]) == 1
    assert int(monthly_df.iloc[0]["PlannerMismatchCandidates"]) == 1
