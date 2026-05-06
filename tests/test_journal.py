from pathlib import Path

import pandas as pd

from algo_backtester.journal import (
    DAILY_SCAN_LOG_SHEET,
    TRADE_JOURNAL_SHEET,
    update_paper_trading_journal,
)


def test_update_paper_trading_journal_creates_workbook_and_dedupes_rows(tmp_path: Path):
    results = [
        {
            "Ticker": "AAPL",
            "SignalDate": "2026-04-25",
            "PlannedExecutionDate": "2026-04-27",
            "UniverseStatus": "ELIGIBLE",
            "UniverseReason": "",
            "Signal": "BUY",
            "SetupStatus": "ACTIONABLE",
            "DistanceToSetup": "Actionable now",
            "Price": 210.5,
            "RSI": 52.1,
            "ATR": 4.2,
            "Equity": 10_000.0,
            "OptionsAction": "CONSIDER_BULLISH_OPTIONS_TRADE",
            "Structure": "Call Debit Spread",
            "DTE": 35,
            "Reason": "Trend pullback bullish entry",
            "OptionsReason": "Bullish signal with acceptable spread economics.",
            "AvgDollarVolume": 50_000_000.0,
            "EarningsDate": "2026-05-08",
            "Expiration": "2026-05-29",
            "LongStrike": 210.0,
            "ShortStrike": 220.0,
            "EstimatedDebit": 3.2,
            "MaxLoss": 320.0,
            "MaxProfit": 680.0,
            "TradeQuality": "VALID",
            "PlannedEntryReference": 210.5,
            "StopLoss": 193.66,
            "TakeProfit": 252.6,
            "RiskPerShare": 16.84,
            "RewardPerShare": 42.1,
            "LiveDebit": 5.45,
            "LiveMaxLoss": 545.0,
            "LiveMaxProfit": 955.0,
            "LiveRewardRisk": 1.75,
            "LiveChainConfirmed": "YES",
            "PlannerMismatch": "YES",
            "SkipReason": "",
        },
        {
            "Ticker": "MSFT",
            "SignalDate": "2026-04-25",
            "PlannedExecutionDate": "2026-04-27",
            "UniverseStatus": "ELIGIBLE",
            "UniverseReason": "",
            "Signal": "HOLD",
            "SetupStatus": "WAIT",
            "DistanceToSetup": "Needs strength (+1.5 RSI)",
            "Price": 400.0,
            "RSI": 53.5,
            "ATR": 5.0,
            "Equity": 10_000.0,
            "OptionsAction": "NO_OPTIONS_TRADE",
            "Structure": "No trade",
            "DTE": 0,
            "Reason": "No actionable setup",
            "OptionsReason": "MSFT has no actionable stock signal. Do not force an options trade.",
            "AvgDollarVolume": 60_000_000.0,
            "EarningsDate": "2026-05-01",
            "Expiration": "N/A",
            "LongStrike": 0.0,
            "ShortStrike": 0.0,
            "EstimatedDebit": 0.0,
            "MaxLoss": 0.0,
            "MaxProfit": 0.0,
            "TradeQuality": "NO_TRADE",
            "PlannedEntryReference": "",
            "StopLoss": "",
            "TakeProfit": "",
            "RiskPerShare": "",
            "RewardPerShare": "",
        },
    ]

    workbook_path = update_paper_trading_journal(results=results, output_dir=str(tmp_path))
    update_paper_trading_journal(results=results, output_dir=str(tmp_path))

    assert workbook_path.exists()

    daily_df = pd.read_excel(workbook_path, sheet_name=DAILY_SCAN_LOG_SHEET)
    trade_df = pd.read_excel(workbook_path, sheet_name=TRADE_JOURNAL_SHEET)

    assert len(daily_df) == 2
    assert len(trade_df) == 1
    assert trade_df.iloc[0]["Trade ID"] == "2026-04-25-AAPL-BUY"
    assert trade_df.iloc[0]["Options Structure"] == "Call Debit Spread"
    assert float(trade_df.iloc[0]["LiveDebit"]) == 5.45
    assert trade_df.iloc[0]["LiveChainConfirmed"] == "YES"
    assert trade_df.iloc[0]["PlannerMismatch"] == "YES"
    assert pd.isna(trade_df.iloc[0]["Actual Entry"])


def test_update_paper_trading_journal_adds_exit_long_row(tmp_path: Path):
    results = [
        {
            "Ticker": "NVDA",
            "SignalDate": "2026-04-25",
            "PlannedExecutionDate": "2026-04-27",
            "UniverseStatus": "ELIGIBLE",
            "UniverseReason": "",
            "Signal": "EXIT_LONG",
            "SetupStatus": "EXIT",
            "DistanceToSetup": "Exit long position next open",
            "Price": 900.0,
            "RSI": 45.0,
            "ATR": 20.0,
            "Equity": 10_000.0,
            "OptionsAction": "NO_OPTIONS_TRADE",
            "Structure": "No trade",
            "DTE": 0,
            "Reason": "Trailing stop triggered",
            "OptionsReason": "Close the long; do not open a bearish options trade from this event.",
            "AvgDollarVolume": 80_000_000.0,
            "EarningsDate": "2026-05-20",
            "Expiration": "N/A",
            "LongStrike": 0.0,
            "ShortStrike": 0.0,
            "EstimatedDebit": 0.0,
            "MaxLoss": 0.0,
            "MaxProfit": 0.0,
            "TradeQuality": "NO_TRADE",
            "PlannedEntryReference": "",
            "StopLoss": "",
            "TakeProfit": "",
            "RiskPerShare": "",
            "RewardPerShare": "",
        },
    ]

    workbook_path = update_paper_trading_journal(results=results, output_dir=str(tmp_path))
    trade_df = pd.read_excel(workbook_path, sheet_name=TRADE_JOURNAL_SHEET)

    assert len(trade_df) == 1
    assert trade_df.iloc[0]["Asset Type"] == "Position Exit"
    assert trade_df.iloc[0]["Direction"] == "EXIT_LONG"
    assert trade_df.iloc[0]["Setup"] == "Exit Long"
