from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo_backtester.strategies.swing_options_debit_spread import (
    DEBIT_SPREAD_PLANNER_DISCLAIMER,
    PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
)


def print_scan_results(results: list[dict]) -> None:
    actionable = [result for result in results if result["Signal"] == "BUY"]
    errors = [result for result in results if result["Signal"] == "ERROR"]

    print("\nSwing Options Debit Spread Scan")
    print("-------------------------------")
    print(PROXY_DEBIT_SPREAD_VALIDATION_LABEL)
    print(DEBIT_SPREAD_PLANNER_DISCLAIMER)
    print(
        f'{"Ticker":<8} '
        f'{"Strategy":<28} '
        f'{"Signal":<8} '
        f'{"ActionState":<12} '
        f'{"Setup":<12} '
        f'{"Regime":<9} '
        f'{"Rating":<10} '
        f'{"FinalScore":>10} '
        f'{"Price":>10} '
        f'{"OptionStructure":<28} '
        f'{"LongStrike":>12} '
        f'{"ShortStrike":>12} '
        f'{"DTE":>6} '
        f'{"EstDebit":>10} '
        f'{"MaxLoss":>10} '
        f'{"MaxProfit":>10} '
        f'{"RewardRisk":>12} '
        f'{"Approx":<8} '
        f'{"Width":>8} '
        f'{"PremiumStatus":<16} '
        f'{"SmallAcct":<10} '
        f"Reason"
    )
    print("-" * 310)

    for result in results:
        if result["Signal"] == "ERROR":
            continue
        print(
            f'{result["Ticker"]:<8} '
            f'{result["Strategy"]:<28} '
            f'{result["Signal"]:<8} '
            f'{result.get("ActionState", "WATCHLIST"):<12} '
            f'{result["Setup"]:<12} '
            f'{result["MarketRegime"]:<9} '
            f'{result["SetupRating"]:<10} '
            f'{float(result.get("FinalScore", result.get("SetupScore", result.get("Score", 0.0))) or 0.0):>10.2f} '
            f'{result["Price"]:>10.2f} '
            f'{result["OptionStructure"]:<28} '
            f'{result["LongStrike"]:>12.2f} '
            f'{result["ShortStrike"]:>12.2f} '
            f'{result["DTE"]:>6} '
            f'{result["EstDebit"]:>10.2f} '
            f'{result["MaxLoss"]:>10.2f} '
            f'{result["MaxProfit"]:>10.2f} '
            f'{result["RewardRisk"]:>12.2f} '
            f'{result["ApproximationConfidence"]:<8} '
            f'{result["SpreadWidth"]:>8.2f} '
            f'{result["PremiumStatus"]:<16} '
            f'{result["SmallAccountEligible"]:<10} '
            f'{result["Reason"]}'
        )

    print("\nActionable Plans")
    print("----------------")
    if not actionable:
        print("No actionable debit spread plans.")
    else:
        for result in actionable:
            print(
                f'{result["Ticker"]}: {result["OptionStructure"]} | '
                f'Debit={result["EstDebit"]:.2f} | MaxProfit={result["MaxProfit"]:.2f} | '
                f'RR={result["RewardRisk"]:.2f}'
            )

    if errors:
        print("\nErrors")
        print("------")
        for result in errors:
            print(f'{result["Ticker"]}: {result["Reason"]}')


def print_ticker_plan(analysis: dict) -> None:
    result = analysis["result"]
    source_df = pd.DataFrame(analysis["sources"])

    print("\nSource Signal Summary")
    print("---------------------")
    print(PROXY_DEBIT_SPREAD_VALIDATION_LABEL)
    print(DEBIT_SPREAD_PLANNER_DISCLAIMER)
    if source_df.empty:
        print("No source strategy data available.")
    else:
        preview_columns = [
            "strategy",
            "signal",
            "setup",
            "price",
            "rsi",
            "atr",
            "trend_quality",
            "volume_confirmed",
            "recent_momentum",
            "bullish_support",
            "notes",
        ]
        print(source_df.loc[:, preview_columns].to_string(index=False))

    print("\nDebit Spread Plan")
    print("-----------------")
    print(f"Ticker: {result['Ticker']}")
    print(f"Signal: {result['Signal']}")
    print(f"Action State: {result.get('ActionState', 'WATCHLIST')}")
    print(f"Setup: {result['Setup']}")
    print(f"Base Score: {float(result.get('BaseScore', 0.0) or 0.0):.2f}")
    print(f"Setup Score: {float(result.get('SetupScore', result.get('Score', 0.0)) or 0.0):.2f}")
    print(f"Final Score: {float(result.get('FinalScore', result.get('SetupScore', result.get('Score', 0.0))) or 0.0):.2f}")
    print(f"Market Regime: {result['MarketRegime']}")
    print(f"Regime Reason: {result['RegimeReason']}")
    print(f"Daily Trend: {result['DailyTrend']}")
    print(f"4H Trend: {result['FourHourTrend']}")
    print(f"Timeframe Confirmation: {result['TimeframeConfirmation']}")
    print(f"Setup Rating: {result['SetupRating']}")
    print(f"Structure: {result['OptionStructure']}")
    print(f"Long Strike: {result['LongStrike']:.2f}")
    print(f"Short Strike: {result['ShortStrike']:.2f}")
    print(f"DTE: {result['DTE']}")
    print(f"Estimated Long Call Ask: {result['EstLongCallAsk']:.2f}")
    print(f"Estimated Short Call Bid: {result['EstShortCallBid']:.2f}")
    print(f"Estimated Debit: {result['EstDebit']:.2f}")
    print(f"Max Loss: {result['MaxLoss']:.2f}")
    print(f"Max Profit: {result['MaxProfit']:.2f}")
    print(f"Reward/Risk: {result['RewardRisk']:.2f}")
    print(f"Approximation Confidence: {result['ApproximationConfidence']}")
    if result["ApproximationWarning"]:
        print(f"Approximation Warning: {result['ApproximationWarning']}")
    print(f"Premium Status: {result['PremiumStatus']}")
    print(f"Small Account Eligible: {result['SmallAccountEligible']}")
    if result["NoTradeReasons"]:
        print(f"No-Trade Reasons: {result['NoTradeReasons']}")
    if result["Warnings"]:
        print(f"Warnings: {result['Warnings']}")
    print(f"Final Decision: {result['FinalDecision']}")
    print(f"Reason: {result['Reason']}")


def save_reports(analysis: dict, output_dir: str = "reports") -> None:
    output_path = Path(output_dir) / "swing_options_debit_spread"
    output_path.mkdir(parents=True, exist_ok=True)

    ticker = str(analysis["result"]["Ticker"]).replace("/", "_").replace(" ", "_").upper()
    plan_file = output_path / f"{ticker}_options_plan.csv"
    source_file = output_path / f"{ticker}_source_signals.csv"
    latest_signal_file = output_path / f"{ticker}_latest_signal.csv"

    pd.DataFrame([analysis["result"]]).to_csv(plan_file, index=False)
    pd.DataFrame(analysis["sources"]).to_csv(source_file, index=False)
    pd.DataFrame([analysis["result"]]).to_csv(latest_signal_file, index=False)

    print("\nSaved Swing Options Debit Spread Reports")
    print("----------------------------------------")
    print(f"Options plan: {plan_file}")
    print(f"Source signals: {source_file}")
    print(f"Latest signal: {latest_signal_file}")


def save_scan_results(results: list[dict], output_dir: str = "reports") -> None:
    output_path = Path(output_dir) / "swing_options_debit_spread"
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    if df.empty:
        return
    date_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    file_path = output_path / f"scan_{date_str}.csv"
    df.to_csv(file_path, index=False)
    print(f"\nSaved Swing Options Debit Spread Scan: {file_path}")
