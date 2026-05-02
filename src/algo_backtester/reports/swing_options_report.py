from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo_backtester.strategies.swing_options import PLANNER_DISCLAIMER


def print_scan_results(results: list[dict]) -> None:
    actionable = [result for result in results if result["Signal"] == "BUY"]
    errors = [result for result in results if result["Signal"] == "ERROR"]

    print("\nSwing Options Watchlist Scan")
    print("---------------------------")
    print(PLANNER_DISCLAIMER)
    print(
        f'{"Ticker":<8} '
        f'{"Strategy":<14} '
        f'{"AccountProfile":<22} '
        f'{"SmallAcct":<10} '
        f'{"PremiumStatus":<14} '
        f'{"Signal":<8} '
        f'{"Setup":<12} '
        f'{"Score":>7} '
        f'{"Price":>10} '
        f'{"OptionType":<10} '
        f'{"Strike":>8} '
        f'{"DTE":>6} '
        f'{"EstPremium":>12} '
        f'{"MaxLoss":>10} '
        f"Reason"
    )
    print("-" * 210)

    for result in results:
        if result["Signal"] == "ERROR":
            continue

        print(
            f'{result["Ticker"]:<8} '
            f'{result["Strategy"]:<14} '
            f'{result.get("AccountProfile", "standard"):<22} '
            f'{result.get("SmallAccountEligible", "NO"):<10} '
            f'{result.get("PremiumStatus", ""):<14} '
            f'{result["Signal"]:<8} '
            f'{result["Setup"]:<12} '
            f'{result["Score"]:>7.2f} '
            f'{result["Price"]:>10.2f} '
            f'{result["OptionType"]:<10} '
            f'{result["Strike"]:>8.2f} '
            f'{result["DTE"]:>6} '
            f'{result["EstPremium"]:>12.2f} '
            f'{result["MaxLoss"]:>10.2f} '
            f'{result["Reason"]}'
        )

    print("\nActionable Plans")
    print("----------------")
    if not actionable:
        print("No actionable swing options plans.")
    else:
        for result in actionable:
            print(
                f'{result["Ticker"]}: score={result["Score"]:.2f} | '
                f'{result["OptionType"]} {result["Strike"]:.2f} | '
                f'DTE={result["DTE"]} | EstPremium={result["EstPremium"]:.2f}'
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
    print(PLANNER_DISCLAIMER)
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

    print("\nSwing Options Score")
    print("-------------------")
    print(f"Ticker: {result['Ticker']}")
    print(f"Score: {result['Score']:.2f}")
    print(f"Setup: {result['Setup']}")
    print(f"Source Summary: {result['SourceSummary']}")

    print("\nOption Plan")
    print("-----------")
    print(f"Signal: {result['Signal']}")
    print(f"Option Type: {result['OptionType']}")
    print(f"Suggested Strike: {result['Strike']:.2f}")
    print(f"Suggested DTE: {result['DTE']}")
    print(f"Target Delta: {result['DeltaTarget']:.2f}")
    print(f"Estimated Premium: {result['EstPremium']:.2f}")
    print(f"Max Loss: {result['MaxLoss']:.2f}")
    print(f"Notes: {result['Notes']}")

    print("\nRisk Plan")
    print("---------")
    print("Contracts: 1")
    print(f"Premium Stop (-25%): {result['StopLoss']}")
    print(f"Exit 1 (+30%): {result['Exit1']:.2f}")
    print(f"Exit 2 (+50%): {result['Exit2']:.2f}")
    print(f"Exit 3 (+80%): {result['Exit3']:.2f}")
    print(f"Time Stop Days: {result['TimeStopDays']}")
    print(f"Max Hold Days: {result['MaxHoldDays']}")

    print("\nJournal-Ready Summary")
    print("---------------------")
    print(f"Structure: {result['Structure']}")
    print(f"Estimated Debit: {result['EstimatedDebit']}")
    print(f"Max Loss: {result['MaxLoss']}")
    print(f"Invalidation: {result['OptionsReason']}")


def save_reports(analysis: dict, output_dir: str = "reports") -> None:
    output_path = Path(output_dir) / "swing_options"
    output_path.mkdir(parents=True, exist_ok=True)

    ticker = str(analysis["result"]["Ticker"]).replace("/", "_").replace(" ", "_").upper()
    plan_file = output_path / f"{ticker}_options_plan.csv"
    source_file = output_path / f"{ticker}_source_signals.csv"
    latest_signal_file = output_path / f"{ticker}_latest_signal.csv"

    pd.DataFrame([analysis["result"]]).to_csv(plan_file, index=False)
    pd.DataFrame(analysis["sources"]).to_csv(source_file, index=False)
    pd.DataFrame([analysis["result"]]).to_csv(latest_signal_file, index=False)

    print("\nSaved Swing Options Reports")
    print("---------------------------")
    print(f"Options plan: {plan_file}")
    print(f"Source signals: {source_file}")
    print(f"Latest signal: {latest_signal_file}")


def save_scan_results(results: list[dict], output_dir: str = "reports") -> None:
    output_path = Path(output_dir) / "swing_options"
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        return

    date_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    file_path = output_path / f"watchlist_scan_{date_str}.csv"
    df.to_csv(file_path, index=False)

    print(f"\nSaved Swing Options Watchlist Scan: {file_path}")
