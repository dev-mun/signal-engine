from __future__ import annotations

from pathlib import Path

import pandas as pd


def print_scan_results(results: list[dict]) -> None:
    actionable = [result for result in results if result["Signal"] == "BUY"]
    errors = [result for result in results if result["Signal"] == "ERROR"]

    print("\nOptions Momentum Watchlist Scan")
    print("------------------------------")
    print(
        f'{"Ticker":<8} '
        f'{"Strategy":<18} '
        f'{"SourceStrategy":<16} '
        f'{"Signal":<8} '
        f'{"Setup":<12} '
        f'{"OptionType":<10} '
        f'{"Strike":>8} '
        f'{"DTE":>6} '
        f'{"DeltaTarget":>12} '
        f'{"EstPremium":>12} '
        f'{"MaxLoss":>10} '
        f'{"Exit1":>8} '
        f'{"Exit2":>8} '
        f'{"Exit3":>8} '
        f'{"Liquidity":<10} '
        f"Notes"
    )
    print("-" * 190)

    for result in results:
        if result["Signal"] == "ERROR":
            continue

        print(
            f'{result["Ticker"]:<8} '
            f'{result["Strategy"]:<18} '
            f'{result["SourceStrategy"]:<16} '
            f'{result["Signal"]:<8} '
            f'{result["Setup"]:<12} '
            f'{result["OptionType"]:<10} '
            f'{result["Strike"]:>8.2f} '
            f'{result["DTE"]:>6} '
            f'{result["DeltaTarget"]:>12.2f} '
            f'{result["EstPremium"]:>12.2f} '
            f'{result["MaxLoss"]:>10.2f} '
            f'{result["Exit1"]:>8.2f} '
            f'{result["Exit2"]:>8.2f} '
            f'{result["Exit3"]:>8.2f} '
            f'{result["Liquidity"]:<10} '
            f'{result["Notes"]}'
        )

    print("\nActionable Plans")
    print("----------------")
    if not actionable:
        print("No actionable options momentum plans.")
    else:
        for result in actionable:
            print(
                f'{result["Ticker"]}: {result["SourceStrategy"]} -> '
                f'{result["OptionType"]} {result["Strike"]:.2f} | '
                f'DTE={result["DTE"]} | EstPremium={result["EstPremium"]:.2f} | '
                f'Exit1/2/3={result["Exit1"]:.2f}/{result["Exit2"]:.2f}/{result["Exit3"]:.2f}'
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
    if source_df.empty:
        print("No source strategy data available.")
    else:
        preview_columns = [
            "source_strategy",
            "signal",
            "setup",
            "price",
            "rsi",
            "atr",
            "qualified",
            "notes",
        ]
        print(source_df.loc[:, preview_columns].to_string(index=False))

    print("\nOptions Overlay Plan")
    print("--------------------")
    print(f"Ticker: {result['Ticker']}")
    print(f"Source Strategy: {result['SourceStrategy']}")
    print(f"Signal: {result['Signal']}")
    print(f"Setup: {result['Setup']}")
    print(f"Direction: {result['Direction']}")
    print(f"Option Type: {result['OptionType']}")
    print(f"Suggested Strike: {result['Strike']:.2f}")
    print(f"Suggested DTE: {result['DTE']}")
    print(f"Target Delta: {result['DeltaTarget']:.2f}")
    print(f"Estimated Premium: {result['EstPremium']:.2f}")
    print(f"Max Loss: {result['MaxLoss']:.2f}")
    print(f"Liquidity: {result['Liquidity']}")
    print(f"Notes: {result['Notes']}")

    print("\nRisk Plan")
    print("---------")
    print("Contracts: 1")
    print(f"Premium Stop (-20%): {result['StopLoss']}")
    print(f"Exit 1 (+25%): {result['Exit1']:.2f}")
    print(f"Exit 2 (+40%): {result['Exit2']:.2f}")
    print(f"Exit 3 (+60%): {result['Exit3']:.2f}")

    print("\nJournal-Ready Trade Summary")
    print("---------------------------")
    print(f"Structure: {result['Structure']}")
    print(f"Expiration: {result['Expiration']}")
    print(f"DTE: {result['DTE']}")
    print(f"Estimated Debit: {result['EstimatedDebit']}")
    print(f"Max Loss: {result['MaxLoss']}")
    print(f"Notes: {result['OptionsReason']}")


def save_reports(analysis: dict, output_dir: str = "reports") -> None:
    output_path = Path(output_dir) / "options_momentum"
    output_path.mkdir(parents=True, exist_ok=True)

    ticker = str(analysis["result"]["Ticker"]).replace("/", "_").replace(" ", "_").upper()
    plan_file = output_path / f"{ticker}_options_plan.csv"
    source_file = output_path / f"{ticker}_source_signals.csv"

    pd.DataFrame([analysis["result"]]).to_csv(plan_file, index=False)
    pd.DataFrame(analysis["sources"]).to_csv(source_file, index=False)

    print("\nSaved Options Momentum Reports")
    print("------------------------------")
    print(f"Options plan: {plan_file}")
    print(f"Source signals: {source_file}")


def save_scan_results(results: list[dict], output_dir: str = "reports") -> None:
    output_path = Path(output_dir) / "options_momentum"
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        return

    date_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    file_path = output_path / f"watchlist_scan_{date_str}.csv"
    df.to_csv(file_path, index=False)

    print(f"\nSaved Options Momentum Watchlist Scan: {file_path}")
