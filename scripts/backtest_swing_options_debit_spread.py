from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algo_backtester.backtests.swing_options_debit_spread_backtester import (
    SwingOptionsDebitSpreadConfig,
    build_tuning_summary,
    run_debit_spread_backtest,
    save_tuning_summary,
)
from algo_backtester.strategies.swing_options_debit_spread import (
    DEBIT_SPREAD_PLANNER_DISCLAIMER,
    PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
)
from algo_backtester.watchlists import get_default_watchlist_for_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proxy backtest for swing-options-debit-spread.")
    parser.add_argument("--start", type=str, default="2024-05-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--mode", type=str, default="tuned", choices=["strict", "tuned"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = get_default_watchlist_for_strategy("swing-options-debit-spread")
    strict_payload = run_debit_spread_backtest(
        tickers=tickers,
        start=args.start,
        end=args.end,
        config=SwingOptionsDebitSpreadConfig(mode="strict"),
        output_dir=args.output_dir,
        save_outputs=args.mode == "strict",
    )
    tuned_payload = run_debit_spread_backtest(
        tickers=tickers,
        start=args.start,
        end=args.end,
        config=SwingOptionsDebitSpreadConfig(mode="tuned"),
        output_dir=args.output_dir,
        save_outputs=args.mode == "tuned",
    )
    tuning_df = build_tuning_summary(strict_payload=strict_payload, tuned_payload=tuned_payload)
    tuning_path = save_tuning_summary(tuning_df=tuning_df, output_dir=args.output_dir)
    payload = strict_payload if args.mode == "strict" else tuned_payload

    print(PROXY_DEBIT_SPREAD_VALIDATION_LABEL)
    print("----------------------------------")
    print(f"Universe: {', '.join(tickers)}")
    print(f"Mode: {args.mode}")
    print(DEBIT_SPREAD_PLANNER_DISCLAIMER)

    print("\nSummary")
    print("-------")
    print(payload["summary"].to_string(index=False))

    print("\nMonthly")
    print("-------")
    print(payload["monthly"].to_string(index=False))

    print("\nSaved Reports")
    print("-------------")
    print(f"Summary: {payload['paths']['summary']}")
    print(f"Trades: {payload['paths']['trades']}")
    print(f"Monthly: {payload['paths']['monthly']}")
    print(f"Tuning Summary: {tuning_path}")


if __name__ == "__main__":
    main()
