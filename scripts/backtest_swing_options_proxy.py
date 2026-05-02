from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algo_backtester.backtests.swing_options_proxy_backtester import PROXY_VALIDATION_LABEL, run_proxy_backtest
from algo_backtester.strategies.swing_options import PLANNER_DISCLAIMER
from algo_backtester.watchlists import get_default_watchlist_for_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer 2 proxy validation for swing-options.")
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--start", type=str, default="2024-05-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports")
    return parser.parse_args()


def _parse_tickers(tickers_arg: str | None) -> list[str] | None:
    if tickers_arg is None:
        return None
    return [ticker.strip().upper() for ticker in tickers_arg.split(",") if ticker.strip()]


def main() -> None:
    args = parse_args()
    tickers = _parse_tickers(args.tickers)
    resolved_tickers = tickers if tickers is not None else get_default_watchlist_for_strategy("swing-options")

    print(PROXY_VALIDATION_LABEL)
    print("---------------------")
    print("Strategy: swing-options")
    print(f"Tickers: {', '.join(resolved_tickers)}")
    print(f"Start: {args.start}")
    print(f"End: {args.end or 'latest available'}")
    print(PLANNER_DISCLAIMER)

    payload = run_proxy_backtest(
        tickers=resolved_tickers,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )

    summary_df = payload["summary"]
    monthly_df = payload["monthly"]
    trades_df = payload["trades"]
    audit_df = payload["audit"]
    errors_df = payload["errors"]
    paths = payload["paths"]

    print("\nSummary")
    print("-------")
    if summary_df.empty:
        print("No summary rows generated.")
    else:
        print(summary_df.to_string(index=False))

    print("\nMonthly Report")
    print("--------------")
    if monthly_df.empty:
        print("No monthly rows generated.")
    else:
        print(monthly_df.to_string(index=False))

    print("\nMove-Quality Distribution")
    print("-------------------------")
    if trades_df.empty:
        print("No BUY-triggered proxy trades.")
    else:
        distribution = trades_df["MoveQuality"].value_counts().rename_axis("MoveQuality").reset_index(name="Count")
        print(distribution.to_string(index=False))

    print("\nActionable Audit")
    print("----------------")
    if audit_df.empty:
        print("No pre-final ACTIONABLE candidates.")
    else:
        block_distribution = audit_df["BlockReason"].value_counts().rename_axis("BlockReason").reset_index(name="Count")
        print(block_distribution.to_string(index=False))

    if not errors_df.empty:
        print("\nTicker Errors")
        print("-------------")
        print(errors_df.to_string(index=False))

    print("\nSaved Reports")
    print("-------------")
    print(f"Summary: {paths['summary']}")
    print(f"Trades: {paths['trades']}")
    print(f"Monthly: {paths['monthly']}")
    print(f"Audit: {paths['audit']}")


if __name__ == "__main__":
    main()
