from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algo_backtester.backtests.rsi_bollinger_v2_backtester import run_parameter_sweep
from algo_backtester.data_loader import load_yfinance_data

DEFAULT_SWEEP_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "CRM",
    "PLTR",
    "SHOP",
    "UBER",
    "META",
    "GOOGL",
    "AVGO",
    "TSM",
    "PANW",
    "CRWD",
]

QUICK_SWEEP_GRID = {
    "rsi_threshold": [38, 40],
    "stop_loss": [0.04],
    "take_profit": [0.05, 0.06],
    "trailing_stop": [0.04],
    "max_hold_days": [7],
    "volume_multiplier": [0.6],
    "band_tolerance": [1.02],
    "close_position_min": [0.35],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep RSI Bollinger V2 parameters.")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_SWEEP_UNIVERSE))
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=str, default="reports/rsi_bollinger_v2")
    return parser.parse_args()


def default_max_workers(number_of_tickers: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(max(cpu_count - 1, 1), number_of_tickers))


def sweep_ticker(
    ticker: str,
    start: str,
    end: str | None,
    quick: bool,
) -> dict[str, object]:
    started_at = time.perf_counter()
    print(f"Starting {ticker}", flush=True)

    try:
        raw_df = load_yfinance_data(ticker=ticker, start=start, end=end)
        sweep_df = run_parameter_sweep(
            ticker=ticker,
            raw_df=raw_df,
            sweep_grid=QUICK_SWEEP_GRID if quick else None,
        )
        elapsed = time.perf_counter() - started_at
        print(f"Finished {ticker} in {elapsed:.2f}s ({len(sweep_df)} rows)", flush=True)
        return {"ticker": ticker, "sweep_df": sweep_df, "error": None}
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        print(f"Finished {ticker} with error in {elapsed:.2f}s", flush=True)
        return {"ticker": ticker, "sweep_df": pd.DataFrame(), "error": str(exc)}


def rank_sweep_results(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return result_df.copy()

    ranked_df = result_df.copy()
    ranked_df["meets_profit_factor"] = ranked_df["profit_factor"] >= 1.3
    ranked_df["meets_sharpe"] = ranked_df["Sharpe"] > 0.3
    ranked_df["meets_drawdown"] = ranked_df["max_drawdown"] > -10.0
    ranked_df["meets_completed_trades"] = ranked_df["completed_trades"] >= 20
    ranked_df["meets_total_return"] = ranked_df["total_return"] > 0.0
    ranked_df["rank_score"] = ranked_df[
        [
            "meets_profit_factor",
            "meets_sharpe",
            "meets_drawdown",
            "meets_completed_trades",
            "meets_total_return",
        ]
    ].sum(axis=1)

    ranked_df = ranked_df.sort_values(
        by=[
            "profit_factor",
            "Sharpe",
            "max_drawdown",
            "completed_trades",
        ],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))
    return ranked_df


def main() -> None:
    args = parse_args()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if not tickers:
        raise SystemExit("No tickers provided.")

    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    rows: list[pd.DataFrame] = []
    errors: list[tuple[str, str]] = []
    max_workers = args.workers if args.workers is not None else default_max_workers(len(tickers))
    started_at = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(sweep_ticker, ticker, args.start, args.end, args.quick): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                payload = future.result()
            except Exception as exc:
                errors.append((ticker, str(exc)))
                print(f"Skipping {ticker}: {exc}")
                continue

            sweep_df = payload["sweep_df"]
            error = payload["error"]
            if isinstance(error, str) and error:
                errors.append((ticker, error))
                print(f"Skipping {ticker}: {error}")
                continue
            if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty:
                rows.append(sweep_df)

    result_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ranked_df = rank_sweep_results(result_df)
    if ranked_df.empty:
        filtered_ranked_df = ranked_df.copy()
    else:
        filtered_ranked_df = ranked_df[
            ranked_df["meets_profit_factor"]
            & ranked_df["meets_sharpe"]
            & ranked_df["meets_drawdown"]
            & ranked_df["meets_completed_trades"]
            & ranked_df["meets_total_return"]
        ].reset_index(drop=True)
        if not filtered_ranked_df.empty:
            filtered_ranked_df = filtered_ranked_df.copy()
            filtered_ranked_df["rank"] = range(1, len(filtered_ranked_df) + 1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "parameter_sweep.csv"
    ranked_file_path = output_path / "parameter_sweep_ranked.csv"
    result_df.to_csv(file_path, index=False)
    filtered_ranked_df.to_csv(ranked_file_path, index=False)
    elapsed = time.perf_counter() - started_at

    print(f"Saved parameter sweep: {file_path}")
    print(f"Saved ranked sweep: {ranked_file_path}")
    print(f"Rows: {len(result_df)}")
    print(f"Ranked Rows: {len(filtered_ranked_df)}")
    print(f"Workers: {max_workers}")
    print(f"Elapsed: {elapsed:.2f}s")
    if errors:
        print("Ticker errors:")
        for ticker, error in errors:
            print(f"  {ticker}: {error}")

    if filtered_ranked_df.empty:
        print("No ranked sweep results passed the filter.")
        return

    if ranked_df.empty:
        print("No sweep results generated.")
        return

    print("\nTop ranked results")
    print("------------------")
    preview_columns = [
        "rank",
        "ticker",
        "params",
        "profit_factor",
        "Sharpe",
        "max_drawdown",
        "completed_trades",
        "trades_per_year",
        "total_return",
        "rank_score",
    ]
    print(filtered_ranked_df.loc[:, preview_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
