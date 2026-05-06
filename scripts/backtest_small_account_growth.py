from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algo_backtester.backtests.swing_options_debit_spread_backtester import (  # noqa: E402
    SwingOptionsDebitSpreadConfig,
    run_debit_spread_backtest,
)
from algo_backtester.strategies.swing_options_debit_spread import (  # noqa: E402
    PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
)
from algo_backtester.watchlists import get_watchlist  # noqa: E402

PROFILE_NAME = "small_account_growth"
OUTPUT_SUBDIR = "swing_options_debit_spread"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy replay for the small-account growth debit-spread deployment universe."
    )
    parser.add_argument("--start", type=str, default="2024-05-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports")
    return parser.parse_args()


def load_small_account_growth_universe() -> list[str]:
    return get_watchlist(PROFILE_NAME)


def _month_count(signal_df: pd.DataFrame) -> int:
    if signal_df.empty:
        return 1
    return max(1, pd.to_datetime(signal_df["SignalDate"]).dt.to_period("M").nunique())


def _planner_mismatch_proxy(series_df: pd.DataFrame) -> pd.Series:
    if series_df.empty:
        return pd.Series(dtype=bool)
    warning_series = (
        series_df["ApproximationWarning"]
        if "ApproximationWarning" in series_df.columns
        else pd.Series([""] * len(series_df), index=series_df.index)
    )
    confidence_series = (
        series_df["ApproximationConfidence"]
        if "ApproximationConfidence" in series_df.columns
        else pd.Series([""] * len(series_df), index=series_df.index)
    )
    warning_mask = warning_series.fillna("").astype(str).str.len() > 0
    confidence_mask = confidence_series.fillna("").astype(str).eq("LOW")
    return warning_mask | confidence_mask


def build_growth_summary(payload: dict[str, object], universe: list[str]) -> pd.DataFrame:
    signal_df = payload["signals"]
    affordable_df = payload["affordable_candidates"]
    trades_df = payload["trades"]
    total_months = _month_count(signal_df)
    actionable_count = int((signal_df["Signal"] == "BUY").sum()) if not signal_df.empty else 0
    affordable_count = len(affordable_df)
    trades_count = len(trades_df)
    mismatch_rate = round(_planner_mismatch_proxy(affordable_df).mean() * 100, 2) if not affordable_df.empty else 0.0

    if trades_df.empty:
        average_debit = round(float(affordable_df["EstDebit"].mean()), 2) if not affordable_df.empty else 0.0
        row = {
            "ValidationType": PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
            "Profile": PROFILE_NAME,
            "Universe": ",".join(universe),
            "TotalBUYSignals": actionable_count,
            "ActionableFrequencyPerMonth": round(actionable_count / total_months, 2),
            "AffordableTradesPerMonth": round(affordable_count / total_months, 2),
            "TradesPerMonth": 0.0,
            "ProxyWinRate": 0.0,
            "AverageDebit": average_debit,
            "PlannerMismatchFrequencyProxy": mismatch_rate,
        }
        return pd.DataFrame([row])

    row = {
        "ValidationType": PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
        "Profile": PROFILE_NAME,
        "Universe": ",".join(universe),
        "TotalBUYSignals": actionable_count,
        "ActionableFrequencyPerMonth": round(actionable_count / total_months, 2),
        "AffordableTradesPerMonth": round(affordable_count / total_months, 2),
        "TradesPerMonth": round(trades_count / total_months, 2),
        "ProxyWinRate": round((trades_df["ProxyWinner"] == "YES").mean() * 100, 2),
        "AverageDebit": round(float(trades_df["EstDebit"].mean()), 2),
        "PlannerMismatchFrequencyProxy": mismatch_rate,
    }
    return pd.DataFrame([row])


def build_growth_monthly_report(payload: dict[str, object]) -> pd.DataFrame:
    signal_df = payload["signals"]
    affordable_df = payload["affordable_candidates"]
    trades_df = payload["trades"]

    if signal_df.empty:
        return pd.DataFrame(
            columns=[
                "ValidationType",
                "Profile",
                "Month",
                "ActionableSignals",
                "AffordableCandidates",
                "TradesTriggered",
                "AverageDebit",
                "ProxyWinRate",
                "PlannerMismatchCandidates",
            ]
        )

    signal_buy_df = signal_df.loc[signal_df["Signal"] == "BUY"].copy()
    if signal_buy_df.empty:
        months = [pd.Timestamp.today().to_period("M").strftime("%Y-%m")]
    else:
        signal_buy_df["Month"] = pd.to_datetime(signal_buy_df["SignalDate"]).dt.to_period("M").astype(str)
        months = sorted(signal_buy_df["Month"].unique())

    monthly_df = pd.DataFrame({"Month": months})
    monthly_df.insert(0, "Profile", PROFILE_NAME)
    monthly_df.insert(0, "ValidationType", PROXY_DEBIT_SPREAD_VALIDATION_LABEL)

    if not signal_buy_df.empty:
        monthly_df["ActionableSignals"] = monthly_df["Month"].map(signal_buy_df.groupby("Month").size()).fillna(0).astype(int)
    else:
        monthly_df["ActionableSignals"] = 0

    if not affordable_df.empty:
        affordable_copy = affordable_df.copy()
        affordable_copy["Month"] = pd.to_datetime(affordable_copy["SignalDate"]).dt.to_period("M").astype(str)
        affordable_copy["PlannerMismatchProxy"] = _planner_mismatch_proxy(affordable_copy)
        monthly_df["AffordableCandidates"] = monthly_df["Month"].map(affordable_copy.groupby("Month").size()).fillna(0).astype(int)
        monthly_df["AverageDebit"] = monthly_df["Month"].map(affordable_copy.groupby("Month")["EstDebit"].mean()).fillna(0.0).round(2)
        monthly_df["PlannerMismatchCandidates"] = monthly_df["Month"].map(
            affordable_copy.groupby("Month")["PlannerMismatchProxy"].sum()
        ).fillna(0).astype(int)
    else:
        monthly_df["AffordableCandidates"] = 0
        monthly_df["AverageDebit"] = 0.0
        monthly_df["PlannerMismatchCandidates"] = 0

    if not trades_df.empty:
        trades_copy = trades_df.copy()
        trades_copy["Month"] = pd.to_datetime(trades_copy["SignalDate"]).dt.to_period("M").astype(str)
        monthly_df["TradesTriggered"] = monthly_df["Month"].map(trades_copy.groupby("Month").size()).fillna(0).astype(int)
        monthly_df["ProxyWinRate"] = monthly_df["Month"].map(
            trades_copy.groupby("Month")["ProxyWinner"].apply(lambda s: round((s == "YES").mean() * 100, 2))
        ).fillna(0.0)
    else:
        monthly_df["TradesTriggered"] = 0
        monthly_df["ProxyWinRate"] = 0.0

    return monthly_df


def _save_outputs(
    payload: dict[str, object],
    summary_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    output_dir: str,
) -> dict[str, Path]:
    output_path = Path(output_dir) / OUTPUT_SUBDIR
    output_path.mkdir(parents=True, exist_ok=True)

    trades_df = payload["trades"].copy()
    if not trades_df.empty:
        selected_df = payload["selected_candidates"][
            [
                "Ticker",
                "SignalDate",
                "ApproximationConfidence",
                "ApproximationWarning",
            ]
        ].copy()
        trades_df = trades_df.merge(selected_df, on=["Ticker", "SignalDate"], how="left")
        trades_df["PlannerMismatchProxy"] = _planner_mismatch_proxy(trades_df).map({True: "YES", False: "NO"})

    summary_path = output_path / "small_account_growth_summary.csv"
    trades_path = output_path / "small_account_growth_trades.csv"
    monthly_path = output_path / "small_account_growth_monthly.csv"

    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

    return {"summary": summary_path, "trades": trades_path, "monthly": monthly_path}


def main() -> None:
    args = parse_args()
    universe = load_small_account_growth_universe()
    payload = run_debit_spread_backtest(
        tickers=universe,
        start=args.start,
        end=args.end,
        config=SwingOptionsDebitSpreadConfig(mode="tuned"),
        output_dir=args.output_dir,
        save_outputs=False,
    )
    summary_df = build_growth_summary(payload, universe=universe)
    monthly_df = build_growth_monthly_report(payload)
    paths = _save_outputs(payload, summary_df=summary_df, monthly_df=monthly_df, output_dir=args.output_dir)

    print(PROXY_DEBIT_SPREAD_VALIDATION_LABEL)
    print("----------------------------------")
    print(f"Profile: {PROFILE_NAME}")
    print(f"Universe: {', '.join(universe)}")
    print("\nSummary")
    print("-------")
    print(summary_df.to_string(index=False))
    print("\nMonthly")
    print("-------")
    print(monthly_df.to_string(index=False))
    print("\nSaved Reports")
    print("-------------")
    print(f"Summary: {paths['summary']}")
    print(f"Trades: {paths['trades']}")
    print(f"Monthly: {paths['monthly']}")


if __name__ == "__main__":
    main()
