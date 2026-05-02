from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algo_backtester.backtests.swing_options_backtester import (
    SMALL_ACCOUNT_OPTIONS_PROFILE,
    SMALL_ACCOUNT_OPTIONS_RULES,
    SwingOptionsConfig,
    apply_execution_profile_labels,
)
from algo_backtester.backtests.swing_options_proxy_backtester import (
    evaluate_proxy_trade,
    build_historical_signal_frame,
)
from algo_backtester.watchlists import get_watchlist

SMALL_ACCOUNT_OUTPUT_DIR = Path("reports") / "swing_options"
PREMIUM_STATUS_ORDER = {"OK": 0, "ACCEPTABLE": 1}
SUITABLE_PLUS = {"SUITABLE", "STRONG", "EXCELLENT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the historical small-account swing-options workflow.")
    parser.add_argument("--start", type=str, default="2024-05-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports")
    return parser.parse_args()


def _month_count(signal_df: pd.DataFrame) -> int:
    if signal_df.empty:
        return 1
    return max(1, pd.to_datetime(signal_df["SignalDate"]).dt.to_period("M").nunique())


def _label_historical_signal_rows(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df.copy()

    labeled_rows: list[dict] = []
    for row in signal_df.to_dict(orient="records"):
        base_row = dict(row)
        base_row["MaxLoss"] = float(base_row.get("MaxPremiumRisk", 0.0) or 0.0)
        base_row["EstPremium"] = round(base_row["MaxLoss"] / 100.0, 2) if base_row["MaxLoss"] > 0 else 0.0
        base_row["Reason"] = str(base_row.get("Notes", ""))
        labeled_rows.append(
            apply_execution_profile_labels(
                result=base_row,
                account_profile=SMALL_ACCOUNT_OPTIONS_PROFILE,
            )
        )

    return pd.DataFrame(labeled_rows)


def build_small_account_signal_history(
    tickers: list[str],
    start: str,
    end: str | None = None,
    config: SwingOptionsConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    effective_config = config or SwingOptionsConfig()
    signal_frames: list[pd.DataFrame] = []
    daily_data_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        clean_ticker = str(ticker).strip().upper()
        if not clean_ticker:
            continue

        signal_df, daily_df, _ = build_historical_signal_frame(
            ticker=clean_ticker,
            start=start,
            end=end,
            config=effective_config,
        )
        labeled_df = _label_historical_signal_rows(signal_df)
        if not labeled_df.empty:
            signal_frames.append(labeled_df)
        daily_data_by_ticker[clean_ticker] = daily_df

    combined_df = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return combined_df, daily_data_by_ticker


def affordable_small_account_buys(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df.copy()

    affordable_df = signal_df[
        (signal_df["Signal"] == "BUY")
        & (signal_df["SmallAccountEligible"] == "YES")
        & (signal_df["PremiumStatus"].isin({"OK", "ACCEPTABLE"}))
    ].copy()
    if affordable_df.empty:
        return affordable_df

    affordable_df["PremiumStatusRank"] = affordable_df["PremiumStatus"].map(PREMIUM_STATUS_ORDER).fillna(99)
    return affordable_df


def select_daily_candidates(affordable_df: pd.DataFrame) -> pd.DataFrame:
    if affordable_df.empty:
        return affordable_df.copy()

    ordered = affordable_df.sort_values(
        by=["SignalDate", "PremiumStatusRank", "Score", "EstPremium", "Ticker"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    selected = ordered.groupby("SignalDate", as_index=False).head(1).reset_index(drop=True)
    return selected.drop(columns=["PremiumStatusRank"], errors="ignore")


def _business_idle_days(previous_exit_date: str | None, next_entry_date: str) -> int:
    if previous_exit_date is None:
        return 0
    gap = len(pd.bdate_range(pd.Timestamp(previous_exit_date), pd.Timestamp(next_entry_date))) - 2
    return max(gap, 0)


def _derive_exit_date(
    trade: dict,
    daily_df: pd.DataFrame,
) -> str:
    if trade.get("ExitDate"):
        return str(trade["ExitDate"])

    frame = daily_df.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    entry_date = pd.Timestamp(str(trade["EntryDate"]))
    if entry_date not in frame.index:
        return str(entry_date.date())

    entry_position = int(frame.index.get_loc(entry_date))
    hold_days = max(int(trade.get("HoldDays", 1) or 1), 1)
    exit_position = min(entry_position + hold_days - 1, len(frame.index) - 1)
    return str(frame.index[exit_position].date())


def replay_small_account_candidates(
    candidate_df: pd.DataFrame,
    daily_data_by_ticker: dict[str, pd.DataFrame],
    config: SwingOptionsConfig | None = None,
    trade_evaluator: Callable[..., dict | None] | None = None,
) -> pd.DataFrame:
    effective_config = config or SwingOptionsConfig()
    evaluator = trade_evaluator or evaluate_proxy_trade
    trades: list[dict] = []
    current_exit_date: str | None = None
    previous_exit_date: str | None = None

    ordered_candidates = candidate_df.sort_values(by=["SignalDate", "PremiumStatus", "Score"], ascending=[True, True, False])

    for row in ordered_candidates.itertuples(index=False):
        signal_date = str(row.SignalDate)
        if current_exit_date is not None and pd.Timestamp(signal_date) <= pd.Timestamp(current_exit_date):
            continue

        ticker = str(row.Ticker)
        daily_df = daily_data_by_ticker[ticker]
        trade = evaluator(
            ticker=ticker,
            signal_date=signal_date,
            atr=float(row.ATR),
            daily_df=daily_df,
            max_hold_days=effective_config.max_hold_days,
            time_stop_days=effective_config.time_stop_days,
        )
        if trade is None:
            continue

        exit_date = _derive_exit_date(trade=trade, daily_df=daily_df)
        idle_days = _business_idle_days(previous_exit_date=previous_exit_date, next_entry_date=str(trade["EntryDate"]))
        previous_exit_date = exit_date
        current_exit_date = exit_date

        trades.append(
            {
                "AccountProfile": SMALL_ACCOUNT_OPTIONS_PROFILE,
                "SignalDate": signal_date,
                "EntryDate": str(trade["EntryDate"]),
                "ExitDate": exit_date,
                "Ticker": ticker,
                "EstimatedPremium": float(row.EstPremium),
                "PremiumStatus": str(row.PremiumStatus),
                "SmallAccountEligible": str(row.SmallAccountEligible),
                "MoveQuality": str(trade["MoveQuality"]),
                "HoldDays": int(trade["HoldDays"]),
                "ExitReason": str(trade["ExitReason"]),
                "MFE_R": float(trade["MFE_R"]),
                "MAE_R": float(trade["MAE_R"]),
                "Score": float(row.Score),
                "Setup": str(row.Setup),
                "IdleDaysBeforeTrade": idle_days,
            }
        )

    return pd.DataFrame(trades)


def build_small_account_monthly_report(signal_df: pd.DataFrame, affordable_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame(
            columns=[
                "AccountProfile",
                "Month",
                "SmallAccountBUYs",
                "AffordableBUYs",
                "TradesTriggered",
                "EXCELLENT",
                "STRONG",
                "SUITABLE",
                "WEAK",
                "FAILED",
            ]
        )

    signal_months = pd.to_datetime(signal_df["SignalDate"]).dt.to_period("M").astype(str)
    months = sorted(signal_months.unique())
    monthly_df = pd.DataFrame({"Month": months})
    monthly_df.insert(0, "AccountProfile", SMALL_ACCOUNT_OPTIONS_PROFILE)

    signal_buy_df = signal_df[signal_df["Signal"] == "BUY"].copy()
    if not signal_buy_df.empty:
        signal_buy_df["Month"] = pd.to_datetime(signal_buy_df["SignalDate"]).dt.to_period("M").astype(str)
        monthly_df["SmallAccountBUYs"] = monthly_df["Month"].map(signal_buy_df.groupby("Month").size()).fillna(0).astype(int)
    else:
        monthly_df["SmallAccountBUYs"] = 0

    if not affordable_df.empty:
        affordable_copy = affordable_df.copy()
        affordable_copy["Month"] = pd.to_datetime(affordable_copy["SignalDate"]).dt.to_period("M").astype(str)
        monthly_df["AffordableBUYs"] = monthly_df["Month"].map(affordable_copy.groupby("Month").size()).fillna(0).astype(int)
    else:
        monthly_df["AffordableBUYs"] = 0

    if not trades_df.empty:
        trades_copy = trades_df.copy()
        trades_copy["Month"] = pd.to_datetime(trades_copy["SignalDate"]).dt.to_period("M").astype(str)
        monthly_df["TradesTriggered"] = monthly_df["Month"].map(trades_copy.groupby("Month").size()).fillna(0).astype(int)
        for quality in ["EXCELLENT", "STRONG", "SUITABLE", "WEAK", "FAILED"]:
            counts = trades_copy.groupby("Month")["MoveQuality"].apply(lambda s, q=quality: int((s == q).sum()))
            monthly_df[quality] = monthly_df["Month"].map(counts).fillna(0).astype(int)
    else:
        monthly_df["TradesTriggered"] = 0
        for quality in ["EXCELLENT", "STRONG", "SUITABLE", "WEAK", "FAILED"]:
            monthly_df[quality] = 0

    return monthly_df


def build_small_account_summary(signal_df: pd.DataFrame, affordable_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    total_months = _month_count(signal_df)
    signal_buy_df = signal_df[signal_df["Signal"] == "BUY"].copy() if not signal_df.empty else pd.DataFrame()
    total_small_account_buys = len(signal_buy_df)
    total_affordable_buys = len(affordable_df)
    total_trades = len(trades_df)

    if trades_df.empty:
        row = {
            "AccountProfile": SMALL_ACCOUNT_OPTIONS_PROFILE,
            "Universe": ",".join(get_watchlist(SMALL_ACCOUNT_OPTIONS_PROFILE)),
            "TotalSmallAccountBUYs": total_small_account_buys,
            "AverageSmallAccountBUYsPerMonth": round(total_small_account_buys / total_months, 2),
            "AverageAffordableBUYsPerMonth": round(total_affordable_buys / total_months, 2),
            "TradesTriggeredPerMonth": 0.0,
            "PctSuitablePlus": 0.0,
            "PctFailed": 0.0,
            "AverageHoldDays": 0.0,
            "AverageMFE_R": 0.0,
            "AverageMAE_R": 0.0,
            "AverageIdleDaysBetweenTrades": 0.0,
        }
        return pd.DataFrame([row])

    row = {
        "AccountProfile": SMALL_ACCOUNT_OPTIONS_PROFILE,
        "Universe": ",".join(get_watchlist(SMALL_ACCOUNT_OPTIONS_PROFILE)),
        "TotalSmallAccountBUYs": total_small_account_buys,
        "AverageSmallAccountBUYsPerMonth": round(total_small_account_buys / total_months, 2),
        "AverageAffordableBUYsPerMonth": round(total_affordable_buys / total_months, 2),
        "TradesTriggeredPerMonth": round(total_trades / total_months, 2),
        "PctSuitablePlus": round((trades_df["MoveQuality"].isin(SUITABLE_PLUS).mean() * 100), 2),
        "PctFailed": round(((trades_df["MoveQuality"] == "FAILED").mean() * 100), 2),
        "AverageHoldDays": round(float(trades_df["HoldDays"].mean()), 2),
        "AverageMFE_R": round(float(trades_df["MFE_R"].mean()), 2),
        "AverageMAE_R": round(float(trades_df["MAE_R"].mean()), 2),
        "AverageIdleDaysBetweenTrades": round(float(trades_df["IdleDaysBeforeTrade"].iloc[1:].mean()), 2) if len(trades_df) > 1 else 0.0,
    }
    return pd.DataFrame([row])


def save_small_account_outputs(
    summary_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    output_dir: str = "reports",
) -> dict[str, Path]:
    output_path = Path(output_dir) / "swing_options"
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "small_account_scan_summary.csv"
    trades_path = output_path / "small_account_scan_trades.csv"
    monthly_path = output_path / "small_account_scan_monthly.csv"

    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

    return {
        "summary": summary_path,
        "trades": trades_path,
        "monthly": monthly_path,
    }


def run_small_account_scan_backtest(
    start: str = "2024-05-01",
    end: str | None = None,
    output_dir: str = "reports",
    config: SwingOptionsConfig | None = None,
) -> dict[str, object]:
    tickers = get_watchlist(SMALL_ACCOUNT_OPTIONS_PROFILE)
    signal_df, daily_data_by_ticker = build_small_account_signal_history(
        tickers=tickers,
        start=start,
        end=end,
        config=config,
    )
    affordable_df = affordable_small_account_buys(signal_df)
    daily_candidates = select_daily_candidates(affordable_df)
    trades_df = replay_small_account_candidates(
        candidate_df=daily_candidates,
        daily_data_by_ticker=daily_data_by_ticker,
        config=config,
    )
    monthly_df = build_small_account_monthly_report(
        signal_df=signal_df,
        affordable_df=affordable_df,
        trades_df=trades_df,
    )
    summary_df = build_small_account_summary(
        signal_df=signal_df,
        affordable_df=affordable_df,
        trades_df=trades_df,
    )
    paths = save_small_account_outputs(
        summary_df=summary_df,
        trades_df=trades_df,
        monthly_df=monthly_df,
        output_dir=output_dir,
    )
    return {
        "signals": signal_df,
        "affordable_buys": affordable_df,
        "daily_candidates": daily_candidates,
        "trades": trades_df,
        "monthly": monthly_df,
        "summary": summary_df,
        "paths": paths,
    }


def main() -> None:
    args = parse_args()
    payload = run_small_account_scan_backtest(
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )

    print("Small Account Swing Options Replay")
    print("----------------------------------")
    print(f"Universe: {', '.join(get_watchlist(SMALL_ACCOUNT_OPTIONS_PROFILE))}")
    print(f"Account size: {int(SMALL_ACCOUNT_OPTIONS_RULES['account_size'])}")
    print(f"Max premium: {int(SMALL_ACCOUNT_OPTIONS_RULES['max_premium'])}")
    print(f"Preferred premium: {int(SMALL_ACCOUNT_OPTIONS_RULES['preferred_premium_min'])}-{int(SMALL_ACCOUNT_OPTIONS_RULES['preferred_premium_max'])}")

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


if __name__ == "__main__":
    main()
