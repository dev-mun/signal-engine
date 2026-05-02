from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from algo_backtester.backtests.ema_rsi_backtester import EmaRsiBacktestConfig, EmaRsiPullbackBacktester
from algo_backtester.backtests.four_hour_trend_backtester import (
    FourHourTrendBacktester,
    FourHourTrendConfig,
    prepare_four_hour_data,
)
from algo_backtester.backtests.rsi_bollinger_v2_backtester import (
    RsiBollingerV2BacktestConfig,
    RsiBollingerV2Backtester,
    resolve_ticker_config,
)
from algo_backtester.backtests.swing_options_backtester import (
    SwingOptionsConfig,
    _finalize_signal_conversion,
    _score_sources,
    _selected_source_strategy,
    _source_summary,
)
from algo_backtester.data_loader import load_yfinance_data
from algo_backtester.strategies.ema_rsi_pullback import classify_setup as ema_rsi_classify_setup
from algo_backtester.strategies.four_hour_trend_pullback import classify_setup as four_hour_classify_setup
from algo_backtester.strategies.rsi_bollinger_v2 import classify_setup as rsi_bollinger_v2_classify_setup
from algo_backtester.strategies.swing_options import (
    PLANNER_DISCLAIMER,
    SwingSourceSignal,
    bullish_trend,
    classify_swing_setup,
)
from algo_backtester.watchlists import get_default_watchlist_for_strategy

PROXY_VALIDATION_LABEL = "PROXY VALIDATION ONLY"
MOVE_SCORE = {
    "FAILED": 0,
    "WEAK": 1,
    "SUITABLE": 2,
    "STRONG": 3,
    "EXCELLENT": 4,
}


def _previous_signal_row(signals_df: pd.DataFrame, index_position: int) -> pd.Series:
    if index_position <= 0:
        return signals_df.iloc[index_position]
    return signals_df.iloc[index_position - 1]


def _ema_source_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    bt = EmaRsiPullbackBacktester(config=EmaRsiBacktestConfig())
    _, _, _, signals_df = bt.run(raw_df)

    rows: list[dict] = []
    for i in range(len(signals_df)):
        row = signals_df.iloc[i]
        prev = _previous_signal_row(signals_df, i)
        signal = str(row["Signal"])
        setup = ema_rsi_classify_setup(signal, float(row["RSI"]))
        trend_quality = bullish_trend(
            close_price=float(row["Close"]),
            ema20=float(row["EMA20"]),
            ema50=float(row["EMA50"]),
            ema200=float(row["EMA200"]),
        )
        recent_momentum = float(row["Close"]) > float(prev["Close"]) and float(row["Close"]) >= float(row["EMA20"])
        rows.append(
            {
                "SignalDate": pd.Timestamp(signals_df.index[i]).normalize(),
                "Signal": signal,
                "Setup": setup,
                "Price": float(row["Close"]),
                "RSI": float(row["RSI"]),
                "ATR": float(row["ATR"]),
                "TrendQuality": trend_quality,
                "VolumeConfirmed": float(row["Volume"]) > float(row["AverageVolume20"]),
                "RecentMomentum": recent_momentum,
                "BullishSupport": signal == "BUY" or (setup == "NEAR_SETUP" and trend_quality and recent_momentum),
                "Notes": str(row["Reason"]),
                "Close": float(row["Close"]),
                "EMA20": float(row["EMA20"]),
                "EMA50": float(row["EMA50"]),
                "EMA200": float(row["EMA200"]),
                "Volume": float(row["Volume"]),
                "AverageVolume20": float(row["AverageVolume20"]),
                "PrevClose": float(prev["Close"]),
            }
        )

    return pd.DataFrame(rows).set_index("SignalDate").sort_index()


def _four_hour_source_history(ticker: str, interval: str) -> pd.DataFrame:
    raw_df = prepare_four_hour_data(ticker=ticker, interval=interval)
    if raw_df.empty:
        raise ValueError(f"No usable intraday data found for ticker: {ticker}")

    bt = FourHourTrendBacktester(config=FourHourTrendConfig(interval=interval))
    _, _, _, signals_df = bt.run(raw_df)

    rows: list[dict] = []
    for i in range(len(signals_df)):
        row = signals_df.iloc[i]
        prev = _previous_signal_row(signals_df, i)
        signal = str(row["Signal"])
        setup = four_hour_classify_setup(signal, row)
        trend_quality = bullish_trend(
            close_price=float(row["Close"]),
            ema20=float(row["EMA20"]),
            ema50=float(row["EMA50"]),
            ema200=float(row["EMA200"]),
        )
        recent_momentum = float(row["Close"]) > float(prev["Close"]) and float(row["Close"]) >= float(row["EMA20"])
        rows.append(
            {
                "SignalTimestamp": pd.Timestamp(signals_df.index[i]),
                "SignalDate": pd.Timestamp(signals_df.index[i]).normalize(),
                "Signal": signal,
                "Setup": setup,
                "Price": float(row["Close"]),
                "RSI": float(row["RSI"]),
                "ATR": float(row["ATR"]),
                "TrendQuality": trend_quality,
                "VolumeConfirmed": float(row["Volume"]) > float(row["AverageVolume20"]),
                "RecentMomentum": recent_momentum,
                "BullishSupport": signal == "BUY" or (setup in {"NEEDS_PULLBACK", "NEAR_SETUP"} and trend_quality and recent_momentum),
                "Notes": str(row["Reason"]),
            }
        )

    history_df = pd.DataFrame(rows).sort_values("SignalTimestamp")
    latest_per_day = history_df.groupby("SignalDate", as_index=False).tail(1)
    return latest_per_day.set_index("SignalDate").sort_index()


def _rsi_bollinger_v2_source_history(ticker: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    _, effective_config = resolve_ticker_config(ticker=ticker, config=RsiBollingerV2BacktestConfig())
    bt = RsiBollingerV2Backtester(config=effective_config)
    _, _, _, signals_df = bt.run(raw_df)

    rows: list[dict] = []
    for i in range(len(signals_df)):
        row = signals_df.iloc[i]
        prev = _previous_signal_row(signals_df, i)
        signal = str(row["Signal"])
        setup = rsi_bollinger_v2_classify_setup(signal, row, band_tolerance=effective_config.band_tolerance)
        trend_quality = float(row["Close"]) > float(row["EMA200"]) and float(row["EMA50"]) > float(row["EMA200"])
        recent_momentum = float(row["Close"]) > float(prev["Close"])
        rows.append(
            {
                "SignalDate": pd.Timestamp(signals_df.index[i]).normalize(),
                "Signal": signal,
                "Setup": setup,
                "Price": float(row["Close"]),
                "RSI": float(row["RSI"]),
                "ATR": float(row["ATR"]),
                "TrendQuality": trend_quality,
                "VolumeConfirmed": float(row["Volume"]) >= float(row["AverageVolume20"]) * effective_config.volume_multiplier,
                "RecentMomentum": recent_momentum,
                "BullishSupport": signal == "BUY" or (setup in {"OVERSOLD", "NEAR_SETUP"} and trend_quality and recent_momentum),
                "Notes": str(row["Reason"]),
            }
        )

    return pd.DataFrame(rows).set_index("SignalDate").sort_index()


def _merge_source_histories(
    ema_df: pd.DataFrame,
    four_hour_df: pd.DataFrame,
    v2_df: pd.DataFrame,
) -> pd.DataFrame:
    base_df = ema_df.reset_index().rename(columns={"index": "SignalDate"}).sort_values("SignalDate")
    four_hour_ready = four_hour_df.reset_index().rename(columns={"index": "SignalDate"}).sort_values("SignalDate")
    v2_ready = v2_df.reset_index().rename(columns={"index": "SignalDate"}).sort_values("SignalDate")

    merged = pd.merge_asof(
        base_df,
        four_hour_ready,
        on="SignalDate",
        direction="backward",
        suffixes=("_ema", "_four_hour"),
    )
    merged = pd.merge_asof(
        merged.sort_values("SignalDate"),
        v2_ready,
        on="SignalDate",
        direction="backward",
        suffixes=("", "_v2"),
    )
    merged = merged.rename(
        columns={
            "Signal": "V2Signal",
            "Setup": "V2Setup",
            "Price": "V2Price",
            "RSI": "V2RSI",
            "ATR": "V2ATR",
            "TrendQuality": "V2TrendQuality",
            "VolumeConfirmed": "V2VolumeConfirmed",
            "RecentMomentum": "V2RecentMomentum",
            "BullishSupport": "V2BullishSupport",
            "Notes": "V2Notes",
        }
    )

    required_columns = [
        "Signal_ema",
        "Signal_four_hour",
        "V2Signal",
        "Close",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI_ema",
        "ATR_ema",
    ]
    return merged.dropna(subset=required_columns).set_index("SignalDate").sort_index()


def build_historical_signal_frame(
    ticker: str,
    start: str,
    end: str | None = None,
    config: SwingOptionsConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effective_config = config or SwingOptionsConfig()
    daily_raw = load_yfinance_data(ticker=ticker, start=start, end=end)

    ema_df = _ema_source_history(daily_raw)
    four_hour_df = _four_hour_source_history(ticker=ticker, interval=effective_config.interval)
    v2_df = _rsi_bollinger_v2_source_history(ticker=ticker, raw_df=daily_raw)
    merged_df = _merge_source_histories(ema_df=ema_df, four_hour_df=four_hour_df, v2_df=v2_df)

    rows: list[dict] = []
    audit_rows: list[dict] = []

    for signal_date, row in merged_df.iterrows():
        ema_source = SwingSourceSignal(
            strategy="ema-rsi",
            signal=str(row["Signal_ema"]),
            setup=str(row["Setup_ema"]),
            price=float(row["Price_ema"]),
            rsi=float(row["RSI_ema"]),
            atr=float(row["ATR_ema"]),
            trend_quality=bool(row["TrendQuality_ema"]),
            volume_confirmed=bool(row["VolumeConfirmed_ema"]),
            recent_momentum=bool(row["RecentMomentum_ema"]),
            bullish_support=bool(row["BullishSupport_ema"]),
            notes=str(row["Notes_ema"]),
        )
        four_hour_source = SwingSourceSignal(
            strategy="four-hour-trend",
            signal=str(row["Signal_four_hour"]),
            setup=str(row["Setup_four_hour"]),
            price=float(row["Price_four_hour"]),
            rsi=float(row["RSI_four_hour"]),
            atr=float(row["ATR_four_hour"]),
            trend_quality=bool(row["TrendQuality_four_hour"]),
            volume_confirmed=bool(row["VolumeConfirmed_four_hour"]),
            recent_momentum=bool(row["RecentMomentum_four_hour"]),
            bullish_support=bool(row["BullishSupport_four_hour"]),
            notes=str(row["Notes_four_hour"]),
        )
        v2_source = SwingSourceSignal(
            strategy="rsi-bollinger-v2",
            signal=str(row["V2Signal"]),
            setup=str(row["V2Setup"]),
            price=float(row["V2Price"]),
            rsi=float(row["V2RSI"]),
            atr=float(row["V2ATR"]),
            trend_quality=bool(row["V2TrendQuality"]),
            volume_confirmed=bool(row["V2VolumeConfirmed"]),
            recent_momentum=bool(row["V2RecentMomentum"]),
            bullish_support=bool(row["V2BullishSupport"]),
            notes=str(row["V2Notes"]),
        )
        sources = [ema_source, four_hour_source, v2_source]

        ema_context = {
            "signal_date": str(signal_date.date()),
            "latest": pd.Series(
                {
                    "Close": float(row["Close"]),
                    "EMA20": float(row["EMA20"]),
                    "EMA50": float(row["EMA50"]),
                    "EMA200": float(row["EMA200"]),
                    "RSI": float(row["RSI_ema"]),
                    "ATR": float(row["ATR_ema"]),
                    "Volume": float(row["Volume"]),
                    "AverageVolume20": float(row["AverageVolume20"]),
                }
            ),
            "prev": pd.Series({"Close": float(row["PrevClose"])}),
        }

        score, is_extended, is_weak_trend, supporting_signals = _score_sources(sources=sources, ema_context=ema_context)
        raw_setup = classify_swing_setup(score=score, is_extended=is_extended, is_weak_trend=is_weak_trend)
        selected_source_strategy = _selected_source_strategy(sources)
        conversion = _finalize_signal_conversion(
            ticker=ticker,
            signal_date=str(signal_date.date()),
            score=score,
            raw_setup=raw_setup,
            sources=sources,
            selected_source_strategy=selected_source_strategy,
            latest_close=float(row["Close"]),
            latest_atr=float(row["ATR_ema"]),
            supporting_signals=supporting_signals,
            config=effective_config,
        )
        plan = conversion["plan"]
        premium_budget = float(conversion["premium_budget"])
        premium_over_budget = bool(conversion["premium_over_budget"])
        final_setup = str(conversion["final_setup"])
        signal = str(conversion["signal"])
        has_strong_source = bool(conversion["has_strong_source"])
        source_summary = _source_summary(sources)

        notes_parts = [source_summary]
        if not has_strong_source:
            notes_parts.append("No strong bullish source setup confirmed.")
        if premium_over_budget and plan is not None:
            notes_parts.append(f"Estimated premium risk {plan.max_loss:.2f} exceeds budget {premium_budget:.2f}.")
        notes_parts.append(PROXY_VALIDATION_LABEL)
        notes_parts.append(PLANNER_DISCLAIMER)

        rows.append(
            {
                "SignalDate": str(signal_date.date()),
                "Ticker": ticker,
                "Strategy": "swing-options",
                "Signal": signal,
                "Setup": final_setup,
                "RawSetup": raw_setup,
                "Score": score,
                "SourceSummary": source_summary,
                "Price": float(row["Close"]),
                "RSI": float(row["RSI_ema"]),
                "ATR": float(row["ATR_ema"]),
                "R": float(row["ATR_ema"]),
                "SelectedSourceStrategy": selected_source_strategy,
                "PremiumOverBudget": premium_over_budget,
                "MaxPremiumRisk": 0.0 if plan is None else float(plan.max_loss),
                "Notes": " | ".join(notes_parts),
                "ValidationType": PROXY_VALIDATION_LABEL,
            }
        )
        if raw_setup == "ACTIONABLE":
            audit_row = dict(conversion["audit"])
            audit_row["ValidationType"] = PROXY_VALIDATION_LABEL
            audit_rows.append(audit_row)

    signal_df = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit_rows)
    return signal_df, daily_raw, audit_df


def _days_to_level(window_df: pd.DataFrame, entry_price: float, level_r: float, risk_unit: float) -> int | None:
    target_price = entry_price + (risk_unit * level_r)
    hits = window_df["High"] >= target_price
    if not bool(hits.any()):
        return None
    first_hit = hits[hits].index[0]
    return int(window_df.index.get_loc(first_hit)) + 1


def _days_to_stop(window_df: pd.DataFrame, entry_price: float, risk_unit: float) -> int | None:
    stop_price = entry_price - risk_unit
    hits = window_df["Low"] <= stop_price
    if not bool(hits.any()):
        return None
    first_hit = hits[hits].index[0]
    return int(window_df.index.get_loc(first_hit)) + 1


def classify_proxy_move_quality(
    days_to_1r: int | None,
    days_to_2r: int | None,
    days_to_3r: int | None,
    days_to_stop: int | None,
    available_days: int,
    time_stop_days: int = 5,
) -> tuple[str, str, int, bool]:
    success_3r = days_to_3r is not None and days_to_3r <= 15 and (days_to_stop is None or days_to_3r < days_to_stop)
    success_2r = days_to_2r is not None and days_to_2r <= 10 and (days_to_stop is None or days_to_2r < days_to_stop)
    success_1r = days_to_1r is not None and days_to_1r <= 5 and (days_to_stop is None or days_to_1r < days_to_stop)

    if success_3r:
        return "EXCELLENT", "TARGET_3R", days_to_3r, False
    if success_2r:
        return "STRONG", "TARGET_2R", days_to_2r, False
    if success_1r:
        return "SUITABLE", "TARGET_1R", days_to_1r, False
    if days_to_stop is not None:
        return "FAILED", "STOP_1R", days_to_stop, True

    hold_days = min(time_stop_days, available_days)
    exit_reason = "TIME_STOP" if hold_days >= time_stop_days else "END_OF_DATA"
    return "WEAK", exit_reason, hold_days, False


def evaluate_proxy_trade(
    ticker: str,
    signal_date: str,
    atr: float,
    daily_df: pd.DataFrame,
    max_hold_days: int = 15,
    time_stop_days: int = 5,
) -> dict | None:
    daily_prices = daily_df.copy()
    if "Date" in daily_prices.columns:
        daily_prices["Date"] = pd.to_datetime(daily_prices["Date"])
        daily_prices = daily_prices.set_index("Date")
    daily_prices.index = pd.to_datetime(daily_prices.index)
    daily_prices = daily_prices.sort_index()

    signal_timestamp = pd.Timestamp(signal_date)
    if signal_timestamp not in daily_prices.index:
        return None

    signal_position = int(daily_prices.index.get_loc(signal_timestamp))
    entry_position = signal_position + 1
    if entry_position >= len(daily_prices):
        return None

    entry_date = daily_prices.index[entry_position]
    window_df = daily_prices.iloc[entry_position: entry_position + max_hold_days].copy()
    if window_df.empty:
        return None

    entry_price = float(daily_prices.iloc[entry_position]["Open"])
    mfe = float((window_df["High"] - entry_price).max())
    mae = float((window_df["Low"] - entry_price).min())
    days_to_1r = _days_to_level(window_df=window_df, entry_price=entry_price, level_r=1.0, risk_unit=atr)
    days_to_2r = _days_to_level(window_df=window_df, entry_price=entry_price, level_r=2.0, risk_unit=atr)
    days_to_3r = _days_to_level(window_df=window_df, entry_price=entry_price, level_r=3.0, risk_unit=atr)
    days_to_stop = _days_to_stop(window_df=window_df, entry_price=entry_price, risk_unit=atr)
    move_quality, exit_reason, hold_days, failed = classify_proxy_move_quality(
        days_to_1r=days_to_1r,
        days_to_2r=days_to_2r,
        days_to_3r=days_to_3r,
        days_to_stop=days_to_stop,
        available_days=len(window_df),
        time_stop_days=time_stop_days,
    )

    return {
        "Ticker": ticker,
        "SignalDate": str(signal_timestamp.date()),
        "EntryDate": str(entry_date.date()),
        "EntryPrice": round(entry_price, 4),
        "ATR": round(float(atr), 4),
        "R": round(float(atr), 4),
        "MFE": round(mfe, 4),
        "MAE": round(mae, 4),
        "MFE_R": round(mfe / atr, 4) if atr > 0 else 0.0,
        "MAE_R": round(mae / atr, 4) if atr > 0 else 0.0,
        "DaysTo1R": days_to_1r,
        "DaysTo2R": days_to_2r,
        "DaysTo3R": days_to_3r,
        "DaysToStop": days_to_stop,
        "HoldDays": hold_days,
        "Reached1R": bool(days_to_1r is not None),
        "Reached2R": bool(days_to_2r is not None),
        "Reached3R": bool(days_to_3r is not None),
        "Failed": failed,
        "ExitReason": exit_reason,
        "MoveQuality": move_quality,
        "ValidationType": PROXY_VALIDATION_LABEL,
    }


def backtest_ticker_proxy(
    ticker: str,
    start: str,
    end: str | None = None,
    config: SwingOptionsConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effective_config = config or SwingOptionsConfig()
    signal_df, daily_df, audit_df = build_historical_signal_frame(
        ticker=ticker,
        start=start,
        end=end,
        config=effective_config,
    )

    trades: list[dict] = []
    for row in signal_df.itertuples(index=False):
        if row.Signal != "BUY":
            continue

        trade = evaluate_proxy_trade(
            ticker=ticker,
            signal_date=row.SignalDate,
            atr=float(row.ATR),
            daily_df=daily_df,
            max_hold_days=effective_config.max_hold_days,
            time_stop_days=effective_config.time_stop_days,
        )
        if trade is None:
            continue
        trades.append(trade)

    return signal_df, pd.DataFrame(trades), audit_df


def _monthly_summary(signal_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame(
            columns=[
                "ValidationType",
                "Month",
                "BUYSignals",
                "ActionableCount",
                "TradesTriggered",
                "EXCELLENT",
                "STRONG",
                "SUITABLE",
                "WEAK",
                "FAILED",
                "AverageHoldDays",
                "AverageMFE_R",
                "AverageMAE_R",
            ]
        )

    signals = signal_df.copy()
    signals["Month"] = pd.to_datetime(signals["SignalDate"]).dt.to_period("M").astype(str)
    trades = trades_df.copy()
    if not trades.empty:
        trades["Month"] = pd.to_datetime(trades["SignalDate"]).dt.to_period("M").astype(str)

    summary = pd.DataFrame({"Month": sorted(signals["Month"].unique())})
    summary["BUYSignals"] = summary["Month"].map(signals.groupby("Month")["Signal"].apply(lambda s: int((s == "BUY").sum()))).fillna(0).astype(int)
    summary["ActionableCount"] = summary["Month"].map(signals.groupby("Month")["RawSetup"].apply(lambda s: int((s == "ACTIONABLE").sum()))).fillna(0).astype(int)
    summary["TradesTriggered"] = 0 if trades.empty else summary["Month"].map(trades.groupby("Month").size()).fillna(0).astype(int)

    for quality in ["EXCELLENT", "STRONG", "SUITABLE", "WEAK", "FAILED"]:
        if trades.empty:
            summary[quality] = 0
        else:
            counts = trades.groupby("Month")["MoveQuality"].apply(lambda s, q=quality: int((s == q).sum()))
            summary[quality] = summary["Month"].map(counts).fillna(0).astype(int)

    if trades.empty:
        summary["AverageHoldDays"] = 0.0
        summary["AverageMFE_R"] = 0.0
        summary["AverageMAE_R"] = 0.0
    else:
        summary["AverageHoldDays"] = summary["Month"].map(trades.groupby("Month")["HoldDays"].mean()).fillna(0.0).round(2)
        summary["AverageMFE_R"] = summary["Month"].map(trades.groupby("Month")["MFE_R"].mean()).fillna(0.0).round(2)
        summary["AverageMAE_R"] = summary["Month"].map(trades.groupby("Month")["MAE_R"].mean()).fillna(0.0).round(2)

    summary.insert(0, "ValidationType", PROXY_VALIDATION_LABEL)
    return summary


def _rank_tickers(trades_df: pd.DataFrame) -> tuple[str, str]:
    if trades_df.empty:
        return "N/A", "N/A"

    ranked = trades_df.copy()
    ranked["MoveScore"] = ranked["MoveQuality"].map(MOVE_SCORE).fillna(0)
    ticker_scores = ranked.groupby("Ticker").agg(
        AverageMoveScore=("MoveScore", "mean"),
        AverageMFE_R=("MFE_R", "mean"),
        TradeCount=("Ticker", "size"),
    )
    ordered = ticker_scores.sort_values(
        by=["AverageMoveScore", "AverageMFE_R", "TradeCount"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return str(ordered.index[0]), str(ordered.index[-1])


def _summary_report(signal_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    total_buy_signals = int((signal_df["Signal"] == "BUY").sum()) if not signal_df.empty else 0
    total_months = max(1, pd.to_datetime(signal_df["SignalDate"]).dt.to_period("M").nunique()) if not signal_df.empty else 1
    best_ticker, worst_ticker = _rank_tickers(trades_df)
    total_trades = len(trades_df)

    if trades_df.empty:
        summary_row = {
            "ValidationType": PROXY_VALIDATION_LABEL,
            "Scope": "Underlying move quality only",
            "Disclaimer": PLANNER_DISCLAIMER,
            "TotalBUYSignals": total_buy_signals,
            "TradesPerMonth": 0.0,
            "PctReached1RWithin5D": 0.0,
            "PctReached2RWithin10D": 0.0,
            "PctReached3RWithin15D": 0.0,
            "PctFailed": 0.0,
            "AverageMFE_R": 0.0,
            "AverageMAE_R": 0.0,
            "AverageHoldDays": 0.0,
            "BestTicker": best_ticker,
            "WorstTicker": worst_ticker,
        }
        return pd.DataFrame([summary_row])

    summary_row = {
        "ValidationType": PROXY_VALIDATION_LABEL,
        "Scope": "Underlying move quality only",
        "Disclaimer": PLANNER_DISCLAIMER,
        "TotalBUYSignals": total_buy_signals,
        "TradesPerMonth": round(total_trades / total_months, 2),
        "PctReached1RWithin5D": round((trades_df["MoveQuality"].isin({"SUITABLE", "STRONG", "EXCELLENT"}).mean() * 100), 2),
        "PctReached2RWithin10D": round((trades_df["MoveQuality"].isin({"STRONG", "EXCELLENT"}).mean() * 100), 2),
        "PctReached3RWithin15D": round(((trades_df["MoveQuality"] == "EXCELLENT").mean() * 100), 2),
        "PctFailed": round(((trades_df["MoveQuality"] == "FAILED").mean() * 100), 2),
        "AverageMFE_R": round(float(trades_df["MFE_R"].mean()), 2),
        "AverageMAE_R": round(float(trades_df["MAE_R"].mean()), 2),
        "AverageHoldDays": round(float(trades_df["HoldDays"].mean()), 2),
        "BestTicker": best_ticker,
        "WorstTicker": worst_ticker,
    }
    return pd.DataFrame([summary_row])


def save_proxy_backtest_outputs(
    summary_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    output_dir: str = "reports",
) -> dict[str, Path]:
    output_path = Path(output_dir) / "swing_options"
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "proxy_backtest_summary.csv"
    trades_path = output_path / "proxy_backtest_trades.csv"
    monthly_path = output_path / "proxy_backtest_monthly.csv"
    audit_path = output_path / "actionable_audit.csv"

    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    return {
        "summary": summary_path,
        "trades": trades_path,
        "monthly": monthly_path,
        "audit": audit_path,
    }


def run_proxy_backtest(
    tickers: Iterable[str] | None = None,
    start: str = "2024-05-01",
    end: str | None = None,
    config: SwingOptionsConfig | None = None,
    output_dir: str = "reports",
) -> dict[str, object]:
    effective_tickers = list(tickers) if tickers is not None else get_default_watchlist_for_strategy("swing-options")

    all_signals: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    errors: list[dict] = []

    for ticker in effective_tickers:
        clean_ticker = str(ticker).strip().upper()
        if not clean_ticker:
            continue

        try:
            signal_df, trades_df, audit_df = backtest_ticker_proxy(
                ticker=clean_ticker,
                start=start,
                end=end,
                config=config,
            )
            if not signal_df.empty:
                all_signals.append(signal_df)
            if not trades_df.empty:
                all_trades.append(trades_df)
            if not audit_df.empty:
                all_audits.append(audit_df)
        except Exception as exc:
            errors.append(
                {
                    "Ticker": clean_ticker,
                    "ValidationType": PROXY_VALIDATION_LABEL,
                    "Error": str(exc),
                }
            )

    signal_df = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    audit_df = pd.concat(all_audits, ignore_index=True) if all_audits else pd.DataFrame(
        columns=[
            "Ticker",
            "SignalDate",
            "RawScore",
            "PreFinalSetup",
            "HasStrongSource",
            "SupportingSignals",
            "SelectedSourceStrategy",
            "PlanCreated",
            "PremiumBudget",
            "EstimatedPremium",
            "PremiumOverBudget",
            "FinalSetup",
            "FinalSignal",
            "BlockReason",
            "ValidationType",
        ]
    )
    monthly_df = _monthly_summary(signal_df=signal_df, trades_df=trades_df)
    summary_df = _summary_report(signal_df=signal_df, trades_df=trades_df)
    paths = save_proxy_backtest_outputs(
        summary_df=summary_df,
        trades_df=trades_df,
        monthly_df=monthly_df,
        audit_df=audit_df,
        output_dir=output_dir,
    )

    return {
        "signals": signal_df,
        "trades": trades_df,
        "audit": audit_df,
        "monthly": monthly_df,
        "summary": summary_df,
        "errors": pd.DataFrame(errors),
        "paths": paths,
    }
