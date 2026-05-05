from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.tseries.offsets import BDay

from algo_backtester.backtests.swing_options_backtester import (
    SwingOptionsConfig,
    _finalize_signal_conversion,
    _score_sources,
    _selected_source_strategy,
    analyze_ticker as analyze_swing_options_ticker,
    evaluate_source_signals,
)
from algo_backtester.backtests.swing_options_proxy_backtester import (
    _ema_source_history,
    _four_hour_source_history,
    _merge_source_histories,
    _rsi_bollinger_v2_source_history,
    evaluate_proxy_trade,
)
from algo_backtester.strategies.swing_options_debit_spread import (
    DEBIT_SPREAD_PLANNER_DISCLAIMER,
    PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
    DebitSpreadPlan,
    build_bull_call_debit_spread,
)
from algo_backtester.data_loader import load_yfinance_data
from algo_backtester.strategies.swing_options import SwingSourceSignal, classify_swing_setup
from algo_backtester.watchlists import get_default_watchlist_for_strategy


@dataclass(frozen=True)
class SwingOptionsDebitSpreadConfig:
    account_size: float = 3_000.0
    max_contracts: int = 1
    max_debit: float = 150.0
    preferred_debit_min: float = 50.0
    preferred_debit_max: float = 125.0
    min_dte: int = 30
    max_dte: int = 60
    preferred_dte: int = 45
    max_hold_days: int = 15
    time_stop_days: int = 5
    interval: str = "1h"
    mode: str = "tuned"


def _planned_execution_date(signal_date: str) -> str:
    return str((pd.Timestamp(signal_date) + BDay(1)).date())


def _hard_blocked(setup: str, plan: DebitSpreadPlan | None) -> bool:
    if setup in {"EXTENDED", "WEAK_TREND"}:
        return True
    if plan is None:
        return False
    return plan.premium_status in {"BAD_REWARD_RISK", "TOO_EXPENSIVE"}


def _near_actionable_support(
    sources: list[SwingSourceSignal],
    close_price: float,
    ema50: float,
    rsi: float,
    prev_close: float,
) -> tuple[bool, int]:
    qualifying = [
        source
        for source in sources
        if source.setup in {"WAIT", "NEAR_SETUP", "NEEDS_PULLBACK"} and source.trend_quality
    ]
    improving_momentum = close_price > prev_close and sum(1 for source in qualifying if source.recent_momentum) >= 2
    qualifies = (
        len(qualifying) >= 2
        and improving_momentum
        and close_price > ema50
        and 45.0 <= rsi <= 68.0
    )
    return qualifies, len(qualifying)


def _resolve_debit_spread_signal(
    *,
    raw_setup: str,
    strict_signal: str,
    score: float,
    plan: DebitSpreadPlan | None,
    sources: list[SwingSourceSignal],
    close_price: float,
    ema50: float,
    rsi: float,
    prev_close: float,
    mode: str,
) -> tuple[str, str, str]:
    premium_status = "N/A" if plan is None else plan.premium_status
    affordable = plan is not None and premium_status in {"OK", "ACCEPTABLE"} and plan.small_account_eligible

    if _hard_blocked(raw_setup, plan):
        return "HOLD", raw_setup, "HARD_BLOCKER"

    strict_buy = strict_signal == "BUY" and affordable
    if mode == "strict":
        return ("BUY", raw_setup, "STRICT_BUY") if strict_buy else ("HOLD", raw_setup, "STRICT_NO_BUY")

    if strict_buy:
        return "BUY", raw_setup, "STRICT_BUY"

    watchlist_buy = raw_setup == "WATCHLIST" and score >= 68.0 and affordable
    if watchlist_buy:
        return "BUY", "WATCHLIST", "TUNED_WATCHLIST_BUY"

    near_actionable, qualifying_sources = _near_actionable_support(
        sources=sources,
        close_price=close_price,
        ema50=ema50,
        rsi=rsi,
        prev_close=prev_close,
    )
    if raw_setup in {"WATCHLIST", "WAIT"} and near_actionable and affordable:
        return "BUY", "WATCHLIST", f"TUNED_NEAR_ACTIONABLE_{qualifying_sources}_SOURCES"

    return "HOLD", raw_setup, "TUNED_NO_BUY"


def _journal_fields(result: dict, plan: DebitSpreadPlan | None) -> dict:
    if plan is None or result["Signal"] != "BUY":
        return {
            "OptionsAction": "NO_OPTIONS_TRADE",
            "Structure": "No trade",
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
        }

    return {
        "OptionsAction": "PLAN_BULL_CALL_DEBIT_SPREAD",
        "Structure": plan.option_structure,
        "Expiration": plan.expiration,
        "LongStrike": plan.long_strike,
        "ShortStrike": plan.short_strike,
        "EstimatedDebit": plan.est_debit,
        "MaxLoss": plan.max_loss,
        "MaxProfit": plan.max_profit,
        "TradeQuality": result["PremiumStatus"],
        "PlannedEntryReference": result["Price"],
        "StopLoss": round(plan.est_debit * 0.50, 2),
        "TakeProfit": plan.max_profit,
        "RiskPerShare": round(plan.est_debit * 0.50, 2),
        "RewardPerShare": round(plan.max_profit / 100.0, 2),
    }


def _blocker_reason(raw_setup: str, plan: DebitSpreadPlan | None) -> str:
    if raw_setup == "EXTENDED":
        return "Blocked because the setup is EXTENDED."
    if raw_setup == "WEAK_TREND":
        return "Blocked because the setup is WEAK_TREND."
    if plan is None:
        return ""
    if plan.premium_status == "TOO_EXPENSIVE":
        return "Blocked because estimated debit exceeds the $150 small-account cap."
    if plan.premium_status == "BAD_REWARD_RISK":
        return "Blocked because estimated reward/risk is below the 1.5 minimum."
    return ""


def _reason_text(
    *,
    signal: str,
    plan: DebitSpreadPlan | None,
    conversion_reason: str,
    raw_setup: str,
) -> str:
    if plan is None:
        return "No debit spread plan generated because the underlying swing-options signal remained non-actionable."

    if conversion_reason == "STRICT_BUY":
        return "Debit spread plan generated from confirmed swing-options BUY signal."

    if conversion_reason in {"TUNED_WATCHLIST_BUY"} or conversion_reason.startswith("TUNED_NEAR_ACTIONABLE_"):
        return (
            "Base swing-options signal was HOLD. Tuned debit-spread conversion upgraded this setup "
            "to BUY due to near-actionable bullish source alignment."
        )

    if signal != "BUY":
        blocker = _blocker_reason(raw_setup=raw_setup, plan=plan)
        base = "No debit spread plan generated because the underlying swing-options signal remained non-actionable."
        return f"{base} {blocker}".strip()

    return "Debit spread plan generated from confirmed swing-options BUY signal."


def _build_result_from_underlying(
    ticker: str,
    underlying_result: dict,
    plan: DebitSpreadPlan | None,
    *,
    mode: str,
    sources: list[SwingSourceSignal],
    close_price: float,
    ema50: float,
    rsi: float,
    prev_close: float,
) -> dict:
    underlying_signal = str(underlying_result["Signal"])
    raw_setup = str(underlying_result["Setup"])
    signal, final_setup, conversion_reason = _resolve_debit_spread_signal(
        raw_setup=raw_setup,
        strict_signal=underlying_signal,
        score=float(underlying_result["Score"]),
        plan=plan,
        sources=sources,
        close_price=close_price,
        ema50=ema50,
        rsi=rsi,
        prev_close=prev_close,
        mode=mode,
    )
    eligible = plan is not None and plan.small_account_eligible and signal == "BUY"
    premium_status = "N/A" if plan is None else plan.premium_status
    reason = _reason_text(
        signal=signal,
        plan=plan,
        conversion_reason=conversion_reason,
        raw_setup=raw_setup,
    )
    option_reason = DEBIT_SPREAD_PLANNER_DISCLAIMER if plan is None else plan.notes

    result = {
        "Ticker": ticker,
        "Strategy": "swing-options-debit-spread",
        "Signal": signal,
        "Setup": final_setup,
        "Score": float(underlying_result["Score"]),
        "Price": float(underlying_result["Price"]),
        "OptionStructure": "N/A" if plan is None else plan.option_structure,
        "LongStrike": 0.0 if plan is None else plan.long_strike,
        "ShortStrike": 0.0 if plan is None else plan.short_strike,
        "DTE": 0 if plan is None else plan.dte,
        "EstDebit": 0.0 if plan is None else plan.est_debit,
        "MaxLoss": 0.0 if plan is None else plan.max_loss,
        "MaxProfit": 0.0 if plan is None else plan.max_profit,
        "RewardRisk": 0.0 if plan is None else plan.reward_risk,
        "SpreadWidth": 0.0 if plan is None else plan.spread_width,
        "PremiumStatus": premium_status,
        "SmallAccountEligible": "YES" if eligible else "NO",
        "Reason": reason,
        "SourceSummary": str(underlying_result.get("SourceSummary", "")),
        "RSI": float(underlying_result.get("RSI", 0.0)),
        "ATR": float(underlying_result.get("ATR", 0.0)),
        "SignalDate": str(underlying_result["SignalDate"]),
        "PlannedExecutionDate": str(underlying_result.get("PlannedExecutionDate", _planned_execution_date(str(underlying_result["SignalDate"])))),
        "UniverseStatus": "ELIGIBLE" if signal == "BUY" else "WATCH",
        "UniverseReason": reason,
        "SetupStatus": final_setup,
        "DistanceToSetup": final_setup,
        "Equity": float(underlying_result.get("Equity", 0.0)),
        "OptionsReason": option_reason,
        "AvgDollarVolume": float(underlying_result.get("AvgDollarVolume", 0.0)),
        "EarningsDate": str(underlying_result.get("EarningsDate", "PLACEHOLDER_OK")),
        "Mode": mode,
        "ConversionReason": conversion_reason,
    }
    result.update(_journal_fields(result=result, plan=plan))
    return result


def analyze_ticker(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    config: SwingOptionsDebitSpreadConfig | None = None,
) -> dict:
    effective_config = config or SwingOptionsDebitSpreadConfig()
    underlying_analysis = analyze_swing_options_ticker(
        ticker=ticker,
        start=start,
        end=end,
        config=SwingOptionsConfig(
            interval=effective_config.interval,
            max_hold_days=effective_config.max_hold_days,
            time_stop_days=effective_config.time_stop_days,
        ),
    )
    underlying_result = underlying_analysis["result"]
    source_evaluation = evaluate_source_signals(
        ticker=ticker,
        start=start,
        end=end,
        interval=effective_config.interval,
    )
    sources = [SwingSourceSignal(**source) for source in underlying_analysis["sources"]]
    ema_context = source_evaluation["contexts"]["ema-rsi"]
    latest = ema_context["latest"]
    prev = ema_context["prev"]
    plan = None
    if str(underlying_result["Setup"]) not in {"EXTENDED", "WEAK_TREND"}:
        plan = build_bull_call_debit_spread(
            ticker=ticker,
            price=float(underlying_result["Price"]),
            atr=float(underlying_result["ATR"]),
            signal_date=str(underlying_result["SignalDate"]),
            score=float(underlying_result["Score"]),
            min_dte=effective_config.min_dte,
            max_dte=effective_config.max_dte,
            preferred_dte=effective_config.preferred_dte,
            max_debit=effective_config.max_debit,
            preferred_debit_min=effective_config.preferred_debit_min,
            preferred_debit_max=effective_config.preferred_debit_max,
        )

    return {
        "result": _build_result_from_underlying(
            ticker=ticker,
            underlying_result=underlying_result,
            plan=plan,
            mode=effective_config.mode,
            sources=sources,
            close_price=float(latest["Close"]),
            ema50=float(latest["EMA50"]),
            rsi=float(latest["RSI"]),
            prev_close=float(prev["Close"]),
        ),
        "sources": underlying_analysis["sources"],
    }


def scan_ticker(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    config: SwingOptionsDebitSpreadConfig | None = None,
) -> dict:
    return analyze_ticker(ticker=ticker, start=start, end=end, config=config)["result"]


def scan_watchlist(
    tickers: Iterable[str],
    start: str = "2018-01-01",
    end: str | None = None,
    config: SwingOptionsDebitSpreadConfig | None = None,
) -> list[dict]:
    results = []
    for ticker in tickers:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            continue
        try:
            results.append(scan_ticker(ticker=clean_ticker, start=start, end=end, config=config))
        except Exception as exc:
            results.append(
                {
                    "Ticker": clean_ticker,
                    "Strategy": "swing-options-debit-spread",
                    "Signal": "ERROR",
                    "Setup": "ERROR",
                    "Score": 0.0,
                    "Price": 0.0,
                    "OptionStructure": "N/A",
                    "LongStrike": 0.0,
                    "ShortStrike": 0.0,
                    "DTE": 0,
                    "EstDebit": 0.0,
                    "MaxLoss": 0.0,
                    "MaxProfit": 0.0,
                    "RewardRisk": 0.0,
                    "SpreadWidth": 0.0,
                    "PremiumStatus": "N/A",
                    "SmallAccountEligible": "NO",
                    "Reason": str(exc),
                    "SourceSummary": "ERROR",
                    "RSI": 0.0,
                    "ATR": 0.0,
                    "SignalDate": str(pd.Timestamp.today().date()),
                    "PlannedExecutionDate": str((pd.Timestamp.today() + BDay(1)).date()),
                    "UniverseStatus": "ERROR",
                    "UniverseReason": str(exc),
                    "SetupStatus": "ERROR",
                    "DistanceToSetup": "ERROR",
                    "Equity": 0.0,
                    "OptionsReason": str(exc),
                    "AvgDollarVolume": 0.0,
                    "EarningsDate": "UNKNOWN",
                    "OptionsAction": "ERROR",
                    "Structure": "N/A",
                    "Expiration": "N/A",
                    "EstimatedDebit": 0.0,
                    "MaxProfit": 0.0,
                    "TradeQuality": "ERROR",
                    "PlannedEntryReference": "",
                    "StopLoss": "",
                    "TakeProfit": "",
                    "RiskPerShare": "",
                    "RewardPerShare": "",
                }
            )
    return results


def _label_historical_signal_rows(
    merged_df: pd.DataFrame,
    ticker: str,
    config: SwingOptionsDebitSpreadConfig,
) -> pd.DataFrame:
    if merged_df.empty:
        return merged_df.copy()

    rows: list[dict] = []
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
        source_summary = "; ".join([f"{source.strategy}:{source.signal}/{source.setup}" for source in sources])
        supporting_signals = sum(1 for source in sources if source.bullish_support)

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
        score, is_extended, is_weak_trend, _ = _score_sources(sources=sources, ema_context=ema_context)
        raw_setup = classify_swing_setup(score=score, is_extended=is_extended, is_weak_trend=is_weak_trend)
        selected_source_strategy = _selected_source_strategy(sources)
        strict_conversion = _finalize_signal_conversion(
            ticker=ticker,
            signal_date=str(signal_date.date()),
            score=score,
            raw_setup=raw_setup,
            sources=sources,
            selected_source_strategy=selected_source_strategy,
            latest_close=float(row["Close"]),
            latest_atr=float(row["ATR_ema"]),
            supporting_signals=supporting_signals,
            config=SwingOptionsConfig(
                interval=config.interval,
                max_hold_days=config.max_hold_days,
                time_stop_days=config.time_stop_days,
            ),
        )

        plan = None
        if raw_setup not in {"EXTENDED", "WEAK_TREND"}:
            plan = build_bull_call_debit_spread(
                ticker=ticker,
                price=float(row["Close"]),
                atr=float(row["ATR_ema"]),
                signal_date=str(signal_date.date()),
                score=float(score),
                min_dte=config.min_dte,
                max_dte=config.max_dte,
                preferred_dte=config.preferred_dte,
                max_debit=config.max_debit,
                preferred_debit_min=config.preferred_debit_min,
                preferred_debit_max=config.preferred_debit_max,
            )

        labeled = _build_result_from_underlying(
            ticker=ticker,
            underlying_result={
                "Signal": str(strict_conversion["signal"]),
                "Setup": str(raw_setup),
                "Score": float(score),
                "Price": float(row["Close"]),
                "RSI": float(row["RSI_ema"]),
                "ATR": float(row["ATR_ema"]),
                "SignalDate": str(signal_date.date()),
                "PlannedExecutionDate": _planned_execution_date(str(signal_date.date())),
                "SourceSummary": source_summary,
                "Equity": 0.0,
                "AvgDollarVolume": 0.0,
                "EarningsDate": "PLACEHOLDER_OK",
            },
            plan=plan,
            mode=config.mode,
            sources=sources,
            close_price=float(row["Close"]),
            ema50=float(row["EMA50"]),
            rsi=float(row["RSI_ema"]),
            prev_close=float(row["PrevClose"]),
        )
        rows.append(labeled)

    return pd.DataFrame(rows)


def build_historical_debit_spread_signal_history(
    tickers: list[str],
    start: str,
    end: str | None = None,
    config: SwingOptionsDebitSpreadConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    effective_config = config or SwingOptionsDebitSpreadConfig()
    signal_frames: list[pd.DataFrame] = []
    daily_data_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        clean_ticker = str(ticker).strip().upper()
        if not clean_ticker:
            continue
        raw_df = load_yfinance_data(ticker=clean_ticker, start=start, end=end)
        ema_df = _ema_source_history(raw_df)
        four_hour_df = _four_hour_source_history(ticker=clean_ticker, interval=effective_config.interval)
        v2_df = _rsi_bollinger_v2_source_history(ticker=clean_ticker, raw_df=raw_df)
        merged_df = _merge_source_histories(ema_df=ema_df, four_hour_df=four_hour_df, v2_df=v2_df)
        labeled_df = _label_historical_signal_rows(
            merged_df=merged_df,
            ticker=clean_ticker,
            config=effective_config,
        )
        if not labeled_df.empty:
            signal_frames.append(labeled_df)
        daily_data_by_ticker[clean_ticker] = raw_df

    combined = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return combined, daily_data_by_ticker


def affordable_debit_spread_candidates(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df.copy()
    return signal_df[
        (signal_df["Signal"] == "BUY")
        & (signal_df["SmallAccountEligible"] == "YES")
        & (signal_df["PremiumStatus"].isin({"OK", "ACCEPTABLE"}))
    ].copy()


def select_best_daily_candidate(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df.copy()
    premium_rank = signal_df["PremiumStatus"].map({"OK": 0, "ACCEPTABLE": 1}).fillna(99)
    ordered = signal_df.assign(_PremiumRank=premium_rank).sort_values(
        by=["SignalDate", "_PremiumRank", "RewardRisk", "Score", "EstDebit", "Ticker"],
        ascending=[True, True, False, False, True, True],
        kind="mergesort",
    )
    return ordered.groupby("SignalDate", as_index=False).head(1).drop(columns=["_PremiumRank"]).reset_index(drop=True)


def _derive_exit_date(trade: dict, daily_df: pd.DataFrame) -> str:
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


def _proxy_spread_pnl(trade: dict, max_profit: float, debit_dollars: float) -> tuple[float, float]:
    max_profit_pct = (max_profit / debit_dollars) * 100 if debit_dollars > 0 else 0.0

    if bool(trade["Failed"]):
        pnl_pct = -50.0
    elif str(trade["MoveQuality"]) == "EXCELLENT":
        pnl_pct = min(150.0, max_profit_pct)
    elif str(trade["MoveQuality"]) == "STRONG":
        pnl_pct = min(100.0, max_profit_pct)
    elif str(trade["MoveQuality"]) == "SUITABLE":
        pnl_pct = min(50.0, max_profit_pct)
    else:
        raw_pct = (float(trade["MFE_R"]) * 35.0) + (float(trade["MAE_R"]) * 10.0)
        pnl_pct = max(min(raw_pct, max_profit_pct), -50.0)

    pnl_dollars = round(debit_dollars * (pnl_pct / 100.0), 2)
    return round(pnl_pct, 2), pnl_dollars


def replay_debit_spread_candidates(
    candidate_df: pd.DataFrame,
    daily_data_by_ticker: dict[str, pd.DataFrame],
    config: SwingOptionsDebitSpreadConfig | None = None,
) -> pd.DataFrame:
    effective_config = config or SwingOptionsDebitSpreadConfig()
    trades: list[dict] = []
    current_exit_date: str | None = None

    for row in candidate_df.sort_values(by=["SignalDate", "RewardRisk", "Score"], ascending=[True, False, False]).itertuples(index=False):
        signal_date = str(row.SignalDate)
        if current_exit_date is not None and pd.Timestamp(signal_date) <= pd.Timestamp(current_exit_date):
            continue

        trade = evaluate_proxy_trade(
            ticker=str(row.Ticker),
            signal_date=signal_date,
            atr=float(row.ATR),
            daily_df=daily_data_by_ticker[str(row.Ticker)],
            max_hold_days=effective_config.max_hold_days,
            time_stop_days=effective_config.time_stop_days,
        )
        if trade is None:
            continue

        exit_date = _derive_exit_date(trade=trade, daily_df=daily_data_by_ticker[str(row.Ticker)])
        current_exit_date = exit_date
        proxy_pnl_pct, proxy_pnl_dollars = _proxy_spread_pnl(
            trade=trade,
            max_profit=float(row.MaxProfit),
            debit_dollars=float(row.MaxLoss),
        )
        trades.append(
            {
                "Ticker": str(row.Ticker),
                "SignalDate": signal_date,
                "EntryDate": str(trade["EntryDate"]),
                "ExitDate": exit_date,
                "OptionStructure": str(row.OptionStructure),
                "LongStrike": float(row.LongStrike),
                "ShortStrike": float(row.ShortStrike),
                "DTE": int(row.DTE),
                "EstDebit": float(row.EstDebit),
                "MaxLoss": float(row.MaxLoss),
                "MaxProfit": float(row.MaxProfit),
                "RewardRisk": float(row.RewardRisk),
                "PremiumStatus": str(row.PremiumStatus),
                "SmallAccountEligible": str(row.SmallAccountEligible),
                "MoveQuality": str(trade["MoveQuality"]),
                "HoldDays": int(trade["HoldDays"]),
                "ExitReason": str(trade["ExitReason"]),
                "MFE_R": float(trade["MFE_R"]),
                "MAE_R": float(trade["MAE_R"]),
                "ProxyPnLPct": proxy_pnl_pct,
                "ProxyPnLDollars": proxy_pnl_dollars,
                "ProxyWinner": "YES" if proxy_pnl_dollars > 0 else "NO",
                "ValidationType": PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
            }
        )

    return pd.DataFrame(trades)


def build_debit_spread_monthly_report(signal_df: pd.DataFrame, affordable_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame(
            columns=[
                "ValidationType",
                "Month",
                "Signals",
                "AffordableCandidates",
                "TradesTriggered",
                "Winners",
                "Losers",
                "AverageProxyPnLPct",
                "TotalProxyPnLDollars",
                "EXCELLENT",
                "STRONG",
                "SUITABLE",
                "WEAK",
                "FAILED",
            ]
        )

    signals = signal_df.loc[signal_df["Signal"] == "BUY"].copy()
    if signals.empty:
        return pd.DataFrame(
            {
                "ValidationType": [PROXY_DEBIT_SPREAD_VALIDATION_LABEL],
                "Month": [pd.Timestamp.today().to_period("M").strftime("%Y-%m")],
                "Signals": [0],
                "AffordableCandidates": [0],
                "TradesTriggered": [0],
                "Winners": [0],
                "Losers": [0],
                "AverageProxyPnLPct": [0.0],
                "TotalProxyPnLDollars": [0.0],
                "EXCELLENT": [0],
                "STRONG": [0],
                "SUITABLE": [0],
                "WEAK": [0],
                "FAILED": [0],
            }
        )

    signals["Month"] = pd.to_datetime(signals["SignalDate"]).dt.to_period("M").astype(str)
    monthly_df = pd.DataFrame({"Month": sorted(signals["Month"].unique())})
    monthly_df.insert(0, "ValidationType", PROXY_DEBIT_SPREAD_VALIDATION_LABEL)
    monthly_df["Signals"] = monthly_df["Month"].map(signals.groupby("Month").size()).fillna(0).astype(int)

    if not affordable_df.empty:
        affordable_copy = affordable_df.copy()
        affordable_copy["Month"] = pd.to_datetime(affordable_copy["SignalDate"]).dt.to_period("M").astype(str)
        monthly_df["AffordableCandidates"] = monthly_df["Month"].map(affordable_copy.groupby("Month").size()).fillna(0).astype(int)
    else:
        monthly_df["AffordableCandidates"] = 0

    if not trades_df.empty:
        trades_copy = trades_df.copy()
        trades_copy["Month"] = pd.to_datetime(trades_copy["SignalDate"]).dt.to_period("M").astype(str)
        monthly_df["TradesTriggered"] = monthly_df["Month"].map(trades_copy.groupby("Month").size()).fillna(0).astype(int)
        monthly_df["Winners"] = monthly_df["Month"].map(trades_copy.groupby("Month")["ProxyWinner"].apply(lambda s: int((s == "YES").sum()))).fillna(0).astype(int)
        monthly_df["Losers"] = monthly_df["Month"].map(trades_copy.groupby("Month")["ProxyWinner"].apply(lambda s: int((s == "NO").sum()))).fillna(0).astype(int)
        monthly_df["AverageProxyPnLPct"] = monthly_df["Month"].map(trades_copy.groupby("Month")["ProxyPnLPct"].mean()).fillna(0.0).round(2)
        monthly_df["TotalProxyPnLDollars"] = monthly_df["Month"].map(trades_copy.groupby("Month")["ProxyPnLDollars"].sum()).fillna(0.0).round(2)
        for quality in ["EXCELLENT", "STRONG", "SUITABLE", "WEAK", "FAILED"]:
            monthly_df[quality] = monthly_df["Month"].map(
                trades_copy.groupby("Month")["MoveQuality"].apply(lambda s, q=quality: int((s == q).sum()))
            ).fillna(0).astype(int)
    else:
        monthly_df["TradesTriggered"] = 0
        monthly_df["Winners"] = 0
        monthly_df["Losers"] = 0
        monthly_df["AverageProxyPnLPct"] = 0.0
        monthly_df["TotalProxyPnLDollars"] = 0.0
        for quality in ["EXCELLENT", "STRONG", "SUITABLE", "WEAK", "FAILED"]:
            monthly_df[quality] = 0

    return monthly_df


def build_debit_spread_summary(signal_df: pd.DataFrame, affordable_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    total_months = max(1, pd.to_datetime(signal_df["SignalDate"]).dt.to_period("M").nunique()) if not signal_df.empty else 1
    total_signals = int((signal_df["Signal"] == "BUY").sum()) if not signal_df.empty else 0
    affordable_trades = len(affordable_df)
    trades_per_month = round(len(trades_df) / total_months, 2) if total_months > 0 else 0.0

    if trades_df.empty:
        row = {
            "ValidationType": PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
            "TotalSignals": total_signals,
            "AffordableTrades": affordable_trades,
            "TradesPerMonth": trades_per_month,
            "WinRateProxy": 0.0,
            "AverageProxyPnLPct": 0.0,
            "AverageProxyPnLDollars": 0.0,
            "ProfitFactorProxy": 0.0,
            "MaxDrawdownProxy": 0.0,
            "AverageHoldDays": 0.0,
            "PctSuitablePlus": 0.0,
            "PctFailed": 0.0,
            "BestTicker": "N/A",
            "WorstTicker": "N/A",
        }
        return pd.DataFrame([row])

    gross_profits = trades_df.loc[trades_df["ProxyPnLDollars"] > 0, "ProxyPnLDollars"].sum()
    gross_losses = trades_df.loc[trades_df["ProxyPnLDollars"] < 0, "ProxyPnLDollars"].sum()
    profit_factor = round(gross_profits / abs(gross_losses), 2) if gross_losses < 0 else 0.0
    cumulative = trades_df["ProxyPnLDollars"].cumsum()
    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
    ticker_scores = trades_df.groupby("Ticker")["ProxyPnLPct"].mean().sort_values(ascending=False)

    row = {
        "ValidationType": PROXY_DEBIT_SPREAD_VALIDATION_LABEL,
        "TotalSignals": total_signals,
        "AffordableTrades": affordable_trades,
        "TradesPerMonth": trades_per_month,
        "WinRateProxy": round((trades_df["ProxyWinner"] == "YES").mean() * 100, 2),
        "AverageProxyPnLPct": round(float(trades_df["ProxyPnLPct"].mean()), 2),
        "AverageProxyPnLDollars": round(float(trades_df["ProxyPnLDollars"].mean()), 2),
        "ProfitFactorProxy": profit_factor,
        "MaxDrawdownProxy": round(float(drawdown.min()), 2),
        "AverageHoldDays": round(float(trades_df["HoldDays"].mean()), 2),
        "PctSuitablePlus": round((trades_df["MoveQuality"].isin({"SUITABLE", "STRONG", "EXCELLENT"}).mean() * 100), 2),
        "PctFailed": round(((trades_df["MoveQuality"] == "FAILED").mean() * 100), 2),
        "BestTicker": str(ticker_scores.index[0]),
        "WorstTicker": str(ticker_scores.index[-1]),
    }
    return pd.DataFrame([row])


def build_tuning_summary(
    strict_payload: dict[str, object],
    tuned_payload: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict] = []
    for mode, payload in [("strict", strict_payload), ("tuned", tuned_payload)]:
        summary = payload["summary"].iloc[0]
        rows.append(
            {
                "Mode": mode,
                "TradesPerMonth": float(summary["TradesPerMonth"]),
                "WinRateProxy": float(summary["WinRateProxy"]),
                "AverageProxyPnLPct": float(summary["AverageProxyPnLPct"]),
                "ProfitFactorProxy": float(summary["ProfitFactorProxy"]),
                "MaxDrawdownProxy": float(summary["MaxDrawdownProxy"]),
                "PctFailed": float(summary["PctFailed"]),
                "PctSuitablePlus": float(summary["PctSuitablePlus"]),
            }
        )
    return pd.DataFrame(rows)


def save_backtest_outputs(summary_df: pd.DataFrame, trades_df: pd.DataFrame, monthly_df: pd.DataFrame, output_dir: str = "reports") -> dict[str, Path]:
    output_path = Path(output_dir) / "swing_options_debit_spread"
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "debit_spread_backtest_summary.csv"
    trades_path = output_path / "debit_spread_backtest_trades.csv"
    monthly_path = output_path / "debit_spread_backtest_monthly.csv"
    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    return {"summary": summary_path, "trades": trades_path, "monthly": monthly_path}


def save_tuning_summary(tuning_df: pd.DataFrame, output_dir: str = "reports") -> Path:
    output_path = Path(output_dir) / "swing_options_debit_spread"
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "debit_spread_tuning_summary.csv"
    tuning_df.to_csv(file_path, index=False)
    return file_path


def run_debit_spread_backtest(
    tickers: list[str],
    start: str,
    end: str | None = None,
    config: SwingOptionsDebitSpreadConfig | None = None,
    output_dir: str = "reports",
    save_outputs: bool = True,
) -> dict[str, object]:
    effective_config = config or SwingOptionsDebitSpreadConfig()
    signal_df, daily_data_by_ticker = build_historical_debit_spread_signal_history(
        tickers=tickers,
        start=start,
        end=end,
        config=effective_config,
    )
    affordable_df = affordable_debit_spread_candidates(signal_df)
    candidate_df = select_best_daily_candidate(affordable_df)
    trades_df = replay_debit_spread_candidates(candidate_df, daily_data_by_ticker, config=effective_config)
    monthly_df = build_debit_spread_monthly_report(signal_df, affordable_df, trades_df)
    summary_df = build_debit_spread_summary(signal_df, affordable_df, trades_df)
    signal_df["Mode"] = effective_config.mode
    affordable_df["Mode"] = effective_config.mode
    candidate_df["Mode"] = effective_config.mode
    if not trades_df.empty:
        trades_df["Mode"] = effective_config.mode
    monthly_df["Mode"] = effective_config.mode
    summary_df["Mode"] = effective_config.mode
    if save_outputs:
        paths = save_backtest_outputs(summary_df, trades_df, monthly_df, output_dir=output_dir)
    else:
        paths = {}
    return {
        "signals": signal_df,
        "affordable_candidates": affordable_df,
        "selected_candidates": candidate_df,
        "trades": trades_df,
        "monthly": monthly_df,
        "summary": summary_df,
        "paths": paths,
    }
