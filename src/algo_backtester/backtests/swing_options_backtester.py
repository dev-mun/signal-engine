from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd
from pandas.tseries.offsets import BDay

from algo_backtester.backtests.ema_rsi_backtester import EmaRsiBacktestConfig, EmaRsiPullbackBacktester
from algo_backtester.backtests.four_hour_trend_backtester import (
    FourHourTrendBacktester,
    FourHourTrendConfig,
    prepare_four_hour_data,
)
from algo_backtester.backtests.rsi_bollinger_v2_backtester import (
    RsiBollingerV2Backtester,
    RsiBollingerV2BacktestConfig,
    resolve_ticker_config,
)
from algo_backtester.data_loader import load_yfinance_data
from algo_backtester.strategies.ema_rsi_pullback import classify_setup as ema_rsi_classify_setup
from algo_backtester.strategies.four_hour_trend_pullback import classify_setup as four_hour_classify_setup
from algo_backtester.strategies.rsi_bollinger_v2 import classify_setup as rsi_bollinger_v2_classify_setup
from algo_backtester.strategies.swing_options import (
    PLANNER_DISCLAIMER,
    SwingOptionsPlan,
    SwingSourceSignal,
    atr_risk_quality_score,
    build_long_call_plan,
    bullish_trend,
    classify_swing_setup,
    ema_structure_score,
    extended_setup,
    extension_distance_score,
    recent_momentum_score,
    rsi_location_score,
    source_signal_strength,
    trend_alignment_score,
    volume_confirmation_score,
    weak_trend,
)

SWING_OPTIONS_ACCOUNT_TIERS = {
    "small_account": 10_000.0,
    "mid_account": 25_000.0,
    "large_account": 50_000.0,
}

SWING_OPTIONS_PREMIUM_BUDGET_MODES = {
    "conservative": 0.015,
    "standard": 0.025,
    "aggressive": 0.04,
}

SMALL_ACCOUNT_OPTIONS_PROFILE = "small_account_options"
SMALL_ACCOUNT_OPTIONS_RULES = {
    "account_size": 3_000.0,
    "max_contracts": 1,
    "max_premium": 150.0,
    "preferred_premium_min": 75.0,
    "preferred_premium_max": 125.0,
    "max_open_positions": 1,
}


@dataclass(frozen=True)
class SwingOptionsConfig:
    initial_cash: float = SWING_OPTIONS_ACCOUNT_TIERS["mid_account"]
    risk_per_trade: float = SWING_OPTIONS_PREMIUM_BUDGET_MODES["standard"]
    max_contracts: int = 1
    interval: str = "1h"
    min_dte: int = 30
    max_dte: int = 60
    preferred_dte: int = 45
    time_stop_days: int = 5
    max_hold_days: int = 15


def _planned_execution_date(signal_date: str) -> str:
    return str((pd.Timestamp(signal_date) + BDay(1)).date())


def _previous_row(df: pd.DataFrame):
    if len(df) >= 2:
        return df.iloc[-2]
    return df.iloc[-1]


def _ema_rsi_source(
    ticker: str,
    start: str,
    end: str | None,
) -> tuple[SwingSourceSignal, dict]:
    raw_df = load_yfinance_data(ticker=ticker, start=start, end=end)
    bt = EmaRsiPullbackBacktester(config=EmaRsiBacktestConfig())
    strategy_df, _, _, signals_df = bt.run(raw_df)
    latest = signals_df.iloc[-1]
    prev = _previous_row(signals_df)
    signal = str(latest["Signal"])
    setup = ema_rsi_classify_setup(signal, float(latest["RSI"]))
    trend_quality = bullish_trend(
        close_price=float(latest["Close"]),
        ema20=float(latest["EMA20"]),
        ema50=float(latest["EMA50"]),
        ema200=float(latest["EMA200"]),
    )
    volume_confirmed = float(latest["Volume"]) > float(latest["AverageVolume20"])
    recent_momentum = float(latest["Close"]) > float(prev["Close"]) and float(latest["Close"]) >= float(latest["EMA20"])
    bullish_support = signal == "BUY" or (setup == "NEAR_SETUP" and trend_quality and recent_momentum)

    source = SwingSourceSignal(
        strategy="ema-rsi",
        signal=signal,
        setup=setup,
        price=float(latest["Close"]),
        rsi=float(latest["RSI"]),
        atr=float(latest["ATR"]),
        trend_quality=trend_quality,
        volume_confirmed=volume_confirmed,
        recent_momentum=recent_momentum,
        bullish_support=bullish_support,
        notes=str(latest["Reason"]),
    )
    context = {
        "signal_date": str(signals_df.index[-1].date()),
        "latest": latest,
        "prev": prev,
        "strategy_df": strategy_df,
    }
    return source, context


def _four_hour_source(
    ticker: str,
    interval: str,
) -> tuple[SwingSourceSignal, dict]:
    raw_df = prepare_four_hour_data(ticker=ticker, interval=interval)
    if raw_df.empty:
        raise ValueError(f"No usable intraday data found for ticker: {ticker}")
    bt = FourHourTrendBacktester(config=FourHourTrendConfig(interval=interval))
    strategy_df, _, _, signals_df = bt.run(raw_df)
    latest = signals_df.iloc[-1]
    prev = _previous_row(signals_df)
    signal = str(latest["Signal"])
    setup = four_hour_classify_setup(signal, latest)
    trend_quality = bullish_trend(
        close_price=float(latest["Close"]),
        ema20=float(latest["EMA20"]),
        ema50=float(latest["EMA50"]),
        ema200=float(latest["EMA200"]),
    )
    volume_confirmed = float(latest["Volume"]) > float(latest["AverageVolume20"])
    recent_momentum = float(latest["Close"]) > float(prev["Close"]) and float(latest["Close"]) >= float(latest["EMA20"])
    bullish_support = signal == "BUY" or (setup in {"NEEDS_PULLBACK", "NEAR_SETUP"} and trend_quality and recent_momentum)

    source = SwingSourceSignal(
        strategy="four-hour-trend",
        signal=signal,
        setup=setup,
        price=float(latest["Close"]),
        rsi=float(latest["RSI"]),
        atr=float(latest["ATR"]),
        trend_quality=trend_quality,
        volume_confirmed=volume_confirmed,
        recent_momentum=recent_momentum,
        bullish_support=bullish_support,
        notes=str(latest["Reason"]),
    )
    context = {
        "signal_date": str(pd.Timestamp(signals_df.index[-1]).date()),
        "latest": latest,
        "prev": prev,
        "strategy_df": strategy_df,
    }
    return source, context


def _rsi_bollinger_v2_source(
    ticker: str,
    start: str,
    end: str | None,
) -> tuple[SwingSourceSignal, dict]:
    raw_df = load_yfinance_data(ticker=ticker, start=start, end=end)
    _, effective_config = resolve_ticker_config(ticker=ticker, config=RsiBollingerV2BacktestConfig())
    bt = RsiBollingerV2Backtester(config=effective_config)
    strategy_df, _, _, signals_df = bt.run(raw_df)
    latest = signals_df.iloc[-1]
    prev = _previous_row(signals_df)
    signal = str(latest["Signal"])
    setup = rsi_bollinger_v2_classify_setup(signal, latest, band_tolerance=effective_config.band_tolerance)
    trend_quality = float(latest["Close"]) > float(latest["EMA200"]) and float(latest["EMA50"]) > float(latest["EMA200"])
    volume_confirmed = float(latest["Volume"]) >= float(latest["AverageVolume20"]) * effective_config.volume_multiplier
    recent_momentum = float(latest["Close"]) > float(prev["Close"])
    bullish_support = signal == "BUY" or (setup in {"OVERSOLD", "NEAR_SETUP"} and trend_quality and recent_momentum)

    source = SwingSourceSignal(
        strategy="rsi-bollinger-v2",
        signal=signal,
        setup=setup,
        price=float(latest["Close"]),
        rsi=float(latest["RSI"]),
        atr=float(latest["ATR"]),
        trend_quality=trend_quality,
        volume_confirmed=volume_confirmed,
        recent_momentum=recent_momentum,
        bullish_support=bullish_support,
        notes=str(latest["Reason"]),
    )
    context = {
        "signal_date": str(signals_df.index[-1].date()),
        "latest": latest,
        "prev": prev,
        "strategy_df": strategy_df,
    }
    return source, context


def evaluate_source_signals(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    interval: str = "1h",
) -> dict:
    ema_source, ema_context = _ema_rsi_source(ticker=ticker, start=start, end=end)
    four_hour_source, four_hour_context = _four_hour_source(ticker=ticker, interval=interval)
    v2_source, v2_context = _rsi_bollinger_v2_source(ticker=ticker, start=start, end=end)

    return {
        "sources": [ema_source, four_hour_source, v2_source],
        "contexts": {
            "ema-rsi": ema_context,
            "four-hour-trend": four_hour_context,
            "rsi-bollinger-v2": v2_context,
        },
    }


def _source_summary(sources: list[SwingSourceSignal]) -> str:
    return "; ".join([f"{source.strategy}:{source.signal}/{source.setup}" for source in sources])


def _score_sources(sources: list[SwingSourceSignal], ema_context: dict) -> tuple[float, bool, bool, int]:
    latest = ema_context["latest"]
    prev = ema_context["prev"]
    close_price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema200 = float(latest["EMA200"])
    rsi = float(latest["RSI"])
    atr = float(latest["ATR"])
    atr_pct = atr / close_price if close_price > 0 else 0.0
    price_to_ema20 = close_price / max(ema20, 1e-9)
    daily_bullish = bullish_trend(close_price=close_price, ema20=ema20, ema50=ema50, ema200=ema200)
    intraday_bullish = next(source.trend_quality for source in sources if source.strategy == "four-hour-trend")
    v2_bullish = next(source.trend_quality for source in sources if source.strategy == "rsi-bollinger-v2")
    daily_volume_confirmed = float(latest["Volume"]) > float(latest["AverageVolume20"])
    supporting_signals = sum(1 for source in sources if source.bullish_support)
    recent_momentum = float(latest["Close"]) > float(prev["Close"])

    score = 0.0
    score += source_signal_strength(sources)
    score += trend_alignment_score(
        daily_bullish=daily_bullish,
        intraday_bullish=intraday_bullish,
        v2_bullish=v2_bullish,
    )
    score += rsi_location_score(rsi=rsi)
    score += ema_structure_score(close_price=close_price, ema20=ema20, ema50=ema50, ema200=ema200)
    score += volume_confirmation_score(volume_confirmed=daily_volume_confirmed, supporting_signals=supporting_signals)
    score += extension_distance_score(rsi=rsi, price_to_ema20=price_to_ema20)
    score += atr_risk_quality_score(atr_pct=atr_pct)
    score += recent_momentum_score(recent_momentum=recent_momentum, close_price=close_price, ema20=ema20)

    is_extended = extended_setup(rsi=rsi, price_to_ema20=price_to_ema20)
    is_weak_trend = weak_trend(close_price=close_price, ema50=ema50, ema200=ema200)
    return min(round(score, 2), 100.0), is_extended, is_weak_trend, supporting_signals


def _premium_risk_budget(config: SwingOptionsConfig) -> float:
    bounded_risk_pct = min(
        max(config.risk_per_trade, SWING_OPTIONS_PREMIUM_BUDGET_MODES["conservative"]),
        SWING_OPTIONS_PREMIUM_BUDGET_MODES["aggressive"],
    )
    return round(config.initial_cash * bounded_risk_pct, 2)


def _small_account_premium_status(contract_premium: float) -> str:
    if contract_premium <= 0:
        return "N/A"
    if contract_premium <= SMALL_ACCOUNT_OPTIONS_RULES["preferred_premium_max"]:
        return "OK"
    if contract_premium <= SMALL_ACCOUNT_OPTIONS_RULES["max_premium"]:
        return "ACCEPTABLE"
    return "TOO_EXPENSIVE"


def _small_account_reason(signal: str, contract_premium: float) -> str:
    if signal != "BUY":
        return "No option plan generated because signal is HOLD."
    if contract_premium <= 0:
        return "No option plan generated because signal is HOLD."
    if contract_premium < SMALL_ACCOUNT_OPTIONS_RULES["preferred_premium_min"]:
        return "Below the preferred $75-$125 premium range but within the $150 cap."
    if contract_premium <= SMALL_ACCOUNT_OPTIONS_RULES["preferred_premium_max"]:
        return "Within the preferred $75-$125 premium range."
    if contract_premium <= SMALL_ACCOUNT_OPTIONS_RULES["max_premium"]:
        return "Within the $150 cap but above the preferred premium range."
    return "Estimated contract premium exceeds the $150 small-account cap."


def apply_execution_profile_labels(result: dict, account_profile: str = "standard") -> dict:
    annotated = dict(result)
    contract_premium = float(annotated.get("MaxLoss", 0.0) or 0.0)
    premium_status = _small_account_premium_status(contract_premium=contract_premium)
    eligible = (
        str(annotated.get("Signal", "")) == "BUY"
        and contract_premium > 0
        and contract_premium <= SMALL_ACCOUNT_OPTIONS_RULES["max_premium"]
    )
    account_reason = _small_account_reason(
        signal=str(annotated.get("Signal", "")),
        contract_premium=contract_premium,
    )
    base_reason = str(annotated.get("Reason", "")).strip()

    annotated["AccountProfile"] = account_profile
    annotated["SmallAccountEligible"] = "YES" if eligible else "NO"
    annotated["PremiumStatus"] = premium_status
    annotated["SmallAccountReason"] = account_reason
    annotated["Reason"] = f"{base_reason} | Small account: {account_reason}" if base_reason else account_reason
    return annotated


def _selected_source_strategy(sources: list[SwingSourceSignal]) -> str:
    bullish_sources = [source for source in sources if source.bullish_support]
    if bullish_sources:
        return sorted(bullish_sources, key=lambda source: (source.signal == "BUY", source.recent_momentum, source.trend_quality), reverse=True)[0].strategy
    return sources[0].strategy


def _block_reason(
    raw_setup: str,
    has_strong_source: bool,
    plan: SwingOptionsPlan | None,
    premium_over_budget: bool,
    final_setup: str,
    signal: str,
) -> str:
    if signal == "BUY":
        return "CONVERTED_TO_BUY"
    if raw_setup != "ACTIONABLE":
        return "NOT_PRE_FINAL_ACTIONABLE"
    if not has_strong_source:
        return "NO_STRONG_SOURCE"
    if plan is None:
        return "PLAN_NOT_CREATED"
    if premium_over_budget:
        return "PREMIUM_OVER_BUDGET"
    if final_setup != "ACTIONABLE":
        return f"FINAL_SETUP_{final_setup}"
    return "ACTIONABLE_NOT_CONVERTED"


def _finalize_signal_conversion(
    *,
    ticker: str,
    signal_date: str,
    score: float,
    raw_setup: str,
    sources: list[SwingSourceSignal],
    selected_source_strategy: str,
    latest_close: float,
    latest_atr: float,
    supporting_signals: int,
    config: SwingOptionsConfig,
) -> dict:
    has_strong_source = supporting_signals > 0
    earnings_risk = False
    premium_budget = _premium_risk_budget(config)

    plan: SwingOptionsPlan | None = None
    if raw_setup in {"ACTIONABLE", "WATCHLIST"} and has_strong_source and not earnings_risk:
        plan = build_long_call_plan(
            ticker=ticker,
            price=latest_close,
            atr=latest_atr,
            signal_date=signal_date,
            score=score,
            premium_budget=premium_budget,
            min_dte=config.min_dte,
            max_dte=config.max_dte,
            preferred_target_dte=config.preferred_dte,
            time_stop_days=config.time_stop_days,
            max_hold_days=config.max_hold_days,
        )

    premium_over_budget = plan is not None and plan.max_loss > premium_budget
    final_setup = "WATCHLIST" if raw_setup == "ACTIONABLE" and premium_over_budget else raw_setup
    watchlist_buy_candidate = (
        final_setup == "WATCHLIST"
        and score >= 72.0
        and has_strong_source
        and plan is not None
        and not premium_over_budget
    )
    signal = "BUY" if ((final_setup == "ACTIONABLE") or watchlist_buy_candidate) and has_strong_source and plan is not None else "HOLD"
    block_reason = _block_reason(
        raw_setup=raw_setup,
        has_strong_source=has_strong_source,
        plan=plan,
        premium_over_budget=premium_over_budget,
        final_setup=final_setup,
        signal=signal,
    )
    audit = {
        "Ticker": ticker,
        "SignalDate": signal_date,
        "RawScore": score,
        "PreFinalSetup": raw_setup,
        "HasStrongSource": has_strong_source,
        "SupportingSignals": supporting_signals,
        "SelectedSourceStrategy": selected_source_strategy,
        "PlanCreated": plan is not None,
        "PremiumBudget": premium_budget,
        "EstimatedPremium": 0.0 if plan is None else plan.estimated_premium,
        "PremiumOverBudget": premium_over_budget,
        "FinalSetup": final_setup,
        "FinalSignal": signal,
        "BlockReason": block_reason,
    }
    return {
        "plan": plan,
        "premium_budget": premium_budget,
        "premium_over_budget": premium_over_budget,
        "final_setup": final_setup,
        "signal": signal,
        "block_reason": block_reason,
        "has_strong_source": has_strong_source,
        "audit": audit,
    }


def _journal_fields(result: dict, plan: SwingOptionsPlan | None) -> dict:
    if plan is None:
        return {
            "OptionsAction": "NO_OPTIONS_TRADE",
            "Structure": "No trade",
            "Expiration": "N/A",
            "LongStrike": 0.0,
            "ShortStrike": 0.0,
            "EstimatedDebit": 0.0,
            "MaxLoss": 0.0,
            "MaxProfit": "",
            "TradeQuality": "NO_TRADE",
            "PlannedEntryReference": "",
            "StopLoss": "",
            "TakeProfit": "",
            "RiskPerShare": "",
            "RewardPerShare": "",
        }

    return {
        "OptionsAction": "PLAN_SWING_LONG_CALL" if result["Signal"] == "BUY" else "WATCHLIST_SWING_LONG_CALL",
        "Structure": f"Long Call {plan.strike:.2f}C",
        "Expiration": plan.expiration,
        "LongStrike": plan.strike,
        "ShortStrike": 0.0,
        "EstimatedDebit": plan.estimated_premium,
        "MaxLoss": plan.max_loss,
        "MaxProfit": "",
        "TradeQuality": result["Setup"],
        "PlannedEntryReference": result["Price"],
        "StopLoss": plan.premium_stop,
        "TakeProfit": plan.exit3,
        "RiskPerShare": round(plan.estimated_premium - plan.premium_stop, 2),
        "RewardPerShare": round(plan.exit3 - plan.estimated_premium, 2),
    }


def analyze_ticker(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    config: SwingOptionsConfig | None = None,
) -> dict:
    effective_config = config or SwingOptionsConfig()
    evaluation = evaluate_source_signals(
        ticker=ticker,
        start=start,
        end=end,
        interval=effective_config.interval,
    )
    sources: list[SwingSourceSignal] = evaluation["sources"]
    contexts = evaluation["contexts"]
    ema_context = contexts["ema-rsi"]
    latest = ema_context["latest"]
    signal_date = ema_context["signal_date"]
    score, is_extended, is_weak_trend, supporting_signals = _score_sources(sources=sources, ema_context=ema_context)
    selected_source_strategy = _selected_source_strategy(sources)
    setup = classify_swing_setup(score=score, is_extended=is_extended, is_weak_trend=is_weak_trend)
    conversion = _finalize_signal_conversion(
        ticker=ticker,
        signal_date=signal_date,
        score=score,
        raw_setup=setup,
        sources=sources,
        selected_source_strategy=selected_source_strategy,
        latest_close=float(latest["Close"]),
        latest_atr=float(latest["ATR"]),
        supporting_signals=supporting_signals,
        config=effective_config,
    )
    plan: SwingOptionsPlan | None = conversion["plan"]
    premium_budget = float(conversion["premium_budget"])
    premium_over_budget = bool(conversion["premium_over_budget"])
    final_setup = str(conversion["final_setup"])
    signal = str(conversion["signal"])
    has_strong_source = bool(conversion["has_strong_source"])
    source_summary = _source_summary(sources)

    notes_parts = [source_summary]
    if not has_strong_source:
        notes_parts.append("No strong bullish source setup confirmed.")
    if premium_over_budget:
        notes_parts.append(f"Estimated premium risk {plan.max_loss:.2f} exceeds budget {premium_budget:.2f}.")
    notes_parts.append(PLANNER_DISCLAIMER)
    notes = " | ".join(notes_parts)

    result = {
        "Ticker": ticker,
        "Strategy": "swing-options",
        "Signal": signal,
        "Setup": final_setup,
        "Score": score,
        "SourceSummary": source_summary,
        "Price": float(latest["Close"]),
        "RSI": float(latest["RSI"]),
        "ATR": float(latest["ATR"]),
        "OptionType": "CALL",
        "Strike": 0.0 if plan is None else plan.strike,
        "DTE": 0 if plan is None else plan.dte,
        "DeltaTarget": 0.0 if plan is None else plan.delta_target,
        "EstPremium": 0.0 if plan is None else plan.estimated_premium,
        "MaxLoss": 0.0 if plan is None else plan.max_loss,
        "Exit1": 0.0 if plan is None else plan.exit1,
        "Exit2": 0.0 if plan is None else plan.exit2,
        "Exit3": 0.0 if plan is None else plan.exit3,
        "TimeStopDays": effective_config.time_stop_days,
        "MaxHoldDays": effective_config.max_hold_days,
        "Notes": notes,
        "SignalDate": signal_date,
        "PlannedExecutionDate": _planned_execution_date(signal_date),
        "UniverseStatus": "ELIGIBLE" if signal == "BUY" else "WATCH",
        "UniverseReason": notes,
        "SetupStatus": final_setup,
        "DistanceToSetup": final_setup,
        "Equity": float(effective_config.initial_cash),
        "Reason": (
            f"score={score:.2f}; setup={final_setup}; source={selected_source_strategy}; "
            "estimated premium is approximate."
        ),
        "OptionsReason": (
            "Exit ladder: +30%, +50%, +80%. "
            f"Invariant: {plan.invalidation_rule if plan else 'No trade.'} "
            f"{PLANNER_DISCLAIMER}"
        ),
        "AvgDollarVolume": 0.0,
        "EarningsDate": "PLACEHOLDER_OK",
    }
    result.update(_journal_fields(result=result, plan=plan))
    return {
        "result": result,
        "sources": [asdict(source) for source in sources],
    }


def scan_ticker(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    config: SwingOptionsConfig | None = None,
) -> dict:
    return analyze_ticker(ticker=ticker, start=start, end=end, config=config)["result"]


def scan_watchlist(
    tickers: Iterable[str],
    start: str = "2018-01-01",
    end: str | None = None,
    config: SwingOptionsConfig | None = None,
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
                    "Strategy": "swing-options",
                    "Signal": "ERROR",
                    "Setup": "ERROR",
                    "Score": 0.0,
                    "SourceSummary": "ERROR",
                    "Price": 0.0,
                    "RSI": 0.0,
                    "ATR": 0.0,
                    "OptionType": "CALL",
                    "Strike": 0.0,
                    "DTE": 0,
                    "DeltaTarget": 0.0,
                    "EstPremium": 0.0,
                    "MaxLoss": 0.0,
                    "Exit1": 0.0,
                    "Exit2": 0.0,
                    "Exit3": 0.0,
                    "TimeStopDays": 5,
                    "MaxHoldDays": 15,
                    "Notes": str(exc),
                    "SignalDate": str(pd.Timestamp.today().date()),
                    "PlannedExecutionDate": str((pd.Timestamp.today() + BDay(1)).date()),
                    "UniverseStatus": "ERROR",
                    "UniverseReason": str(exc),
                    "SetupStatus": "ERROR",
                    "DistanceToSetup": "ERROR",
                    "Equity": 0.0,
                    "Reason": str(exc),
                    "OptionsReason": str(exc),
                    "AvgDollarVolume": 0.0,
                    "EarningsDate": "UNKNOWN",
                    "OptionsAction": "ERROR",
                    "Structure": "N/A",
                    "Expiration": "N/A",
                    "LongStrike": 0.0,
                    "ShortStrike": 0.0,
                    "EstimatedDebit": 0.0,
                    "MaxProfit": "",
                    "TradeQuality": "ERROR",
                    "PlannedEntryReference": "",
                    "StopLoss": "",
                    "TakeProfit": "",
                    "RiskPerShare": "",
                    "RewardPerShare": "",
                }
            )

    return results
