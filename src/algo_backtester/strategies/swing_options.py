from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from algo_backtester.options_utils import liquidity_score_for_ticker, round_to_strike

PLANNER_DISCLAIMER = (
    "Approximation only: no live option chain, no real Greeks, no broker integration. "
    "Planner/scanner only, not a historical options PnL backtest."
)


@dataclass(frozen=True)
class SwingSourceSignal:
    strategy: str
    signal: str
    setup: str
    price: float
    rsi: float
    atr: float
    trend_quality: bool
    volume_confirmed: bool
    recent_momentum: bool
    bullish_support: bool
    notes: str


@dataclass(frozen=True)
class SwingOptionsPlan:
    option_type: str
    strike: float
    dte: int
    delta_target: float
    estimated_premium: float
    max_loss: float
    exit1: float
    exit2: float
    exit3: float
    premium_stop: float
    time_stop_days: int
    max_hold_days: int
    invalidation_rule: str
    expiration: str
    liquidity: str
    notes: str


def bullish_trend(close_price: float, ema20: float, ema50: float, ema200: float) -> bool:
    return close_price > ema200 and ema50 > ema200 and ema20 >= ema50


def weak_trend(close_price: float, ema50: float, ema200: float) -> bool:
    return close_price < ema200 or ema50 <= ema200


def extended_setup(rsi: float, price_to_ema20: float) -> bool:
    return rsi >= 70 or price_to_ema20 >= 1.08


def source_signal_strength(source_signals: Iterable[SwingSourceSignal]) -> float:
    points = 0.0

    for signal in source_signals:
        if signal.strategy == "ema-rsi":
            if signal.signal == "BUY":
                points += 18.0
            elif signal.setup == "NEAR_SETUP" and signal.recent_momentum and signal.trend_quality:
                points += 12.0
        elif signal.strategy == "four-hour-trend":
            if signal.signal == "BUY":
                points += 20.0
            elif signal.setup == "NEEDS_PULLBACK" and signal.trend_quality and signal.recent_momentum:
                points += 14.0
            elif signal.setup == "NEAR_SETUP" and signal.trend_quality:
                points += 10.0
        elif signal.strategy == "rsi-bollinger-v2":
            if signal.signal == "BUY":
                points += 17.0
            elif signal.setup in {"OVERSOLD", "NEAR_SETUP"} and signal.trend_quality and signal.recent_momentum:
                points += 12.0
            elif signal.setup == "WAIT" and signal.trend_quality and signal.recent_momentum:
                points += 6.0

    return min(points, 35.0)


def trend_alignment_score(daily_bullish: bool, intraday_bullish: bool, v2_bullish: bool) -> float:
    bullish_count = sum(1 for value in [daily_bullish, intraday_bullish, v2_bullish] if value)
    if bullish_count == 3:
        return 15.0
    if bullish_count == 2:
        return 11.0
    if bullish_count == 1:
        return 6.0
    return 0.0


def rsi_location_score(rsi: float) -> float:
    if 48 <= rsi <= 60:
        return 10.0
    if 42 <= rsi <= 65:
        return 7.0
    if 35 <= rsi < 42:
        return 4.0
    return 0.0


def ema_structure_score(close_price: float, ema20: float, ema50: float, ema200: float) -> float:
    if close_price > ema20 > ema50 > ema200:
        return 10.0
    if close_price > ema50 > ema200:
        return 7.0
    return 0.0


def volume_confirmation_score(volume_confirmed: bool, supporting_signals: int) -> float:
    if volume_confirmed and supporting_signals >= 2:
        return 8.0
    if volume_confirmed:
        return 5.0
    return 0.0


def extension_distance_score(rsi: float, price_to_ema20: float) -> float:
    if rsi <= 62 and price_to_ema20 <= 1.03:
        return 8.0
    if rsi <= 65 and price_to_ema20 <= 1.05:
        return 5.0
    return 0.0


def atr_risk_quality_score(atr_pct: float) -> float:
    if 0.02 <= atr_pct <= 0.06:
        return 7.0
    if 0.01 <= atr_pct <= 0.08:
        return 4.0
    return 0.0


def recent_momentum_score(recent_momentum: bool, close_price: float, ema20: float) -> float:
    if recent_momentum and close_price >= ema20:
        return 7.0
    if close_price >= ema20:
        return 4.0
    return 0.0


def classify_swing_setup(score: float, is_extended: bool, is_weak_trend: bool) -> str:
    if is_extended:
        return "EXTENDED"
    if is_weak_trend:
        return "WEAK_TREND"
    if score >= 80:
        return "ACTIONABLE"
    if score >= 65:
        return "WATCHLIST"
    if score >= 45:
        return "WAIT"
    return "AVOID"


def preferred_dte(atr_pct: float) -> int:
    if atr_pct >= 0.06:
        return 30
    if atr_pct <= 0.02:
        return 60
    return 45


def preferred_delta(score: float, atr_pct: float) -> float:
    if score >= 90:
        return 0.70
    if atr_pct >= 0.05:
        return 0.65
    return 0.60


def estimate_call_premium(price: float, strike: float, dte: int, delta_target: float, atr_pct: float) -> float:
    intrinsic_value = max(price - strike, 0.0)
    time_value = price * max(0.018, atr_pct * 0.75) * (dte / 45.0)
    delta_loading = 0.90 + ((delta_target - 0.55) * 0.9)
    premium = (intrinsic_value * 0.95) + (time_value * delta_loading)
    return round(max(premium, 0.75), 2)


def _strike_increment(price: float) -> float:
    if price < 50:
        return 1.0
    if price < 200:
        return 2.5
    if price < 500:
        return 5.0
    return 10.0


def _build_plan(
    ticker: str,
    signal_date: str,
    strike: float,
    dte: int,
    delta_target: float,
    premium: float,
    time_stop_days: int,
    max_hold_days: int,
    notes: str,
) -> SwingOptionsPlan:
    expiration = str((pd.Timestamp(signal_date) + pd.Timedelta(days=dte)).date())

    return SwingOptionsPlan(
        option_type="CALL",
        strike=strike,
        dte=dte,
        delta_target=delta_target,
        estimated_premium=premium,
        max_loss=round(premium * 100, 2),
        exit1=round(premium * 1.30, 2),
        exit2=round(premium * 1.50, 2),
        exit3=round(premium * 1.80, 2),
        premium_stop=round(premium * 0.75, 2),
        time_stop_days=time_stop_days,
        max_hold_days=max_hold_days,
        invalidation_rule="Exit if the source setup breaks or earnings placeholder becomes active.",
        expiration=expiration,
        liquidity=liquidity_score_for_ticker(ticker),
        notes=notes,
    )


def build_long_call_plan(
    ticker: str,
    price: float,
    atr: float,
    signal_date: str,
    score: float,
    premium_budget: float | None = None,
    min_dte: int = 30,
    max_dte: int = 60,
    preferred_target_dte: int | None = None,
    time_stop_days: int = 5,
    max_hold_days: int = 15,
) -> SwingOptionsPlan:
    atr_pct = atr / price if price > 0 else 0.0
    base_dte = preferred_target_dte if preferred_target_dte is not None else preferred_dte(atr_pct=atr_pct)
    base_dte = max(min_dte, min(max_dte, base_dte))
    base_delta = preferred_delta(score=score, atr_pct=atr_pct)
    lower_delta_candidates = [candidate for candidate in [0.55, 0.50] if candidate < base_delta]
    shorter_dte = max(min_dte, min(base_dte, 30))

    candidate_specs = [(base_delta, base_dte, 0, "Preferred contract approximation.")]
    for candidate_delta in lower_delta_candidates:
        candidate_specs.append((candidate_delta, base_dte, 0, f"Affordable fallback: lower delta to {candidate_delta:.2f}."))
    if shorter_dte != base_dte:
        working_delta = lower_delta_candidates[-1] if lower_delta_candidates else base_delta
        candidate_specs.append((working_delta, shorter_dte, 0, f"Affordable fallback: shorter DTE to {shorter_dte}."))
        candidate_specs.append((working_delta, shorter_dte, 1, "Affordable fallback: one strike farther OTM."))
    else:
        working_delta = lower_delta_candidates[-1] if lower_delta_candidates else base_delta
        candidate_specs.append((working_delta, base_dte, 1, "Affordable fallback: one strike farther OTM."))

    unique_specs: list[tuple[float, int, int, str]] = []
    seen: set[tuple[float, int, int]] = set()
    for delta_target, dte, otm_steps, note in candidate_specs:
        key = (round(delta_target, 2), dte, otm_steps)
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append((delta_target, dte, otm_steps, note))

    candidate_plans: list[SwingOptionsPlan] = []
    for delta_target, dte, otm_steps, note in unique_specs:
        strike = round_to_strike(max(price - (atr * max(delta_target - 0.55, 0.0) * 0.5), 1.0))
        if otm_steps > 0:
            strike = round_to_strike(strike + (_strike_increment(price) * otm_steps))
        premium = estimate_call_premium(
            price=price,
            strike=strike,
            dte=dte,
            delta_target=delta_target,
            atr_pct=atr_pct,
        )
        plan = _build_plan(
            ticker=ticker,
            signal_date=signal_date,
            strike=strike,
            dte=dte,
            delta_target=delta_target,
            premium=premium,
            time_stop_days=time_stop_days,
            max_hold_days=max_hold_days,
            notes=f"{PLANNER_DISCLAIMER} {note}".strip(),
        )
        candidate_plans.append(plan)
        if premium_budget is None or plan.max_loss <= premium_budget:
            return plan

    return min(candidate_plans, key=lambda plan: plan.max_loss)
