from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

LIQUIDITY_SCORES = {
    "SPY": "A+",
    "QQQ": "A+",
    "NVDA": "A+",
    "AAPL": "A+",
    "MSFT": "A+",
    "META": "A",
    "AVGO": "A",
}


@dataclass(frozen=True)
class LongCallPlan:
    direction: str
    option_type: str
    strike: float
    dte: int
    delta_target: float
    estimated_premium: float
    max_loss: float
    exit1: float
    exit2: float
    exit3: float
    stop_loss: float
    liquidity: str
    notes: str
    expiration: str


def liquidity_score_for_ticker(ticker: str) -> str:
    return LIQUIDITY_SCORES.get(ticker.upper(), "B")


def _strike_increment(price: float) -> float:
    if price < 50:
        return 1.0
    if price < 200:
        return 2.5
    if price < 500:
        return 5.0
    return 10.0


def round_to_strike(price: float) -> float:
    increment = _strike_increment(price)
    return round(round(price / increment) * increment, 2)


def target_dte(source_strategy: str, atr_pct: float) -> int:
    if source_strategy == "four-hour-trend":
        if atr_pct >= 0.05:
            return 14
        return 21

    if atr_pct <= 0.025:
        return 30
    return 21


def target_delta(source_strategy: str, atr_pct: float) -> float:
    if source_strategy == "four-hour-trend":
        return 0.70 if atr_pct >= 0.05 else 0.65
    return 0.65 if atr_pct >= 0.04 else 0.60


def suggested_call_strike(price: float, atr: float, delta_target: float) -> float:
    itm_buffer = atr * max(delta_target - 0.50, 0.0)
    return round_to_strike(max(price - itm_buffer, 1.0))


def estimated_call_premium(price: float, strike: float, dte: int, atr_pct: float, delta_target: float) -> float:
    intrinsic_value = max(price - strike, 0.0)
    time_value = price * max(0.012, atr_pct * 0.55) * math.sqrt(max(dte, 1) / 30.0)
    delta_loading = 0.85 + ((delta_target - 0.60) * 0.9)
    premium = (intrinsic_value * 0.92) + (time_value * delta_loading)
    return round(max(premium, 0.5), 2)


def estimated_expiration(signal_date: str, dte: int) -> str:
    base_date = pd.Timestamp(signal_date).date()
    return str((pd.Timestamp(base_date) + pd.Timedelta(days=dte)).date())


def build_long_call_plan(
    ticker: str,
    source_strategy: str,
    price: float,
    atr: float,
    signal_date: str,
) -> LongCallPlan:
    atr_pct = atr / price if price > 0 else 0.0
    dte = target_dte(source_strategy=source_strategy, atr_pct=atr_pct)
    delta = target_delta(source_strategy=source_strategy, atr_pct=atr_pct)
    strike = suggested_call_strike(price=price, atr=atr, delta_target=delta)
    premium = estimated_call_premium(
        price=price,
        strike=strike,
        dte=dte,
        atr_pct=atr_pct,
        delta_target=delta,
    )
    liquidity = liquidity_score_for_ticker(ticker)
    expiration = estimated_expiration(signal_date=signal_date, dte=dte)

    return LongCallPlan(
        direction="BULLISH",
        option_type="CALL",
        strike=strike,
        dte=dte,
        delta_target=delta,
        estimated_premium=premium,
        max_loss=round(premium * 100, 2),
        exit1=round(premium * 1.25, 2),
        exit2=round(premium * 1.40, 2),
        exit3=round(premium * 1.60, 2),
        stop_loss=round(premium * 0.80, 2),
        liquidity=liquidity,
        notes=(
            "Approximation only. Target 1 contract, ATM to slightly ITM call, "
            "14-30 DTE, 0.60-0.70 delta."
        ),
        expiration=expiration,
    )
