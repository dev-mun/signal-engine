from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algo_backtester.options_utils import liquidity_score_for_ticker, round_to_strike
from algo_backtester.strategies.swing_options import estimate_call_premium, preferred_delta

PROXY_DEBIT_SPREAD_VALIDATION_LABEL = "PROXY DEBIT SPREAD VALIDATION ONLY"
DEBIT_SPREAD_PLANNER_DISCLAIMER = (
    "Approximation only: no live option chain, no real Greeks, no broker integration. "
    "Proxy debit spread validation only, not real options PnL."
)


@dataclass(frozen=True)
class DebitSpreadPlan:
    option_structure: str
    long_strike: float
    short_strike: float
    dte: int
    est_debit: float
    max_loss: float
    max_profit: float
    reward_risk: float
    spread_width: float
    premium_status: str
    small_account_eligible: bool
    long_delta_target: float
    expiration: str
    liquidity: str
    notes: str


def _strike_increment(price: float) -> float:
    if price < 50:
        return 1.0
    if price < 200:
        return 2.5
    if price < 500:
        return 5.0
    return 10.0


def _estimate_short_call_credit(price: float, short_strike: float, dte: int, atr_pct: float, long_delta_target: float) -> float:
    short_delta_target = max(0.20, min(0.45, long_delta_target - 0.20))
    base_credit = estimate_call_premium(
        price=price,
        strike=short_strike,
        dte=dte,
        delta_target=short_delta_target,
        atr_pct=atr_pct,
    )
    return round(base_credit * 0.90, 2)


def classify_premium_status(max_loss: float, reward_risk: float, max_debit: float = 150.0) -> str:
    if max_loss <= 0:
        return "N/A"
    if max_loss > max_debit:
        return "TOO_EXPENSIVE"
    if reward_risk < 1.5:
        return "BAD_REWARD_RISK"
    if max_loss <= 125.0 and reward_risk >= 2.0:
        return "OK"
    return "ACCEPTABLE"


def _candidate_plan(
    ticker: str,
    price: float,
    atr: float,
    signal_date: str,
    long_delta_target: float,
    dte: int,
    short_target_r: float,
    max_debit: float,
    notes: str,
) -> DebitSpreadPlan:
    atr_pct = atr / price if price > 0 else 0.0
    increment = _strike_increment(price)
    long_strike = round_to_strike(max(price - (atr * max(long_delta_target - 0.50, 0.0) * 0.35), increment))
    short_reference = max(price + (atr * short_target_r), long_strike + increment)
    short_strike = round_to_strike(short_reference)
    if short_strike <= long_strike:
        short_strike = round_to_strike(long_strike + increment)

    long_premium = estimate_call_premium(
        price=price,
        strike=long_strike,
        dte=dte,
        delta_target=long_delta_target,
        atr_pct=atr_pct,
    )
    short_credit = _estimate_short_call_credit(
        price=price,
        short_strike=short_strike,
        dte=dte,
        atr_pct=atr_pct,
        long_delta_target=long_delta_target,
    )
    est_debit = round(max(long_premium - short_credit, 0.25), 2)
    spread_width = round(short_strike - long_strike, 2)
    max_loss = round(est_debit * 100, 2)
    max_profit = round(max((spread_width * 100) - max_loss, 0.0), 2)
    reward_risk = round(max_profit / max_loss, 2) if max_loss > 0 else 0.0
    premium_status = classify_premium_status(max_loss=max_loss, reward_risk=reward_risk, max_debit=max_debit)
    expiration = str((pd.Timestamp(signal_date) + pd.Timedelta(days=dte)).date())
    eligible = premium_status in {"OK", "ACCEPTABLE"}

    return DebitSpreadPlan(
        option_structure=f"Bull Call Debit Spread {long_strike:.2f}/{short_strike:.2f}",
        long_strike=long_strike,
        short_strike=short_strike,
        dte=dte,
        est_debit=est_debit,
        max_loss=max_loss,
        max_profit=max_profit,
        reward_risk=reward_risk,
        spread_width=spread_width,
        premium_status=premium_status,
        small_account_eligible=eligible,
        long_delta_target=long_delta_target,
        expiration=expiration,
        liquidity=liquidity_score_for_ticker(ticker),
        notes=f"{DEBIT_SPREAD_PLANNER_DISCLAIMER} {notes}".strip(),
    )


def build_bull_call_debit_spread(
    ticker: str,
    price: float,
    atr: float,
    signal_date: str,
    score: float,
    min_dte: int = 30,
    max_dte: int = 60,
    preferred_dte: int = 45,
    max_debit: float = 150.0,
    preferred_debit_min: float = 50.0,
    preferred_debit_max: float = 125.0,
) -> DebitSpreadPlan:
    atr_pct = atr / price if price > 0 else 0.0
    preferred_delta_target = preferred_delta(score=score, atr_pct=atr_pct)
    dte_candidates = []
    for dte in [preferred_dte, 30, 60]:
        bounded = max(min_dte, min(max_dte, dte))
        if bounded not in dte_candidates:
            dte_candidates.append(bounded)
    delta_candidates = []
    for delta_target in [preferred_delta_target, 0.60, 0.55, 0.50]:
        rounded = round(delta_target, 2)
        if rounded not in delta_candidates:
            delta_candidates.append(rounded)

    candidate_plans: list[DebitSpreadPlan] = []
    for dte in dte_candidates:
        for delta_target in delta_candidates:
            for short_target_r in [1.0, 1.5, 2.0]:
                candidate_plans.append(
                    _candidate_plan(
                        ticker=ticker,
                        price=price,
                        atr=atr,
                        signal_date=signal_date,
                        long_delta_target=delta_target,
                        dte=dte,
                        short_target_r=short_target_r,
                        max_debit=max_debit,
                        notes=f"Targeted short call at {short_target_r:.1f}R above the underlying.",
                    )
                )

    def sort_key(plan: DebitSpreadPlan) -> tuple[int, int, float, float, float]:
        status_rank = {
            "OK": 0,
            "ACCEPTABLE": 1,
            "BAD_REWARD_RISK": 2,
            "TOO_EXPENSIVE": 3,
            "N/A": 4,
        }[plan.premium_status]
        preferred_band_rank = 0 if preferred_debit_min <= plan.max_loss <= preferred_debit_max else 1
        debit_distance = abs(plan.max_loss - ((preferred_debit_min + preferred_debit_max) / 2.0))
        max_debit_penalty = 0.0 if plan.max_loss <= max_debit else plan.max_loss - max_debit
        return (status_rank, preferred_band_rank, -plan.reward_risk, debit_distance, max_debit_penalty)

    return sorted(candidate_plans, key=sort_key)[0]
