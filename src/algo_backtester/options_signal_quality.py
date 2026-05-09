from __future__ import annotations

from dataclasses import asdict, dataclass

from algo_backtester.market_regime import MarketRegimeSnapshot
from algo_backtester.strategies.swing_options import SwingSourceSignal


@dataclass(frozen=True)
class TimeframeConfirmation:
    aligned: bool
    four_hour_only: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OptionsSignalAssessment:
    market_regime: str
    regime_reason: str
    timeframe_confirmation: dict
    daily_trend: str
    four_hour_trend: str
    setup_score: float
    setup_rating: str
    evaluated_setup: str
    no_trade_reasons: list[str]
    warnings: list[str]
    final_decision: str

    def to_dict(self) -> dict:
        return asdict(self)


def classify_daily_trend(close_price: float, ema20: float, ema50: float, ema200: float) -> str:
    if close_price > ema20 > ema50 > ema200:
        return "BULLISH"
    if close_price < ema20 < ema50 < ema200:
        return "BEARISH"
    return "MIXED"


def classify_four_hour_trend(source: SwingSourceSignal) -> str:
    if source.signal == "BUY" or (source.trend_quality and source.recent_momentum):
        return "BULLISH"
    if source.signal in {"SHORT_SETUP", "SELL_SHORT", "BEARISH_ENTRY"} or (not source.trend_quality and not source.recent_momentum):
        return "BEARISH"
    return "MIXED"


def _four_hour_bullish_continuation(source: SwingSourceSignal) -> bool:
    if source.signal == "BUY":
        return True
    return source.setup in {"NEAR_SETUP", "NEEDS_PULLBACK"} and source.trend_quality and source.recent_momentum


def _timeframe_confirmation(daily_trend: str, four_hour_source: SwingSourceSignal) -> tuple[TimeframeConfirmation, str]:
    four_hour_trend = classify_four_hour_trend(four_hour_source)
    four_hour_support = _four_hour_bullish_continuation(four_hour_source)

    if daily_trend == "BULLISH" and four_hour_support:
        return TimeframeConfirmation(True, False, "Daily and 4H are aligned for a bullish continuation or recovery."), four_hour_trend
    if daily_trend != "BEARISH" and four_hour_support:
        return TimeframeConfirmation(False, True, "4H shows bullish continuation or recovery, but the daily trend is not fully aligned."), four_hour_trend
    if daily_trend == "BEARISH":
        return TimeframeConfirmation(False, False, "Daily trend is bearish, so bullish debit spreads are not allowed."), four_hour_trend
    return TimeframeConfirmation(False, False, "4H continuation or recovery is not strong enough yet."), four_hour_trend


def _earnings_within_five_days(earnings_date: str, signal_date: str) -> bool:
    if earnings_date in {"", "N/A", "UNKNOWN", "PLACEHOLDER_OK"}:
        return False
    try:
        import pandas as pd

        signal_ts = pd.Timestamp(signal_date)
        earnings_ts = pd.Timestamp(earnings_date)
    except Exception:
        return False
    business_days = len(pd.bdate_range(signal_ts, earnings_ts)) - 1
    return 0 <= business_days <= 5


def _risk_reward_quality(reward_risk: float) -> float:
    if reward_risk >= 2.0:
        return 10.0
    if reward_risk >= 1.5:
        return 7.0
    return 0.0


def _extension_quality(rsi: float, price_to_ema20: float) -> float:
    if rsi <= 62.0 and price_to_ema20 <= 1.02:
        return 10.0
    if rsi <= 66.0 and price_to_ema20 <= 1.04:
        return 7.0
    if rsi <= 68.0 and price_to_ema20 <= 1.05:
        return 4.0
    return 0.0


def _rate_setup(score: float) -> str:
    if score >= 85.0:
        return "A_SETUP"
    if score >= 70.0:
        return "B_SETUP"
    if score >= 55.0:
        return "WATCHLIST"
    return "NO_TRADE"


def _final_decision(evaluated_setup: str, no_trade_reasons: list[str], warnings: list[str]) -> str:
    if evaluated_setup == "NO_TRADE":
        primary_reason = no_trade_reasons[0] if no_trade_reasons else "signal quality is below threshold"
        return f"No trade. {primary_reason} Preserve capital and reassess after the next close."
    if evaluated_setup == "ACTIONABLE":
        return "Review before market open. Only enter if continuation structure remains intact and liquidity is acceptable."
    if warnings:
        return f"Watchlist only. {warnings[0]}"
    return "Watchlist only. Wait for stronger alignment before considering entry."


def evaluate_long_options_setup(
    *,
    signal_date: str,
    market_regime_snapshot: MarketRegimeSnapshot,
    daily_close: float,
    ema20: float,
    ema50: float,
    ema200: float,
    rsi: float,
    atr: float,
    avg_dollar_volume: float,
    volume_confirmed: bool,
    recent_momentum: bool,
    liquidity: str,
    earnings_date: str,
    reward_risk: float,
    four_hour_source: SwingSourceSignal,
) -> OptionsSignalAssessment:
    price_to_ema20 = daily_close / max(ema20, 1e-9)
    atr_pct = atr / max(daily_close, 1e-9)

    daily_trend = classify_daily_trend(daily_close, ema20, ema50, ema200)
    timeframe_confirmation, four_hour_trend = _timeframe_confirmation(daily_trend, four_hour_source)

    regime_alignment_score = {"BULLISH": 20.0, "MIXED": 10.0, "BEARISH": 0.0}[market_regime_snapshot.market_regime]
    daily_alignment_score = {"BULLISH": 20.0, "MIXED": 10.0, "BEARISH": 0.0}[daily_trend]
    if four_hour_source.signal == "BUY":
        four_hour_setup_quality = 25.0
    elif four_hour_source.setup in {"NEAR_SETUP", "NEEDS_PULLBACK"} and four_hour_source.trend_quality and four_hour_source.recent_momentum:
        four_hour_setup_quality = 18.0
    elif four_hour_source.trend_quality:
        four_hour_setup_quality = 12.0
    else:
        four_hour_setup_quality = 0.0

    base_score = (
        regime_alignment_score
        + daily_alignment_score
        + four_hour_setup_quality
        + (10.0 if volume_confirmed else 0.0)
        + _risk_reward_quality(reward_risk)
        + _extension_quality(rsi, price_to_ema20)
        + (5.0 if recent_momentum else 0.0)
    )

    no_trade_reasons: list[str] = []
    warnings: list[str] = []

    earnings_soon = _earnings_within_five_days(earnings_date, signal_date)
    atr_too_low = atr_pct < 0.015
    weak_volume = avg_dollar_volume < 20_000_000.0
    poor_liquidity = liquidity not in {"A+", "A", "B"}
    too_extended_from_ema20 = price_to_ema20 >= 1.06
    regime_conflict = market_regime_snapshot.market_regime == "BEARISH"
    move_exhaustion = rsi >= 68.0 and price_to_ema20 >= 1.05 and recent_momentum

    penalties = 0.0
    if regime_conflict:
        penalties += 20.0
        no_trade_reasons.append("Market regime strongly conflicts with a bullish debit spread.")
    if too_extended_from_ema20:
        penalties += 10.0
        no_trade_reasons.append("Price is too extended from EMA20.")
    if weak_volume:
        penalties += 10.0
        no_trade_reasons.append("Average volume is too weak.")
    if poor_liquidity:
        penalties += 10.0
        no_trade_reasons.append("Spread or liquidity is too poor.")
    if earnings_soon:
        penalties += 15.0
        no_trade_reasons.append("Earnings are within 5 trading days.")
    if atr_too_low:
        penalties += 10.0
        no_trade_reasons.append("ATR is too low.")
    if move_exhaustion:
        penalties += 10.0
        no_trade_reasons.append("Expected move exhaustion is already present.")

    if daily_trend == "BEARISH":
        warnings.append("Daily trend is bearish, so the setup cannot be promoted.")
    if not timeframe_confirmation.aligned and timeframe_confirmation.four_hour_only:
        warnings.append("Only the 4H timeframe aligns right now; keep this on watchlist status.")
    elif not timeframe_confirmation.aligned:
        warnings.append(timeframe_confirmation.reason)

    if market_regime_snapshot.market_regime == "MIXED":
        warnings.append("Market regime is mixed, so borderline setups should stay on watchlist.")

    setup_score = max(min(round(base_score - penalties, 2), 100.0), 0.0)
    setup_rating = _rate_setup(setup_score)

    if no_trade_reasons:
        evaluated_setup = "NO_TRADE"
    elif daily_trend == "BEARISH":
        evaluated_setup = "NO_TRADE"
    elif setup_rating == "NO_TRADE":
        evaluated_setup = "NO_TRADE"
    elif timeframe_confirmation.aligned and setup_rating in {"A_SETUP", "B_SETUP"}:
        evaluated_setup = "ACTIONABLE"
    elif timeframe_confirmation.four_hour_only:
        evaluated_setup = "WATCHLIST"
    else:
        evaluated_setup = "WATCHLIST"

    if (
        market_regime_snapshot.market_regime == "MIXED"
        and evaluated_setup == "ACTIONABLE"
        and 70.0 <= setup_score < 75.0
    ):
        evaluated_setup = "WATCHLIST"
        warnings.append("Mixed regime downgraded a borderline actionable setup to watchlist.")

    return OptionsSignalAssessment(
        market_regime=market_regime_snapshot.market_regime,
        regime_reason=market_regime_snapshot.regime_reason,
        timeframe_confirmation=timeframe_confirmation.to_dict(),
        daily_trend=daily_trend,
        four_hour_trend=four_hour_trend,
        setup_score=setup_score,
        setup_rating=setup_rating,
        evaluated_setup=evaluated_setup,
        no_trade_reasons=no_trade_reasons,
        warnings=warnings,
        final_decision=_final_decision(evaluated_setup, no_trade_reasons, warnings),
    )
