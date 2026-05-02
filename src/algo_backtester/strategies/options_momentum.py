from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSignalSnapshot:
    ticker: str
    source_strategy: str
    signal: str
    setup: str
    price: float
    rsi: float
    atr: float
    signal_date: str
    reason: str
    trend_quality: bool
    volume_confirmed: bool
    earnings_risk: bool
    acceptable_volatility: bool
    score: float
    qualified: bool
    notes: str


def trend_quality_strong(row) -> bool:
    return (
        float(row["Close"]) > float(row["EMA200"])
        and float(row["EMA20"]) > float(row["EMA50"])
        and float(row["EMA50"]) > float(row["EMA200"])
    )


def volume_confirmed(row) -> bool:
    return float(row["Volume"]) > float(row["AverageVolume20"])


def acceptable_volatility_structure(price: float, atr: float) -> bool:
    if price <= 0 or atr <= 0:
        return False
    atr_pct = atr / price
    return 0.01 <= atr_pct <= 0.12


def placeholder_earnings_risk(_ticker: str) -> bool:
    return False


def build_snapshot(
    ticker: str,
    source_strategy: str,
    latest_row,
    signal: str,
    setup: str,
    signal_date: str,
    reason: str,
) -> SourceSignalSnapshot:
    price = float(latest_row["Close"])
    rsi = float(latest_row["RSI"])
    atr = float(latest_row["ATR"])
    trend_quality = trend_quality_strong(latest_row)
    confirmed_volume = volume_confirmed(latest_row)
    earnings_risk = placeholder_earnings_risk(ticker)
    acceptable_volatility = acceptable_volatility_structure(price=price, atr=atr)

    qualified = (
        signal == "BUY"
        and setup == "ACTIONABLE"
        and trend_quality
        and confirmed_volume
        and not earnings_risk
        and acceptable_volatility
    )

    volume_ratio = float(latest_row["Volume"]) / max(float(latest_row["AverageVolume20"]), 1.0)
    base_score = 2.0 if source_strategy == "four-hour-trend" else 1.5
    score = base_score + min(volume_ratio, 3.0)

    rejection_notes: list[str] = []
    if signal != "BUY":
        rejection_notes.append("Source signal is not BUY.")
    if setup != "ACTIONABLE":
        rejection_notes.append("Source setup is not ACTIONABLE.")
    if not trend_quality:
        rejection_notes.append("Trend quality is not strong.")
    if not confirmed_volume:
        rejection_notes.append("Volume confirmation failed.")
    if earnings_risk:
        rejection_notes.append("Earnings risk placeholder rejected the setup.")
    if not acceptable_volatility:
        rejection_notes.append("ATR/volatility structure is outside the acceptable range.")

    notes = "Qualified for options overlay." if qualified else " | ".join(rejection_notes)

    return SourceSignalSnapshot(
        ticker=ticker,
        source_strategy=source_strategy,
        signal=signal,
        setup=setup,
        price=price,
        rsi=rsi,
        atr=atr,
        signal_date=signal_date,
        reason=reason,
        trend_quality=trend_quality,
        volume_confirmed=confirmed_volume,
        earnings_risk=earnings_risk,
        acceptable_volatility=acceptable_volatility,
        score=score,
        qualified=qualified,
        notes=notes,
    )
