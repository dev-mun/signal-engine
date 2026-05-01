from __future__ import annotations

import numpy as np
import pandas as pd


def trend_quality_passes(row: pd.Series) -> bool:
    close_price = float(row["Close"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])
    return close_price > ema200 and ema50 > ema200 and close_price >= ema50 * 0.95


def close_position_in_range(row: pd.Series, close_position_min: float = 0.35) -> bool:
    high_price = float(row["High"])
    low_price = float(row["Low"])
    close_price = float(row["Close"])
    candle_range = high_price - low_price

    if candle_range <= 0:
        close_position = 0.5
    else:
        close_position = (close_price - low_price) / candle_range

    return close_position >= close_position_min


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    frame["EMA50"] = frame["Close"].ewm(span=50, adjust=False).mean()
    frame["EMA200"] = frame["Close"].ewm(span=200, adjust=False).mean()
    frame["AVG_VOL20"] = frame["Volume"].rolling(20).mean()

    rolling_mean = frame["Close"].rolling(20).mean()
    rolling_std = frame["Close"].rolling(20).std()
    frame["BB_MIDDLE"] = rolling_mean
    frame["BB_UPPER"] = rolling_mean + 2 * rolling_std
    frame["BB_LOWER"] = rolling_mean - 2 * rolling_std

    delta = frame["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)

    frame["RSI14"] = 100 - (100 / (1 + rs))
    frame["RSI14"] = frame["RSI14"].fillna(50)
    frame["RSI"] = frame["RSI14"]

    high_low = frame["High"] - frame["Low"]
    high_close = (frame["High"] - frame["Close"].shift(1)).abs()
    low_close = (frame["Low"] - frame["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    frame["ATR14"] = true_range.rolling(14).mean()
    frame["ATR"] = frame["ATR14"]

    return frame.dropna()


def near_lower_band(row: pd.Series, band_tolerance: float = 1.02) -> bool:
    close_price = float(row["Close"])
    low_price = float(row["Low"])
    lower_band = float(row["BB_LOWER"])
    return low_price <= lower_band * band_tolerance or close_price <= lower_band * band_tolerance


def close_below_middle_band(row: pd.Series) -> bool:
    return float(row["Close"]) <= float(row["BB_MIDDLE"])


def should_buy(
        row: pd.Series,
        prev: pd.Series,
        rsi_threshold: float = 38.0,
        volume_multiplier: float = 0.6,
        band_tolerance: float = 1.02,
        close_position_min: float = 0.35,
        require_confirmation: bool = False,
) -> tuple[bool, str]:
    close_price = float(row["Close"])
    avg_volume = float(row["AVG_VOL20"])
    confirmation_ok = close_price > float(prev["Close"]) or close_price > float(row["Open"])

    conditions = [
        trend_quality_passes(row),
        float(row["RSI"]) <= rsi_threshold,
        close_below_middle_band(row),
        near_lower_band(row, band_tolerance=band_tolerance),
        close_price >= float(row["BB_LOWER"]) * 0.90,
        float(row["Volume"]) >= avg_volume * volume_multiplier,
        close_position_in_range(row, close_position_min=close_position_min),
        confirmation_ok if require_confirmation else True,
    ]

    if all(conditions):
        return True, "RSI Bollinger V2 entry confirmed."

    return False, "No actionable RSI Bollinger V2 setup."


def should_sell(
        row: pd.Series,
        entry_price: float,
        highest_close_since_entry: float,
        hold_days: int,
        stop_loss_pct: float = 0.04,
        take_profit_pct: float = 0.05,
        trailing_stop_pct: float = 0.04,
        max_hold_days: int = 7,
) -> tuple[bool, str]:
    close_price = float(row["Close"])

    stop_price = entry_price * (1 - stop_loss_pct)
    take_profit_price = entry_price * (1 + take_profit_pct)
    trailing_stop_price = highest_close_since_entry * (1 - trailing_stop_pct)

    if close_price <= stop_price:
        return True, "Stop loss triggered."

    if close_price >= take_profit_price:
        return True, "Take profit reached."

    if close_price >= float(row["BB_MIDDLE"]):
        return True, "Price reached Bollinger middle band."

    if float(row["RSI"]) >= 50:
        return True, "RSI mean reversion exit triggered."

    if close_price <= trailing_stop_price:
        return True, "Trailing stop triggered."

    if hold_days >= max_hold_days:
        return True, "Max hold reached."

    return False, "Position remains valid."


def classify_setup(signal: str, row: pd.Series, band_tolerance: float = 1.02) -> str:
    close_price = float(row["Close"])
    rsi = float(row["RSI"])

    if signal in {"BUY", "SELL"}:
        return "ACTIONABLE"

    if close_price < float(row["EMA200"]) or float(row["EMA50"]) <= float(row["EMA200"]):
        return "WEAK_TREND"

    if rsi >= 65 or close_price >= float(row["BB_UPPER"]):
        return "EXTENDED"

    if rsi <= 35 and near_lower_band(row, band_tolerance=band_tolerance) and trend_quality_passes(row):
        return "OVERSOLD"

    if rsi <= 42 and close_below_middle_band(row) and trend_quality_passes(row):
        return "NEAR_SETUP"

    return "WAIT"


def distance_to_setup(signal: str, row: pd.Series, band_tolerance: float = 1.02) -> str:
    close_price = float(row["Close"])
    rsi = float(row["RSI"])

    if signal in {"BUY", "SELL"}:
        return "Actionable now"

    if close_price < float(row["EMA200"]) or float(row["EMA50"]) <= float(row["EMA200"]):
        return "Below EMA200 or weak trend, skip"

    if rsi >= 65 or close_price >= float(row["BB_UPPER"]):
        return "Extended, avoid new long"

    if rsi <= 35 and near_lower_band(row, band_tolerance=band_tolerance) and trend_quality_passes(row):
        return "Oversold bounce candidate"

    if rsi <= 42 and close_below_middle_band(row) and trend_quality_passes(row):
        return "Near mean reversion setup"

    if not near_lower_band(row, band_tolerance=band_tolerance):
        return "Waiting for lower band test"

    return "No setup"
