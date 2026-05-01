from __future__ import annotations

import numpy as np
import pandas as pd


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


def _near_lower_band(row: pd.Series) -> bool:
    close_price = float(row["Close"])
    low_price = float(row["Low"])
    lower_band = float(row["BB_LOWER"])
    return close_price <= lower_band * 1.01 or low_price <= lower_band


def _near_upper_band(row: pd.Series) -> bool:
    close_price = float(row["Close"])
    high_price = float(row["High"])
    upper_band = float(row["BB_UPPER"])
    return close_price >= upper_band * 0.99 or high_price >= upper_band


def should_buy(row: pd.Series, prev: pd.Series) -> tuple[bool, str]:
    close_price = float(row["Close"])
    ema50 = float(row["EMA50"])
    avg_volume = float(row["AVG_VOL20"])

    conditions = [
        close_price > float(row["EMA200"]),
        close_price <= float(row["BB_LOWER"]) or float(row["Low"]) <= float(row["BB_LOWER"]),
        float(row["RSI"]) <= 35,
        close_price >= ema50 * 0.92,
        float(row["Volume"]) >= avg_volume * 0.8,
        close_price > float(prev["Close"]) or close_price > float(row["Open"]),
    ]

    if all(conditions):
        return True, "RSI Bollinger mean reversion entry confirmed."

    return False, "No actionable RSI Bollinger mean reversion setup."


def should_sell(
        row: pd.Series,
        entry_price: float,
        highest_close_since_entry: float,
        hold_days: int,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.06,
        trailing_stop_pct: float = 0.04,
        max_hold_days: int = 10,
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


def classify_setup(signal: str, row: pd.Series) -> str:
    rsi = float(row["RSI"])
    close_price = float(row["Close"])
    ema200 = float(row["EMA200"])
    near_lower = _near_lower_band(row)
    near_upper = _near_upper_band(row)

    if signal in {"BUY", "SELL"}:
        return "ACTIONABLE"

    if close_price < ema200:
        return "WEAK_TREND"

    if rsi >= 65 or near_upper:
        return "EXTENDED"

    if rsi <= 35 and near_lower:
        return "OVERSOLD"

    if 35 < rsi <= 42 and near_lower:
        return "NEAR_SETUP"

    if not near_lower:
        return "WAIT"

    return "WAIT"


def distance_to_setup(signal: str, row: pd.Series) -> str:
    rsi = float(row["RSI"])
    close_price = float(row["Close"])
    ema200 = float(row["EMA200"])
    near_lower = _near_lower_band(row)
    near_upper = _near_upper_band(row)

    if signal in {"BUY", "SELL"}:
        return "Actionable now"

    if close_price < ema200:
        return "Below EMA200, skip"

    if rsi >= 65 or near_upper:
        return "Extended, avoid new long"

    if rsi <= 35 and near_lower:
        return "Oversold bounce candidate"

    if 35 < rsi <= 42 and near_lower:
        return "Near mean reversion setup"

    if not near_lower:
        return "Waiting for lower band test"

    return "No setup"
