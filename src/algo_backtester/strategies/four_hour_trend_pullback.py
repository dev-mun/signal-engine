from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["AVG_VOL20"] = df["Volume"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (100 / (1 + rs))
    df["RSI14"] = df["RSI14"].fillna(50)
    df["RSI"] = df["RSI14"]

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()
    df["ATR"] = df["ATR14"]

    return df.dropna()


def is_bullish_trend(row: pd.Series) -> bool:
    return (
        float(row["Close"]) > float(row["EMA200"])
        and float(row["EMA20"]) > float(row["EMA50"])
        and float(row["EMA50"]) > float(row["EMA200"])
    )


def is_bearish_trend(row: pd.Series) -> bool:
    return (
        float(row["Close"]) < float(row["EMA200"])
        and float(row["EMA20"]) < float(row["EMA50"])
        and float(row["EMA50"]) < float(row["EMA200"])
    )


def should_buy(row: pd.Series, prev: pd.Series) -> tuple[bool, str]:
    close_price = float(row["Close"])
    open_price = float(row["Open"])

    conditions = [
        is_bullish_trend(row),
        45 <= float(row["RSI"]) <= 60,
        float(prev["Close"]) <= float(prev["EMA20"]) and close_price > float(row["EMA20"]),
        float(row["Volume"]) > float(row["AVG_VOL20"]),
        close_price > float(prev["High"]) or close_price > open_price,
    ]

    if all(conditions):
        return True, "4H bullish trend continuation entry confirmed."

    return False, "No actionable four-hour bullish continuation setup."


def should_short_setup(row: pd.Series, prev: pd.Series) -> tuple[bool, str]:
    close_price = float(row["Close"])
    open_price = float(row["Open"])

    conditions = [
        is_bearish_trend(row),
        40 <= float(row["RSI"]) <= 55,
        float(prev["Close"]) >= float(prev["EMA20"]) and close_price < float(row["EMA20"]),
        float(row["Volume"]) > float(row["AVG_VOL20"]),
        close_price < float(prev["Low"]) or close_price < open_price,
    ]

    if all(conditions):
        return True, "4H bearish continuation setup confirmed."

    return False, "No actionable four-hour bearish continuation setup."


def should_sell_long(
        row: pd.Series,
        entry_price: float,
        highest_close_since_entry: float,
        hold_candles: int,
        stop_loss_pct: float = 0.04,
        take_profit_pct: float = 0.08,
        trailing_stop_pct: float = 0.05,
        max_hold_candles: int = 12,
) -> tuple[bool, str]:
    close_price = float(row["Close"])
    ema50 = float(row["EMA50"])

    stop_price = entry_price * (1 - stop_loss_pct)
    take_profit_price = entry_price * (1 + take_profit_pct)
    trailing_stop_price = highest_close_since_entry * (1 - trailing_stop_pct)

    if close_price <= stop_price:
        return True, "Stop loss triggered."

    if close_price >= take_profit_price:
        return True, "Take profit reached."

    if close_price <= trailing_stop_price:
        return True, "Trailing stop triggered."

    if hold_candles >= max_hold_candles:
        return True, "Max hold reached."

    if close_price < ema50:
        return True, "Close fell below EMA50."

    return False, "Position remains valid."


def classify_setup(signal: str, row: pd.Series) -> str:
    rsi = float(row["RSI"])
    bullish_trend = is_bullish_trend(row)
    bearish_trend = is_bearish_trend(row)

    if signal in {"BUY", "SELL", "SHORT_SETUP", "SELL_SHORT", "COVER"}:
        return "ACTIONABLE"

    if rsi >= 70:
        return "EXTENDED"

    if bullish_trend and 55 <= rsi < 60:
        return "NEAR_SETUP"

    if bearish_trend and 35 <= rsi < 40:
        return "NEAR_SETUP"

    if bullish_trend and rsi > 60:
        return "NEEDS_PULLBACK"

    if bearish_trend and rsi < 35:
        return "NEEDS_PULLBACK"

    if bullish_trend and rsi < 35:
        return "WEAK"

    return "WAIT"


def distance_to_setup(signal: str, row: pd.Series) -> str:
    rsi = float(row["RSI"])
    bullish_trend = is_bullish_trend(row)
    bearish_trend = is_bearish_trend(row)

    if signal in {"BUY", "SELL", "SHORT_SETUP", "SELL_SHORT", "COVER"}:
        return "Actionable now"

    if rsi >= 70:
        return "Too extended"

    if bullish_trend and 55 <= rsi < 60:
        return "Near long setup"

    if bearish_trend and 35 <= rsi < 40:
        return "Near short setup"

    if bullish_trend and rsi > 60:
        return "Needs RSI pullback"

    if bearish_trend and rsi < 35:
        return "Needs RSI pullback"

    if bullish_trend and 45 <= rsi < 55:
        return "Waiting for EMA20 reclaim"

    if bearish_trend and 40 <= rsi < 55:
        return "Waiting for EMA20 breakdown"

    if bullish_trend and rsi < 35:
        return "Too weak"

    return "Trend not aligned"
