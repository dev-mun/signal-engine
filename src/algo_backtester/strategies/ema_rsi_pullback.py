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


def should_buy(row: pd.Series, prev: pd.Series) -> tuple[bool, str]:
    conditions = [
        float(row["Close"]) > float(row["EMA200"]),
        float(row["EMA20"]) > float(row["EMA50"]),
        float(row["EMA50"]) > float(row["EMA200"]),
        40 <= float(row["RSI"]) <= 55,
        float(prev["Close"]) <= float(prev["EMA20"]) and float(row["Close"]) > float(row["EMA20"]),
        float(row["Volume"]) > float(row["AVG_VOL20"]),
    ]

    if all(conditions):
        return True, "EMA RSI pullback entry confirmed."

    return False, "No actionable EMA RSI pullback entry."


def should_sell(
        row: pd.Series,
        entry_price: float,
        highest_close_since_entry: float,
        hold_days: int,
        stop_loss_pct: float = 0.07,
        take_profit_pct: float = 0.15,
        trailing_stop_pct: float = 0.08,
        max_hold_days: int = 45,
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

    if hold_days >= max_hold_days:
        return True, "Max hold reached."

    if close_price < ema50:
        return True, "Close fell below EMA50."

    return False, "Position remains valid."


def classify_setup(signal: str, rsi: float) -> str:
    if signal in {"BUY", "SELL"}:
        return "ACTIONABLE"

    if rsi >= 70:
        return "EXTENDED"

    if 60 <= rsi < 70:
        return "NEEDS_PULLBACK"

    if 55 <= rsi < 60:
        return "NEAR_SETUP"

    if 40 <= rsi < 55:
        return "WAIT"

    return "WEAK"


def distance_to_setup(signal: str, rsi: float) -> str:
    if signal in {"BUY", "SELL"}:
        return "Actionable now"

    if rsi >= 70:
        return "Too hot"

    if 60 <= rsi < 70:
        return f"Needs pullback (-{rsi - 55:.1f} RSI)"

    if 55 <= rsi < 60:
        return "Near setup"

    if 40 <= rsi < 55:
        return "Waiting for EMA20 reclaim"

    return "Too weak"
