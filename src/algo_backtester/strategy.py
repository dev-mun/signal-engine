import pandas as pd


def crossed_above_ema20(prev: pd.Series, row: pd.Series) -> bool:
    return bool(prev["Close"] <= prev["EMA20"] and row["Close"] > row["EMA20"])


def crossed_below_ema20(prev: pd.Series, row: pd.Series) -> bool:
    return bool(prev["Close"] >= prev["EMA20"] and row["Close"] < row["EMA20"])


def should_buy(prev: pd.Series, row: pd.Series, in_position: bool) -> bool:
    return bool(
        not in_position
        and row["Close"] > row["EMA200"]
        and row["EMA50"] > row["EMA200"]
        and 40 <= row["RSI"] <= 60
        and crossed_above_ema20(prev, row)
        and row["Volume"] > row["AVG_VOL20"]
    )


def should_bearish_entry(prev: pd.Series, row: pd.Series, in_position: bool) -> bool:
    return bool(
        not in_position
        and row["Close"] < row["EMA200"]
        and row["EMA50"] < row["EMA200"]
        and 40 <= row["RSI"] <= 60
        and crossed_below_ema20(prev, row)
        and row["Volume"] > row["AVG_VOL20"]
    )


def should_exit_long(
        row,
        pnl: float,
        hold_days: int,
        highest_close_since_entry: float,
        stop_loss: float,
        take_profit: float,
        trailing_stop: float,
        max_hold_days: int,
) -> tuple[bool, str]:
    close = float(row["Close"])

    if pnl <= -stop_loss:
        return True, "Stop loss"

    if pnl >= take_profit:
        return True, "Take profit"

    trailing_drawdown = (close - highest_close_since_entry) / highest_close_since_entry

    if trailing_drawdown <= -trailing_stop:
        return True, "Trailing stop"

    if hold_days >= max_hold_days:
        return True, "Max hold period"

    return False, ""
