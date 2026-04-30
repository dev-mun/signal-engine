from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from algo_backtester.data_loader import load_yfinance_intraday_data, resample_to_4h, validate_ohlcv
from algo_backtester.strategies.four_hour_trend_pullback import (
    add_indicators,
    classify_setup,
    distance_to_setup,
    should_buy,
    should_sell_long,
    should_short_setup,
)


@dataclass(frozen=True)
class FourHourTrendConfig:
    initial_cash: float = 10_000.0
    stop_loss: float = 0.04
    take_profit: float = 0.08
    trailing_stop: float = 0.05
    max_hold_candles: int = 12
    risk_per_trade: float = 0.0075
    atr_multiple: float = 1.5
    interval: str = "1h"


def resample_to_four_hour(df: pd.DataFrame) -> pd.DataFrame:
    return resample_to_4h(df)


def prepare_four_hour_data(
        ticker: str,
        interval: str = "1h",
        period: str = "730d",
        prepost: bool = False,
) -> pd.DataFrame:
    intraday_df = load_yfinance_intraday_data(
        ticker=ticker,
        interval=interval,
        period=period,
        prepost=prepost,
    )

    if intraday_df.empty:
        return intraday_df

    four_hour_df = resample_to_4h(intraday_df)

    print(f"{ticker} | Fetched 1H candles: {len(intraday_df)}")
    print(f"{ticker} | Resampled 4H candles: {len(four_hour_df)}")

    if not four_hour_df.empty:
        print(f"{ticker} | Range: {four_hour_df.index[0].date()} -> {four_hour_df.index[-1].date()}")

    return four_hour_df


def make_demo_intraday_data(rows: int = 1600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01 09:00:00", periods=rows, freq="h")

    drift = 0.00025
    volatility = 0.006
    returns = rng.normal(drift, volatility, rows)

    close = 100 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.0015, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.008, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.008, rows))
    volume = rng.integers(100_000, 750_000, rows)
    volume[::24] = volume[::24] * 2

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


class FourHourTrendBacktester:
    def __init__(self, config: FourHourTrendConfig | None = None):
        self.config = config or FourHourTrendConfig()

    def run(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        base_df = validate_ohlcv(raw_df)

        if len(base_df.index) > 1:
            median_delta = pd.to_timedelta(base_df.index.to_series().diff().dropna().median()).total_seconds()
        else:
            median_delta = 0.0

        if median_delta and median_delta <= 5400:
            working_df = resample_to_4h(base_df)
        else:
            working_df = base_df

        working_df = add_indicators(validate_ohlcv(working_df))

        cash = float(self.config.initial_cash)
        shares = 0
        in_position = False
        entry_price = 0.0
        entry_index = 0
        highest_close_since_entry = 0.0
        initial_stop_price = 0.0
        pending_order: dict | None = None

        equity_curve: list[dict] = []
        trades: list[dict] = []
        signals: list[dict] = []

        for i in range(1, len(working_df)):
            row = working_df.iloc[i]
            prev = working_df.iloc[i - 1]
            date = working_df.index[i]
            open_price = float(row["Open"])
            close_price = float(row["Close"])

            if pending_order:
                action = pending_order["Action"]

                if action == "BUY":
                    risk_budget = cash * self.config.risk_per_trade
                    atr_risk = float(pending_order["ATR"]) * self.config.atr_multiple
                    shares = int(risk_budget // atr_risk) if atr_risk > 0 else 0
                    max_affordable = int(cash // open_price)
                    shares = min(shares, max_affordable)

                    if shares > 0:
                        cash -= shares * open_price
                        in_position = True
                        entry_price = open_price
                        entry_index = i
                        highest_close_since_entry = entry_price
                        initial_stop_price = entry_price * (1 - self.config.stop_loss)

                        trades.append(
                            {
                                "Date": date,
                                "SignalDate": pending_order["SignalDate"],
                                "Action": "BUY",
                                "Price": open_price,
                                "Shares": shares,
                                "ATRRisk": round(atr_risk, 2),
                                "InitialStop": round(initial_stop_price, 2),
                                "Reason": pending_order["Reason"],
                            }
                        )

                elif action == "SELL" and in_position and shares > 0:
                    pnl = (open_price - entry_price) / entry_price
                    hold_candles = i - entry_index
                    trailing_drawdown_pct = (
                        (open_price - highest_close_since_entry) / highest_close_since_entry * 100
                        if highest_close_since_entry > 0
                        else 0.0
                    )

                    cash += shares * open_price

                    trades.append(
                        {
                            "Date": date,
                            "SignalDate": pending_order["SignalDate"],
                            "Action": "SELL",
                            "Price": open_price,
                            "Shares": shares,
                            "PnL %": pnl * 100,
                            "Hold Candles": hold_candles,
                            "Highest Close Since Entry": highest_close_since_entry,
                            "Trailing Drawdown %": trailing_drawdown_pct,
                            "Reason": pending_order["Reason"],
                        }
                    )

                    shares = 0
                    in_position = False
                    entry_price = 0.0
                    entry_index = 0
                    highest_close_since_entry = 0.0
                    initial_stop_price = 0.0

                pending_order = None

            signal = "HOLD"
            reason = "No actionable four-hour setup."

            if in_position:
                highest_close_since_entry = max(highest_close_since_entry, close_price)
                exit_now, reason = should_sell_long(
                    row=row,
                    entry_price=entry_price,
                    highest_close_since_entry=highest_close_since_entry,
                    hold_candles=i - entry_index,
                    stop_loss_pct=self.config.stop_loss,
                    take_profit_pct=self.config.take_profit,
                    trailing_stop_pct=self.config.trailing_stop,
                    max_hold_candles=self.config.max_hold_candles,
                )

                if exit_now:
                    signal = "SELL"
                    pending_order = {
                        "Action": "SELL",
                        "SignalDate": date,
                        "Reason": reason,
                    }
                else:
                    signal = "HOLD_POSITION"
            else:
                buy_now, buy_reason = should_buy(row=row, prev=prev)
                short_now, short_reason = should_short_setup(row=row, prev=prev)

                if buy_now:
                    signal = "BUY"
                    reason = buy_reason
                    pending_order = {
                        "Action": "BUY",
                        "SignalDate": date,
                        "Reason": buy_reason,
                        "ATR": float(row["ATR"]),
                    }
                elif short_now:
                    signal = "SHORT_SETUP"
                    reason = short_reason
                else:
                    reason = "No actionable four-hour setup."

            equity = cash + shares * close_price
            equity_curve.append({"Date": date, "Equity": equity})
            signals.append(
                {
                    "Date": date,
                    "Signal": signal,
                    "Reason": reason,
                    "Close": close_price,
                    "RSI": float(row["RSI"]),
                    "RSI14": float(row["RSI14"]),
                    "EMA20": float(row["EMA20"]),
                    "EMA50": float(row["EMA50"]),
                    "EMA200": float(row["EMA200"]),
                    "Volume": float(row["Volume"]),
                    "AverageVolume20": float(row["AVG_VOL20"]),
                    "InPositionAfterSignal": in_position,
                    "Equity": equity,
                    "ATR": float(row["ATR"]),
                    "ATR14": float(row["ATR14"]),
                    "EntryPrice": entry_price if in_position else 0.0,
                    "InitialStopPrice": initial_stop_price if in_position else 0.0,
                    "PendingOrderAction": pending_order["Action"] if pending_order else "",
                }
            )

        equity_df = pd.DataFrame(equity_curve).set_index("Date") if equity_curve else pd.DataFrame(columns=["Equity"])
        trades_df = pd.DataFrame(trades)
        signals_df = pd.DataFrame(signals).set_index("Date") if signals else pd.DataFrame()

        return working_df, equity_df, trades_df, signals_df


def _build_scan_result(ticker: str, signals_df: pd.DataFrame) -> dict:
    latest = signals_df.iloc[-1]
    signal = str(latest["Signal"])

    return {
        "Ticker": ticker,
        "Strategy": "four-hour-trend",
        "Signal": signal,
        "Setup": classify_setup(signal, latest),
        "Price": float(latest["Close"]),
        "RSI": float(latest["RSI"]),
        "ATR": float(latest["ATR"]),
        "Distance": distance_to_setup(signal, latest),
        "Reason": str(latest["Reason"]),
        "SignalDate": str(signals_df.index[-1]),
    }


def scan_ticker(
        ticker: str,
        interval: str = "1h",
        config: FourHourTrendConfig | None = None,
) -> dict:
    raw_df = prepare_four_hour_data(
        ticker=ticker,
        interval=interval,
    )
    if raw_df.empty:
        raise ValueError(f"No usable intraday data found for ticker: {ticker}")
    bt = FourHourTrendBacktester(config=config)
    _, _, _, signals_df = bt.run(raw_df)
    return _build_scan_result(ticker=ticker, signals_df=signals_df)


def scan_watchlist(
        tickers: Iterable[str],
        interval: str = "1h",
        config: FourHourTrendConfig | None = None,
) -> list[dict]:
    results = []

    for ticker in tickers:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            continue

        try:
            results.append(
                scan_ticker(
                    ticker=clean_ticker,
                    interval=interval,
                    config=config,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "Ticker": clean_ticker,
                    "Strategy": "four-hour-trend",
                    "Signal": "ERROR",
                    "Setup": "ERROR",
                    "Price": 0.0,
                    "RSI": 0.0,
                    "ATR": 0.0,
                    "Distance": "ERROR",
                    "Reason": str(exc),
                    "SignalDate": str(pd.Timestamp.today()),
                }
            )

    return results
