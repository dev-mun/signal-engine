from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from algo_backtester.data_loader import load_yfinance_data, validate_ohlcv
from algo_backtester.strategies.ema_rsi_pullback import (
    add_indicators,
    classify_setup,
    distance_to_setup,
    should_buy,
    should_sell,
)


@dataclass(frozen=True)
class EmaRsiBacktestConfig:
    initial_cash: float = 10_000.0
    stop_loss: float = 0.07
    take_profit: float = 0.15
    trailing_stop: float = 0.08
    max_hold_days: int = 45
    risk_per_trade: float = 0.01
    atr_multiple: float = 2.0


class EmaRsiPullbackBacktester:
    def __init__(self, config: EmaRsiBacktestConfig | None = None):
        self.config = config or EmaRsiBacktestConfig()

    def run(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = add_indicators(validate_ohlcv(raw_df))

        cash = float(self.config.initial_cash)
        shares = 0
        in_position = False
        entry_price = 0.0
        entry_index = 0
        highest_close_since_entry = 0.0
        initial_stop_price = 0.0
        pending_order = None

        equity_curve = []
        trades = []
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            date = df.index[i]
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
                    hold_days = i - entry_index
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
                            "Hold Days": hold_days,
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
            reason = "No actionable setup."

            if in_position:
                highest_close_since_entry = max(highest_close_since_entry, close_price)
                exit_now, reason = should_sell(
                    row=row,
                    entry_price=entry_price,
                    highest_close_since_entry=highest_close_since_entry,
                    hold_days=i - entry_index,
                    stop_loss_pct=self.config.stop_loss,
                    take_profit_pct=self.config.take_profit,
                    trailing_stop_pct=self.config.trailing_stop,
                    max_hold_days=self.config.max_hold_days,
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
                buy_now, reason = should_buy(row=row, prev=prev)
                if buy_now:
                    signal = "BUY"
                    pending_order = {
                        "Action": "BUY",
                        "SignalDate": date,
                        "Reason": reason,
                        "ATR": float(row["ATR"]),
                    }

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

        return df, equity_df, trades_df, signals_df


def _build_scan_result(ticker: str, signals_df: pd.DataFrame) -> dict:
    latest = signals_df.iloc[-1]
    signal = str(latest["Signal"])
    rsi = float(latest["RSI"])

    return {
        "Ticker": ticker,
        "Strategy": "ema-rsi",
        "Signal": signal,
        "Setup": classify_setup(signal, rsi),
        "Price": float(latest["Close"]),
        "RSI": rsi,
        "ATR": float(latest["ATR"]),
        "Distance": distance_to_setup(signal, rsi),
        "Reason": str(latest["Reason"]),
        "SignalDate": str(signals_df.index[-1].date()),
    }


def scan_ticker(
        ticker: str,
        start: str = "2018-01-01",
        end: str | None = None,
        config: EmaRsiBacktestConfig | None = None,
) -> dict:
    raw_df = load_yfinance_data(ticker=ticker, start=start, end=end)
    bt = EmaRsiPullbackBacktester(config=config)
    _, _, _, signals_df = bt.run(raw_df)
    return _build_scan_result(ticker=ticker, signals_df=signals_df)


def scan_watchlist(
        tickers: Iterable[str],
        start: str = "2018-01-01",
        end: str | None = None,
        config: EmaRsiBacktestConfig | None = None,
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
                    start=start,
                    end=end,
                    config=config,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "Ticker": clean_ticker,
                    "Strategy": "ema-rsi",
                    "Signal": "ERROR",
                    "Setup": "ERROR",
                    "Price": 0.0,
                    "RSI": 0.0,
                    "ATR": 0.0,
                    "Distance": "ERROR",
                    "Reason": str(exc),
                    "SignalDate": str(pd.Timestamp.today().date()),
                }
            )

    return results
