from __future__ import annotations

from typing import Tuple

import pandas as pd

from algo_backtester.config import BacktestConfig
from algo_backtester.data_loader import validate_ohlcv
from algo_backtester.indicators import add_indicators
from algo_backtester.strategy import (
    should_bearish_entry,
    should_buy,
    should_exit_long,
)


class TrendPullbackBacktester:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
            self,
            raw_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

            signal = "HOLD"
            reason = "No actionable setup"

            if pending_order:
                action = pending_order["Action"]

                if action == "BUY":
                    risk_budget = cash * self.config.risk_per_trade
                    stop_price = open_price * (1 - self.config.stop_loss)
                    risk_per_share = open_price - stop_price

                    shares = int(risk_budget // risk_per_share) if risk_per_share > 0 else 0
                    max_affordable = int(cash // open_price)
                    shares = min(shares, max_affordable)

                    if shares > 0:
                        cash -= shares * open_price
                        in_position = True
                        entry_price = open_price
                        entry_index = i
                        highest_close_since_entry = entry_price
                        initial_stop_price = stop_price

                        trades.append(
                            {
                                "Date": date,
                                "SignalDate": pending_order["SignalDate"],
                                "Action": "BUY",
                                "Price": open_price,
                                "Shares": shares,
                                "InitialStop": initial_stop_price,
                                "Reason": pending_order["Reason"],
                            }
                        )

                elif action == "EXIT_LONG" and in_position and shares > 0:
                    pnl = (open_price - entry_price) / entry_price
                    hold_days = i - entry_index
                    trailing_drawdown_pct = (
                        (open_price - highest_close_since_entry)
                        / highest_close_since_entry
                        * 100
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

            if in_position:
                highest_close_since_entry = max(highest_close_since_entry, close_price)

                pnl = (close_price - entry_price) / entry_price
                hold_days = i - entry_index

                exit_signal, exit_reason = should_exit_long(
                    row=row,
                    pnl=pnl,
                    hold_days=hold_days,
                    highest_close_since_entry=highest_close_since_entry,
                    stop_loss=self.config.stop_loss,
                    take_profit=self.config.take_profit,
                    trailing_stop=self.config.trailing_stop,
                    max_hold_days=self.config.max_hold_days,
                )

                if exit_signal:
                    signal = "EXIT_LONG"
                    reason = exit_reason
                    pending_order = {
                        "Action": "EXIT_LONG",
                        "SignalDate": date,
                        "Reason": reason,
                    }
                else:
                    signal = "HOLD_POSITION"
                    reason = "Existing position still valid"
            elif should_buy(prev, row, in_position):
                signal = "BUY"
                reason = "Trend pullback bullish entry"
                pending_order = {
                    "Action": "BUY",
                    "SignalDate": date,
                    "Reason": reason,
                }
            elif should_bearish_entry(prev, row, in_position):
                signal = "BEARISH_ENTRY"
                reason = "Trend pullback bearish entry"

            equity = cash + shares * close_price

            equity_curve.append(
                {
                    "Date": date,
                    "Equity": equity,
                }
            )

            signals.append(
                {
                    "Date": date,
                    "Signal": signal,
                    "Reason": reason,
                    "Close": close_price,
                    "RSI": float(row["RSI"]),
                    "ATR": float(row["ATR"]),
                    "EMA20": float(row["EMA20"]),
                    "EMA50": float(row["EMA50"]),
                    "EMA200": float(row["EMA200"]),
                    "Volume": float(row["Volume"]),
                    "AverageVolume20": float(row["AVG_VOL20"]),
                    "InPositionAfterSignal": in_position,
                    "EntryPrice": entry_price if in_position else 0.0,
                    "InitialStopPrice": initial_stop_price if in_position else 0.0,
                    "HighestCloseSinceEntry": (
                        highest_close_since_entry if in_position else 0.0
                    ),
                    "PendingOrderAction": pending_order["Action"] if pending_order else "",
                    "Equity": equity,
                }
            )

        equity_df = pd.DataFrame(equity_curve).set_index("Date")
        trades_df = pd.DataFrame(trades)
        signals_df = pd.DataFrame(signals).set_index("Date")

        return df, equity_df, trades_df, signals_df
