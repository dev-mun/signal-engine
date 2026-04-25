import numpy as np
import pandas as pd

from src.algo_backtester.data_loader import validate_ohlcv


def performance_summary(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_cash: float = 10_000.0) -> dict:
    if equity_df.empty:
        raise ValueError("Equity curve is empty.")

    final_value = float(equity_df["Equity"].iloc[-1])
    total_return = (final_value / initial_cash - 1) * 100

    daily_returns = equity_df["Equity"].pct_change().dropna()
    daily_std = daily_returns.std()

    sharpe = 0.0 if daily_std == 0 or np.isnan(daily_std) else float(np.sqrt(252) * daily_returns.mean() / daily_std)

    rolling_max = equity_df["Equity"].cummax()
    drawdown = equity_df["Equity"] / rolling_max - 1
    max_drawdown = float(drawdown.min() * 100)

    sells = pd.DataFrame()
    if not trades_df.empty and "Action" in trades_df.columns:
        sells = trades_df[trades_df["Action"] == "SELL"]

    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0

    if not sells.empty and "PnL %" in sells.columns:
        wins = sells[sells["PnL %"] > 0]
        losses = sells[sells["PnL %"] <= 0]

        win_rate = float((sells["PnL %"] > 0).mean() * 100)
        avg_win = float(wins["PnL %"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["PnL %"].mean()) if not losses.empty else 0.0

        gross_profit = float(wins["PnL %"].sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses["PnL %"].sum())) if not losses.empty else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return {
        "Final Value": final_value,
        "Total Return %": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": max_drawdown,
        "Completed Trades": int(len(sells)),
        "Win Rate %": win_rate,
        "Average Win %": avg_win,
        "Average Loss %": avg_loss,
        "Profit Factor": profit_factor,
    }


def print_summary(summary: dict) -> None:
    print("Performance Summary")
    print("-------------------")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value}")


def buy_and_hold_curve(raw_df: pd.DataFrame, equity_index: pd.Index, initial_cash: float = 10_000.0) -> pd.DataFrame:
    df = validate_ohlcv(raw_df)
    df = df.loc[df.index.intersection(equity_index)].copy()

    if df.empty:
        raise ValueError("No overlapping dates between strategy equity and buy-and-hold data.")

    first_price = df["Close"].iloc[0]
    df["BuyHold"] = initial_cash * (df["Close"] / first_price)

    return df[["BuyHold"]]
