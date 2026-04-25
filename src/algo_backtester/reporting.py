from pathlib import Path

import pandas as pd


def latest_signal(signals_df: pd.DataFrame) -> dict:
    if signals_df.empty:
        return {
            "Signal": "NO_DATA",
            "Reason": "No signal data available",
        }

    row = signals_df.iloc[-1]

    return {
        "Date": str(signals_df.index[-1].date()),
        "Signal": row["Signal"],
        "Reason": row["Reason"],
        "Close": float(row["Close"]),
        "RSI": float(row["RSI"]),
        "EMA20": float(row["EMA20"]),
        "EMA50": float(row["EMA50"]),
        "EMA200": float(row["EMA200"]),
        "Volume": float(row["Volume"]),
        "AverageVolume20": float(row["AverageVolume20"]),
        "InPositionAfterSignal": bool(row["InPositionAfterSignal"]),
        "Equity": float(row["Equity"]),
        "ATR": float(row["ATR"]) if "ATR" in row else 0.0,
        "EntryPrice": float(row["EntryPrice"]) if "EntryPrice" in row else 0.0,
        "InitialStopPrice": float(row["InitialStopPrice"]) if "InitialStopPrice" in row else 0.0,
        "PendingOrderAction": str(row["PendingOrderAction"]) if "PendingOrderAction" in row else "",
    }


def print_latest_signal(label: str, signals_df: pd.DataFrame) -> dict:
    signal = latest_signal(signals_df)

    print("\nLatest Signal")
    print("-------------")
    print(f"Ticker/Data: {label}")
    for key, value in signal.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value}")

    return signal


def save_reports(label: str, equity_df: pd.DataFrame, trades_df: pd.DataFrame, signals_df: pd.DataFrame, output_dir: str = "reports") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_label = label.replace("/", "_").replace(" ", "_").upper()

    equity_file = output_path / f"{safe_label}_equity.csv"
    trades_file = output_path / f"{safe_label}_trades.csv"
    signals_file = output_path / f"{safe_label}_signals.csv"
    latest_signal_file = output_path / f"{safe_label}_latest_signal.csv"

    equity_df.to_csv(equity_file)
    trades_df.to_csv(trades_file)
    signals_df.to_csv(signals_file)

    latest = pd.DataFrame([latest_signal(signals_df)])
    latest.to_csv(latest_signal_file, index=False)

    print("\nSaved Reports")
    print("-------------")
    print(f"Equity curve: {equity_file}")
    print(f"Trades: {trades_file}")
    print(f"Signals: {signals_file}")
    print(f"Latest signal: {latest_signal_file}")
