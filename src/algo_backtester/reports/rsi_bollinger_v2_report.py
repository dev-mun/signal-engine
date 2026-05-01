from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from algo_backtester.metrics import performance_summary, print_summary


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
        "RSI14": float(row["RSI14"]),
        "EMA50": float(row["EMA50"]),
        "EMA200": float(row["EMA200"]),
        "BB_MIDDLE": float(row["BB_MIDDLE"]),
        "BB_UPPER": float(row["BB_UPPER"]),
        "BB_LOWER": float(row["BB_LOWER"]),
        "Volume": float(row["Volume"]),
        "AverageVolume20": float(row["AverageVolume20"]),
        "ClosePosition": float(row["ClosePosition"]),
        "InPositionAfterSignal": bool(row["InPositionAfterSignal"]),
        "Equity": float(row["Equity"]),
        "ATR": float(row["ATR"]),
        "ATR14": float(row["ATR14"]),
        "EntryPrice": float(row["EntryPrice"]),
        "InitialStopPrice": float(row["InitialStopPrice"]),
        "PendingOrderAction": str(row["PendingOrderAction"]),
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


def save_reports(
        label: str,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        output_dir: str = "reports",
) -> None:
    output_path = Path(output_dir) / "rsi_bollinger_v2"
    output_path.mkdir(parents=True, exist_ok=True)

    safe_label = label.replace("/", "_").replace(" ", "_").upper()

    equity_file = output_path / f"{safe_label}_equity.csv"
    trades_file = output_path / f"{safe_label}_trades.csv"
    signals_file = output_path / f"{safe_label}_signals.csv"
    latest_signal_file = output_path / f"{safe_label}_latest_signal.csv"

    equity_df.to_csv(equity_file)
    trades_df.to_csv(trades_file, index=False)
    signals_df.to_csv(signals_file)
    pd.DataFrame([latest_signal(signals_df)]).to_csv(latest_signal_file, index=False)

    print("\nSaved RSI Bollinger V2 Reports")
    print("------------------------------")
    print(f"Equity curve: {equity_file}")
    print(f"Trades: {trades_file}")
    print(f"Signals: {signals_file}")
    print(f"Latest signal: {latest_signal_file}")


def print_scan_results(results: list[dict]) -> None:
    actionable = [r for r in results if r["Signal"] in {"BUY", "SELL"}]
    errors = [r for r in results if r["Signal"] == "ERROR"]

    print("\nRSI Bollinger V2 Watchlist Scan")
    print("-------------------------------")
    print(
        f'{"Ticker":<8} '
        f'{"Strategy":<18} '
        f'{"Profile":<10} '
        f'{"Signal":<10} '
        f'{"Setup":<14} '
        f'{"Price":>10} '
        f'{"RSI":>8} '
        f'{"ATR":>8}  '
        f'{"Distance":<28} '
        f"Reason"
    )
    print("-" * 163)

    for result in results:
        if result["Signal"] == "ERROR":
            continue

        print(
            f'{result["Ticker"]:<8} '
            f'{result["Strategy"]:<18} '
            f'{result.get("Profile", "default"):<10} '
            f'{result["Signal"]:<10} '
            f'{result["Setup"]:<14} '
            f'{result["Price"]:>10.2f} '
            f'{result["RSI"]:>8.2f} '
            f'{result["ATR"]:>8.2f}  '
            f'{result["Distance"]:<28} '
            f'{result["Reason"]}'
        )

    print("\nActionable Signals")
    print("------------------")
    if not actionable:
        print("No actionable RSI Bollinger V2 signals.")
    else:
        for result in actionable:
            print(
                f'{result["Ticker"]}: {result["Signal"]} | '
                f'Price={result["Price"]:.2f} | '
                f'RSI={result["RSI"]:.2f} | '
                f'ATR={result["ATR"]:.2f} | '
                f'{result["Reason"]}'
            )

    if errors:
        print("\nErrors")
        print("------")
        for result in errors:
            print(f'{result["Ticker"]}: {result["Reason"]}')


def save_scan_results(results: list[dict], output_dir: str = "reports") -> None:
    output_path = Path(output_dir) / "rsi_bollinger_v2"
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        return

    date_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    file_path = output_path / f"watchlist_scan_{date_str}.csv"
    df.to_csv(file_path, index=False)

    print(f"\nSaved RSI Bollinger V2 Watchlist Scan: {file_path}")


def build_signal_overlay_data(
        price_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        take_profit_pct: float,
) -> dict[str, pd.Series]:
    aligned_index = price_df.index.intersection(signals_df.index)
    close = price_df.loc[aligned_index, "Close"]
    signals = signals_df.loc[aligned_index]

    in_position_mask = signals["InPositionAfterSignal"].astype(bool)
    entry_line = signals.loc[in_position_mask, "EntryPrice"]

    return {
        "buy_markers": close[signals["Signal"] == "BUY"],
        "sell_markers": close[signals["Signal"] == "SELL"],
        "entry_line": entry_line,
        "stop_line": signals.loc[in_position_mask, "InitialStopPrice"],
        "target_line": entry_line * (1 + take_profit_pct),
    }


def plot_results(
        label: str,
        price_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        bh_df: Optional[pd.DataFrame] = None,
        take_profit_pct: float = 0.05,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed. Skipping chart.")
        return

    overlays = build_signal_overlay_data(
        price_df=price_df,
        signals_df=signals_df,
        take_profit_pct=take_profit_pct,
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.5, 1.5, 1.5]},
    )
    price_ax, rsi_ax, volume_ax, equity_ax = axes

    price_ax.plot(price_df.index, price_df["Close"], label="Close", color="#1f1f1f", linewidth=1.6)
    price_ax.plot(price_df.index, price_df["EMA50"], label="EMA50", color="#2a9d8f", linewidth=1.2)
    price_ax.plot(price_df.index, price_df["EMA200"], label="EMA200", color="#577590", linewidth=1.2)
    price_ax.plot(price_df.index, price_df["BB_MIDDLE"], label="BB Middle", color="#e07a1f", linewidth=1.2)
    price_ax.plot(price_df.index, price_df["BB_UPPER"], label="BB Upper", color="#9c6644", linewidth=1.0, linestyle="--")
    price_ax.plot(price_df.index, price_df["BB_LOWER"], label="BB Lower", color="#9c6644", linewidth=1.0, linestyle="--")

    if not overlays["buy_markers"].empty:
        price_ax.scatter(overlays["buy_markers"].index, overlays["buy_markers"].values, label="BUY", color="#1b9e77", marker="^", s=90, zorder=5)

    if not overlays["sell_markers"].empty:
        price_ax.scatter(overlays["sell_markers"].index, overlays["sell_markers"].values, label="SELL", color="#d62828", marker="v", s=90, zorder=5)

    if not overlays["stop_line"].empty:
        price_ax.plot(overlays["stop_line"].index, overlays["stop_line"].values, label="Initial Stop", color="#c1121f", linestyle="--", linewidth=1.0, alpha=0.75)

    if not overlays["target_line"].empty:
        price_ax.plot(overlays["target_line"].index, overlays["target_line"].values, label="Target", color="#588157", linestyle="--", linewidth=1.0, alpha=0.75)

    price_ax.set_title(f"{label}: RSI Bollinger Mean Reversion V2")
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.25)
    price_ax.legend(loc="upper left", ncol=4)

    rsi_ax.plot(price_df.index, price_df["RSI"], color="#6a4c93", linewidth=1.2)
    rsi_ax.axhline(38, color="#adb5bd", linestyle="--", linewidth=1.0)
    rsi_ax.axhline(50, color="#adb5bd", linestyle="--", linewidth=1.0)
    rsi_ax.set_ylabel("RSI")
    rsi_ax.set_ylim(0, 100)
    rsi_ax.grid(True, alpha=0.2)

    volume_ax.bar(price_df.index, price_df["Volume"], color="#90caf9", alpha=0.7, label="Volume")
    volume_ax.plot(price_df.index, price_df["AVG_VOL20"], color="#1565c0", linewidth=1.2, label="AVG_VOL20")
    volume_ax.set_ylabel("Volume")
    volume_ax.grid(True, alpha=0.2)
    volume_ax.legend(loc="upper left")

    equity_ax.plot(equity_df.index, equity_df["Equity"], label="Strategy", color="#264653", linewidth=1.4)
    if bh_df is not None and not bh_df.empty:
        equity_ax.plot(bh_df.index, bh_df["BuyHold"], label="Buy & Hold", color="#8d99ae", linewidth=1.2)

    equity_ax.set_title("Equity Curve")
    equity_ax.set_ylabel("Equity")
    equity_ax.set_xlabel("Date")
    equity_ax.grid(True, alpha=0.25)
    equity_ax.legend(loc="upper left")

    fig.tight_layout()
    plt.show()
