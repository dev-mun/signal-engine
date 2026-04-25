from typing import Optional

import pandas as pd


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
        "exit_markers": close[signals["Signal"] == "EXIT_LONG"],
        "bearish_markers": close[signals["Signal"] == "BEARISH_ENTRY"],
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
        take_profit_pct: float = 0.20,
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
    price_ax.plot(price_df.index, price_df["EMA20"], label="EMA20", color="#e07a1f", linewidth=1.2)
    price_ax.plot(price_df.index, price_df["EMA50"], label="EMA50", color="#2a9d8f", linewidth=1.2)
    price_ax.plot(price_df.index, price_df["EMA200"], label="EMA200", color="#577590", linewidth=1.2)

    if not overlays["buy_markers"].empty:
        price_ax.scatter(
            overlays["buy_markers"].index,
            overlays["buy_markers"].values,
            label="BUY",
            color="#1b9e77",
            marker="^",
            s=90,
            zorder=5,
        )

    if not overlays["exit_markers"].empty:
        price_ax.scatter(
            overlays["exit_markers"].index,
            overlays["exit_markers"].values,
            label="EXIT_LONG",
            color="#ff7f11",
            marker="v",
            s=90,
            zorder=5,
        )

    if not overlays["bearish_markers"].empty:
        price_ax.scatter(
            overlays["bearish_markers"].index,
            overlays["bearish_markers"].values,
            label="BEARISH_ENTRY",
            color="#d62828",
            marker="v",
            s=90,
            zorder=5,
        )

    if not overlays["stop_line"].empty:
        price_ax.plot(
            overlays["stop_line"].index,
            overlays["stop_line"].values,
            label="Initial Stop",
            color="#c1121f",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )

    if not overlays["target_line"].empty:
        price_ax.plot(
            overlays["target_line"].index,
            overlays["target_line"].values,
            label="Target",
            color="#588157",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )

    price_ax.set_title(f"{label}: Signal Overlay")
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.25)
    price_ax.legend(loc="upper left", ncol=4)

    rsi_ax.plot(price_df.index, price_df["RSI"], color="#6a4c93", linewidth=1.2)
    rsi_ax.axhline(40, color="#adb5bd", linestyle="--", linewidth=1.0)
    rsi_ax.axhline(60, color="#adb5bd", linestyle="--", linewidth=1.0)
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
