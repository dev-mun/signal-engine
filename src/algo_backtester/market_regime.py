from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from algo_backtester.data_loader import load_yfinance_data

VIX_TICKER = "^VIX"


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    market_regime: str
    regime_reason: str
    spy_close: float
    spy_sma20: float
    spy_sma50: float
    qqq_close: float
    qqq_sma20: float
    qqq_sma50: float
    vix_close: float
    vix_trend: str

    def to_dict(self) -> dict:
        return asdict(self)


def _prepare_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame


def _vix_trend_label(vix_values: pd.Series) -> str:
    if len(vix_values) < 3:
        return "FLAT"

    current = float(vix_values.iloc[-1])
    prior_1 = float(vix_values.iloc[-2])
    prior_2 = float(vix_values.iloc[-3])

    if current <= max(prior_1, prior_2) * 1.01 and current <= prior_2 * 1.01:
        return "FLAT_OR_DECLINING"
    if current >= min(prior_1, prior_2) * 1.01 and current >= prior_2 * 1.02 and current > prior_1:
        return "RISING"
    return "FLAT"


def _classify_row(row: pd.Series) -> tuple[str, str]:
    spy_above = bool(row["SPY_Close"] > row["SPY_SMA20"] and row["SPY_Close"] > row["SPY_SMA50"])
    qqq_above = bool(row["QQQ_Close"] > row["QQQ_SMA20"] and row["QQQ_Close"] > row["QQQ_SMA50"])
    spy_below = bool(row["SPY_Close"] < row["SPY_SMA20"] and row["SPY_Close"] < row["SPY_SMA50"])
    qqq_below = bool(row["QQQ_Close"] < row["QQQ_SMA20"] and row["QQQ_Close"] < row["QQQ_SMA50"])
    vix_trend = str(row["VIX_Trend"])

    if spy_above and qqq_above and vix_trend == "FLAT_OR_DECLINING":
        return (
            "BULLISH",
            "SPY and QQQ are above their 20/50 SMAs while VIX is flat to lower over the last 3 sessions.",
        )
    if spy_below and qqq_below and vix_trend == "RISING":
        return (
            "BEARISH",
            "SPY and QQQ are below their 20/50 SMAs while VIX is rising over the last 3 sessions.",
        )
    return (
        "MIXED",
        "SPY, QQQ, and VIX are not aligned enough to confirm a clear bullish or bearish regime.",
    )


def build_market_regime_history(start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    spy_df = _prepare_daily_frame(load_yfinance_data("SPY", start=start, end=end))
    qqq_df = _prepare_daily_frame(load_yfinance_data("QQQ", start=start, end=end))
    vix_df = _prepare_daily_frame(load_yfinance_data(VIX_TICKER, start=start, end=end))

    regime_df = pd.DataFrame(
        {
            "SPY_Close": spy_df["Close"],
            "QQQ_Close": qqq_df["Close"],
            "VIX_Close": vix_df["Close"],
        }
    ).sort_index()
    regime_df["SPY_SMA20"] = regime_df["SPY_Close"].rolling(20).mean()
    regime_df["SPY_SMA50"] = regime_df["SPY_Close"].rolling(50).mean()
    regime_df["QQQ_SMA20"] = regime_df["QQQ_Close"].rolling(20).mean()
    regime_df["QQQ_SMA50"] = regime_df["QQQ_Close"].rolling(50).mean()
    regime_df["VIX_Trend"] = regime_df["VIX_Close"].rolling(3).apply(
        lambda window: 1.0
        if _vix_trend_label(pd.Series(window)) == "RISING"
        else -1.0
        if _vix_trend_label(pd.Series(window)) == "FLAT_OR_DECLINING"
        else 0.0,
        raw=False,
    )
    regime_df["VIX_Trend"] = regime_df["VIX_Trend"].map({1.0: "RISING", -1.0: "FLAT_OR_DECLINING", 0.0: "FLAT"}).fillna("FLAT")
    regime_df = regime_df.dropna(subset=["SPY_SMA20", "SPY_SMA50", "QQQ_SMA20", "QQQ_SMA50", "VIX_Close"])

    regime_labels = regime_df.apply(_classify_row, axis=1)
    regime_df["market_regime"] = regime_labels.map(lambda value: value[0])
    regime_df["regime_reason"] = regime_labels.map(lambda value: value[1])

    return regime_df


def analyze_market_regime(start: str = "2018-01-01", end: str | None = None) -> MarketRegimeSnapshot:
    history = build_market_regime_history(start=start, end=end)
    latest = history.iloc[-1]
    return MarketRegimeSnapshot(
        market_regime=str(latest["market_regime"]),
        regime_reason=str(latest["regime_reason"]),
        spy_close=float(latest["SPY_Close"]),
        spy_sma20=float(latest["SPY_SMA20"]),
        spy_sma50=float(latest["SPY_SMA50"]),
        qqq_close=float(latest["QQQ_Close"]),
        qqq_sma20=float(latest["QQQ_SMA20"]),
        qqq_sma50=float(latest["QQQ_SMA50"]),
        vix_close=float(latest["VIX_Close"]),
        vix_trend=str(latest["VIX_Trend"]),
    )
