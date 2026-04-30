from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def _normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    return df


def normalize_intraday_index(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], utc=True)
        frame = frame.set_index("Date")
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)

    return frame


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_intraday_index(df)

    if frame.empty:
        return frame

    resampled = frame.resample(
        "4h",
        origin=frame.index[0],
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    candle_counts = frame["Close"].resample("4h", origin=frame.index[0]).count()
    if not candle_counts.empty:
        expected_per_candle = int(candle_counts.max())
        resampled = resampled[candle_counts == expected_per_candle]

    resampled = resampled.dropna(subset=list(REQUIRED_COLUMNS))
    resampled = resampled.sort_index()

    return resampled


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input data is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    df = df.sort_index()

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(REQUIRED_COLUMNS))

    if len(df) < 220:
        raise ValueError("Need at least 220 rows because the strategy uses EMA200.")

    return df


def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_yfinance_data(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "yfinance is not installed. Install it with: pip install yfinance"
        ) from exc

    # Fresh download on every run.
    # No local cache is used here.
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df = _normalize_yfinance_columns(df)

    return df


def load_yfinance_intraday_data(
        ticker: str,
        interval: str = "1h",
        period: str = "730d",
        prepost: bool = False,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "yfinance is not installed. Install it with: pip install yfinance"
        ) from exc

    normalized_interval = "1h"
    if interval.lower() not in {"1h", "60m", "4h", "240m"}:
        normalized_interval = interval.lower()

    last_error: Exception | None = None

    for _ in range(2):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=normalized_interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                prepost=prepost,
            )
        except Exception as exc:
            last_error = exc
            df = pd.DataFrame()

        if not df.empty:
            df = _normalize_yfinance_columns(df)
            df = normalize_intraday_index(df)
            return df

    if last_error is not None:
        print(f"{ticker}: intraday fetch failed after retry: {last_error}")

    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def make_demo_data(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)

    drift = 0.00045
    volatility = 0.012
    returns = rng.normal(drift, volatility, rows)

    close = 100 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.003, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, rows))
    volume = rng.integers(1_000_000, 5_000_000, rows)
    volume[::35] = volume[::35] * 2

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
