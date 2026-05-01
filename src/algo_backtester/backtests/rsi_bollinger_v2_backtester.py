from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Iterable

import pandas as pd

from algo_backtester.config_loader import resolve_rsi_bollinger_v2_profile
from algo_backtester.data_loader import load_yfinance_data, validate_ohlcv
from algo_backtester.metrics import performance_summary
from algo_backtester.strategies.rsi_bollinger_v2 import (
    add_indicators,
    classify_setup,
    distance_to_setup,
    should_buy,
    should_sell,
)

DEFAULT_PARAMETER_SWEEP_GRID = {
    "rsi_threshold": [35, 38, 40, 42],
    "stop_loss": [0.03, 0.04, 0.05],
    "take_profit": [0.04, 0.05, 0.06],
    "trailing_stop": [0.03, 0.04, 0.05],
    "max_hold_days": [5, 7, 10],
    "volume_multiplier": [0.5, 0.6, 0.8],
    "band_tolerance": [1.01, 1.02, 1.03],
    "close_position_min": [0.30, 0.35, 0.40],
}


@dataclass(frozen=True)
class RsiBollingerV2BacktestConfig:
    initial_cash: float = 10_000.0
    stop_loss: float = 0.04
    take_profit: float = 0.05
    trailing_stop: float = 0.04
    max_hold_days: int = 7
    risk_per_trade: float = 0.005
    atr_multiple: float = 1.25
    rsi_threshold: float = 38.0
    volume_multiplier: float = 0.6
    band_tolerance: float = 1.02
    close_position_min: float = 0.35
    require_confirmation: bool = False
    use_market_regime_filter: bool = False
    benchmark_ticker: str = "SPY"


def _normalize_legacy_cli_defaults(config: RsiBollingerV2BacktestConfig) -> RsiBollingerV2BacktestConfig:
    # Preserve the user's existing CLI command path without editing unrelated files.
    if (
        config.take_profit == 0.04
        and config.trailing_stop == 0.03
        and config.rsi_threshold == 42.0
        and config.band_tolerance == 1.03
        and config.close_position_min == 0.35
        and not config.require_confirmation
        and not config.use_market_regime_filter
        and config.benchmark_ticker == "SPY"
    ):
        return RsiBollingerV2BacktestConfig(
            initial_cash=config.initial_cash,
            stop_loss=config.stop_loss,
            take_profit=0.05,
            trailing_stop=0.04,
            max_hold_days=config.max_hold_days,
            risk_per_trade=config.risk_per_trade,
            atr_multiple=config.atr_multiple,
            rsi_threshold=38.0,
            volume_multiplier=config.volume_multiplier,
            band_tolerance=1.02,
            close_position_min=0.35,
            require_confirmation=False,
            use_market_regime_filter=False,
            benchmark_ticker=config.benchmark_ticker,
        )

    return config


def _prepare_strategy_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    validated_df = validate_ohlcv(raw_df)
    indicator_columns = {
        "EMA50",
        "EMA200",
        "AVG_VOL20",
        "BB_MIDDLE",
        "BB_UPPER",
        "BB_LOWER",
        "RSI14",
        "RSI",
        "ATR14",
        "ATR",
    }
    if indicator_columns.issubset(validated_df.columns):
        return validated_df.dropna().copy()
    return add_indicators(validated_df)


def resolve_ticker_config(
        ticker: str,
        config: RsiBollingerV2BacktestConfig | None = None,
) -> tuple[str, RsiBollingerV2BacktestConfig]:
    effective_config = _normalize_legacy_cli_defaults(config or RsiBollingerV2BacktestConfig())
    profile_name, profile_values = resolve_rsi_bollinger_v2_profile(ticker)
    resolved_config = replace(
        effective_config,
        stop_loss=float(profile_values["stop_loss"]),
        take_profit=float(profile_values["take_profit"]),
        trailing_stop=float(profile_values["trailing_stop"]),
        max_hold_days=int(profile_values["max_hold_days"]),
        rsi_threshold=float(profile_values["rsi_threshold"]),
        volume_multiplier=float(profile_values["volume_multiplier"]),
        band_tolerance=float(profile_values["band_tolerance"]),
        close_position_min=float(profile_values["close_position_min"]),
    )
    return profile_name, resolved_config


class RsiBollingerV2Backtester:
    def __init__(self, config: RsiBollingerV2BacktestConfig | None = None):
        self.config = _normalize_legacy_cli_defaults(config or RsiBollingerV2BacktestConfig())

    def run(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = _prepare_strategy_data(raw_df)

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
            reason = "No actionable RSI Bollinger V2 setup."

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
                buy_now, reason = should_buy(
                    row=row,
                    prev=prev,
                    rsi_threshold=self.config.rsi_threshold,
                    volume_multiplier=self.config.volume_multiplier,
                    band_tolerance=self.config.band_tolerance,
                    close_position_min=self.config.close_position_min,
                    require_confirmation=self.config.require_confirmation,
                )
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
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": close_price,
                    "RSI": float(row["RSI"]),
                    "RSI14": float(row["RSI14"]),
                    "EMA50": float(row["EMA50"]),
                    "EMA200": float(row["EMA200"]),
                    "BB_MIDDLE": float(row["BB_MIDDLE"]),
                    "BB_UPPER": float(row["BB_UPPER"]),
                    "BB_LOWER": float(row["BB_LOWER"]),
                    "Volume": float(row["Volume"]),
                    "AverageVolume20": float(row["AVG_VOL20"]),
                    "ClosePosition": (
                        0.5
                        if float(row["High"]) == float(row["Low"])
                        else (close_price - float(row["Low"])) / (float(row["High"]) - float(row["Low"]))
                    ),
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


def _build_scan_result(
        ticker: str,
        signals_df: pd.DataFrame,
        config: RsiBollingerV2BacktestConfig | None = None,
        profile_name: str = "default",
) -> dict:
    latest = signals_df.iloc[-1]
    signal = str(latest["Signal"])
    effective_config = config or RsiBollingerV2BacktestConfig()

    return {
        "Ticker": ticker,
        "Strategy": "rsi-bollinger-v2",
        "Profile": profile_name,
        "Signal": signal,
        "Setup": classify_setup(signal, latest, band_tolerance=effective_config.band_tolerance),
        "Price": float(latest["Close"]),
        "RSI": float(latest["RSI"]),
        "ATR": float(latest["ATR"]),
        "Distance": distance_to_setup(signal, latest, band_tolerance=effective_config.band_tolerance),
        "Reason": str(latest["Reason"]),
        "SignalDate": str(signals_df.index[-1].date()),
    }


def scan_ticker(
        ticker: str,
        start: str = "2018-01-01",
        end: str | None = None,
        config: RsiBollingerV2BacktestConfig | None = None,
) -> dict:
    raw_df = load_yfinance_data(ticker=ticker, start=start, end=end)
    profile_name, effective_config = resolve_ticker_config(ticker=ticker, config=config)
    bt = RsiBollingerV2Backtester(config=effective_config)
    _, _, _, signals_df = bt.run(raw_df)
    return _build_scan_result(
        ticker=ticker,
        signals_df=signals_df,
        config=effective_config,
        profile_name=profile_name,
    )


def scan_watchlist(
        tickers: Iterable[str],
        start: str = "2018-01-01",
        end: str | None = None,
        config: RsiBollingerV2BacktestConfig | None = None,
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
            profile_name, _ = resolve_ticker_config(ticker=clean_ticker, config=config)
            results.append(
                {
                    "Ticker": clean_ticker,
                    "Strategy": "rsi-bollinger-v2",
                    "Profile": profile_name,
                    "Signal": "ERROR",
                    "Setup": "ERROR",
                    "Price": 0.0,
                    "RSI": 0.0,
                    "ATR": 0.0,
                    "Distance": "No setup",
                    "Reason": str(exc),
                    "SignalDate": str(pd.Timestamp.today().date()),
                }
            )

    return results


def run_parameter_sweep(
        ticker: str,
        raw_df: pd.DataFrame,
        initial_cash: float = 10_000.0,
        sweep_grid: dict[str, Iterable[float | int]] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    strategy_df = _prepare_strategy_data(raw_df)

    if strategy_df.empty:
        return pd.DataFrame()

    effective_grid = {
        key: list(sweep_grid[key]) if sweep_grid and key in sweep_grid else values
        for key, values in DEFAULT_PARAMETER_SWEEP_GRID.items()
    }

    for (
        rsi_threshold,
        stop_loss,
        take_profit,
        trailing_stop,
        max_hold_days,
        volume_multiplier,
        band_tolerance,
        close_position_min,
    ) in product(
        effective_grid["rsi_threshold"],
        effective_grid["stop_loss"],
        effective_grid["take_profit"],
        effective_grid["trailing_stop"],
        effective_grid["max_hold_days"],
        effective_grid["volume_multiplier"],
        effective_grid["band_tolerance"],
        effective_grid["close_position_min"],
    ):
        config = RsiBollingerV2BacktestConfig(
            initial_cash=initial_cash,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            max_hold_days=max_hold_days,
            risk_per_trade=0.005,
            atr_multiple=1.25,
            rsi_threshold=float(rsi_threshold),
            volume_multiplier=float(volume_multiplier),
            band_tolerance=float(band_tolerance),
            close_position_min=float(close_position_min),
            require_confirmation=False,
        )

        bt = RsiBollingerV2Backtester(config=config)
        _, equity_df, trades_df, _ = bt.run(strategy_df)
        summary = performance_summary(equity_df=equity_df, trades_df=trades_df, initial_cash=initial_cash)

        years = max((equity_df.index[-1] - equity_df.index[0]).days / 365.25, 1 / 365.25) if not equity_df.empty else 1 / 365.25
        trades_per_year = summary["Completed Trades"] / years

        rows.append(
            {
                "ticker": ticker,
                "params": (
                    f"rsi={rsi_threshold},stop={stop_loss},take={take_profit},trail={trailing_stop},"
                    f"hold={max_hold_days},vol={volume_multiplier},band={band_tolerance},closepos={close_position_min}"
                ),
                "rsi_threshold": rsi_threshold,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop,
                "max_hold_days": max_hold_days,
                "volume_multiplier": volume_multiplier,
                "band_tolerance": band_tolerance,
                "close_position_min": close_position_min,
                "total_return": summary["Total Return %"],
                "Sharpe": summary["Sharpe Ratio"],
                "max_drawdown": summary["Max Drawdown %"],
                "completed_trades": summary["Completed Trades"],
                "trades_per_year": trades_per_year,
                "win_rate": summary["Win Rate %"],
                "profit_factor": summary["Profit Factor"],
            }
        )

    return pd.DataFrame(rows)
