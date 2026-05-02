from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd
from pandas.tseries.offsets import BDay

from algo_backtester.backtests.ema_rsi_backtester import EmaRsiBacktestConfig, EmaRsiPullbackBacktester
from algo_backtester.backtests.four_hour_trend_backtester import (
    FourHourTrendBacktester,
    FourHourTrendConfig,
    prepare_four_hour_data,
)
from algo_backtester.data_loader import load_yfinance_data
from algo_backtester.options_utils import LongCallPlan, build_long_call_plan
from algo_backtester.strategies.ema_rsi_pullback import classify_setup as ema_rsi_classify_setup
from algo_backtester.strategies.four_hour_trend_pullback import classify_setup as four_hour_classify_setup
from algo_backtester.strategies.options_momentum import SourceSignalSnapshot, build_snapshot


@dataclass(frozen=True)
class OptionsMomentumConfig:
    initial_cash: float = 10_000.0
    risk_per_trade: float = 0.015
    max_contracts: int = 1
    min_dte: int = 14
    max_dte: int = 30
    interval: str = "1h"


def _planned_execution_date(signal_date: str) -> str:
    return str((pd.Timestamp(signal_date) + BDay(1)).date())


def _ema_rsi_snapshot(
    ticker: str,
    start: str,
    end: str | None,
) -> SourceSignalSnapshot:
    raw_df = load_yfinance_data(ticker=ticker, start=start, end=end)
    bt = EmaRsiPullbackBacktester(config=EmaRsiBacktestConfig())
    _, _, _, signals_df = bt.run(raw_df)
    latest = signals_df.iloc[-1]
    signal = str(latest["Signal"])
    return build_snapshot(
        ticker=ticker,
        source_strategy="ema-rsi",
        latest_row=latest,
        signal=signal,
        setup=ema_rsi_classify_setup(signal, float(latest["RSI"])),
        signal_date=str(signals_df.index[-1].date()),
        reason=str(latest["Reason"]),
    )


def _four_hour_snapshot(
    ticker: str,
    interval: str,
) -> SourceSignalSnapshot:
    raw_df = prepare_four_hour_data(ticker=ticker, interval=interval)
    if raw_df.empty:
        raise ValueError(f"No usable intraday data found for ticker: {ticker}")
    bt = FourHourTrendBacktester(config=FourHourTrendConfig(interval=interval))
    _, _, _, signals_df = bt.run(raw_df)
    latest = signals_df.iloc[-1]
    signal = str(latest["Signal"])
    return build_snapshot(
        ticker=ticker,
        source_strategy="four-hour-trend",
        latest_row=latest,
        signal=signal,
        setup=four_hour_classify_setup(signal, latest),
        signal_date=str(signals_df.index[-1]),
        reason=str(latest["Reason"]),
    )


def evaluate_source_strategies(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    interval: str = "1h",
) -> list[SourceSignalSnapshot]:
    return [
        _ema_rsi_snapshot(ticker=ticker, start=start, end=end),
        _four_hour_snapshot(ticker=ticker, interval=interval),
    ]


def _select_best_snapshot(snapshots: list[SourceSignalSnapshot]) -> SourceSignalSnapshot:
    qualified = [snapshot for snapshot in snapshots if snapshot.qualified]
    if qualified:
        return sorted(qualified, key=lambda snapshot: (snapshot.score, snapshot.source_strategy), reverse=True)[0]
    return sorted(snapshots, key=lambda snapshot: (snapshot.score, snapshot.source_strategy), reverse=True)[0]


def _journal_fields(result: dict, plan: LongCallPlan | None) -> dict:
    if plan is None or result["Signal"] != "BUY":
        return {
            "OptionsAction": "NO_OPTIONS_TRADE",
            "Structure": "No trade",
            "Expiration": "N/A",
            "LongStrike": 0.0,
            "ShortStrike": 0.0,
            "EstimatedDebit": 0.0,
            "MaxLoss": 0.0,
            "MaxProfit": "",
            "TradeQuality": "NO_TRADE",
            "PlannedEntryReference": "",
            "StopLoss": "",
            "TakeProfit": "",
            "RiskPerShare": "",
            "RewardPerShare": "",
        }

    return {
        "OptionsAction": "PLAN_LONG_CALL",
        "Structure": f"Long Call {plan.strike:.2f}C",
        "Expiration": plan.expiration,
        "LongStrike": plan.strike,
        "ShortStrike": 0.0,
        "EstimatedDebit": plan.estimated_premium,
        "MaxLoss": plan.max_loss,
        "MaxProfit": "",
        "TradeQuality": "VALID",
        "PlannedEntryReference": result["Price"],
        "StopLoss": plan.stop_loss,
        "TakeProfit": plan.exit3,
        "RiskPerShare": round(plan.estimated_premium - plan.stop_loss, 2),
        "RewardPerShare": round(plan.exit3 - plan.estimated_premium, 2),
    }


def _build_result(
    ticker: str,
    selected_snapshot: SourceSignalSnapshot,
    plan: LongCallPlan | None,
) -> dict:
    signal = "BUY" if selected_snapshot.qualified and plan is not None else "HOLD"
    notes = selected_snapshot.notes if plan is None else f"{selected_snapshot.notes} | {plan.notes}"
    planned_execution_date = _planned_execution_date(selected_snapshot.signal_date)

    result = {
        "Ticker": ticker,
        "Strategy": "options-momentum",
        "SourceStrategy": selected_snapshot.source_strategy,
        "Signal": signal,
        "Setup": "ACTIONABLE" if signal == "BUY" else selected_snapshot.setup,
        "OptionType": "CALL",
        "Strike": 0.0 if plan is None else plan.strike,
        "DTE": 0 if plan is None else plan.dte,
        "DeltaTarget": 0.0 if plan is None else plan.delta_target,
        "EstPremium": 0.0 if plan is None else plan.estimated_premium,
        "MaxLoss": 0.0 if plan is None else plan.max_loss,
        "Exit1": 0.0 if plan is None else plan.exit1,
        "Exit2": 0.0 if plan is None else plan.exit2,
        "Exit3": 0.0 if plan is None else plan.exit3,
        "Liquidity": "N/A" if plan is None else plan.liquidity,
        "Notes": notes,
        "Direction": "BULLISH",
        "Price": selected_snapshot.price,
        "RSI": selected_snapshot.rsi,
        "ATR": selected_snapshot.atr,
        "Reason": selected_snapshot.reason,
        "SignalDate": selected_snapshot.signal_date,
        "PlannedExecutionDate": planned_execution_date,
        "UniverseStatus": "ELIGIBLE" if signal == "BUY" else "WATCH",
        "UniverseReason": notes,
        "SetupStatus": "ACTIONABLE" if signal == "BUY" else selected_snapshot.setup,
        "DistanceToSetup": "Actionable now" if signal == "BUY" else selected_snapshot.notes,
        "Equity": 0.0,
        "OptionsReason": notes,
        "AvgDollarVolume": 0.0,
        "EarningsDate": "PLACEHOLDER_OK",
        "SourceSignal": selected_snapshot.signal,
    }
    result.update(_journal_fields(result=result, plan=plan))
    return result


def analyze_ticker(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    config: OptionsMomentumConfig | None = None,
) -> dict:
    effective_config = config or OptionsMomentumConfig()
    snapshots = evaluate_source_strategies(
        ticker=ticker,
        start=start,
        end=end,
        interval=effective_config.interval,
    )
    selected_snapshot = _select_best_snapshot(snapshots)
    plan = (
        build_long_call_plan(
            ticker=ticker,
            source_strategy=selected_snapshot.source_strategy,
            price=selected_snapshot.price,
            atr=selected_snapshot.atr,
            signal_date=selected_snapshot.signal_date,
        )
        if selected_snapshot.qualified
        else None
    )

    return {
        "result": _build_result(ticker=ticker, selected_snapshot=selected_snapshot, plan=plan),
        "sources": [asdict(snapshot) for snapshot in snapshots],
    }


def scan_ticker(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    config: OptionsMomentumConfig | None = None,
) -> dict:
    return analyze_ticker(ticker=ticker, start=start, end=end, config=config)["result"]


def scan_watchlist(
    tickers: Iterable[str],
    start: str = "2018-01-01",
    end: str | None = None,
    config: OptionsMomentumConfig | None = None,
) -> list[dict]:
    results = []

    for ticker in tickers:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            continue

        try:
            results.append(scan_ticker(ticker=clean_ticker, start=start, end=end, config=config))
        except Exception as exc:
            results.append(
                {
                    "Ticker": clean_ticker,
                    "Strategy": "options-momentum",
                    "SourceStrategy": "ERROR",
                    "Signal": "ERROR",
                    "Setup": "ERROR",
                    "OptionType": "CALL",
                    "Strike": 0.0,
                    "DTE": 0,
                    "DeltaTarget": 0.0,
                    "EstPremium": 0.0,
                    "MaxLoss": 0.0,
                    "Exit1": 0.0,
                    "Exit2": 0.0,
                    "Exit3": 0.0,
                    "Liquidity": "N/A",
                    "Notes": str(exc),
                    "Direction": "BULLISH",
                    "Price": 0.0,
                    "RSI": 0.0,
                    "ATR": 0.0,
                    "Reason": str(exc),
                    "SignalDate": str(pd.Timestamp.today().date()),
                    "PlannedExecutionDate": str((pd.Timestamp.today() + BDay(1)).date()),
                    "UniverseStatus": "ERROR",
                    "UniverseReason": str(exc),
                    "SetupStatus": "ERROR",
                    "DistanceToSetup": "ERROR",
                    "Equity": 0.0,
                    "OptionsReason": str(exc),
                    "AvgDollarVolume": 0.0,
                    "EarningsDate": "UNKNOWN",
                    "SourceSignal": "ERROR",
                    "OptionsAction": "ERROR",
                    "Structure": "N/A",
                    "Expiration": "N/A",
                    "LongStrike": 0.0,
                    "ShortStrike": 0.0,
                    "EstimatedDebit": 0.0,
                    "MaxProfit": "",
                    "TradeQuality": "ERROR",
                    "PlannedEntryReference": "",
                    "StopLoss": "",
                    "TakeProfit": "",
                    "RiskPerShare": "",
                    "RewardPerShare": "",
                }
            )

    return results
