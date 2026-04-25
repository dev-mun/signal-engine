from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class OptionsRecommendation:
    stock_signal: str
    options_action: str
    structure: str
    trade_quality: str
    ticker: str
    expiration: str
    dte: int
    long_strike: float
    short_strike: float
    estimated_debit: float
    max_loss: float
    max_profit: float
    breakeven: float
    reward_risk: float
    iv_rank_proxy: float
    iv_status: str
    max_risk_dollars: float
    estimated_contracts: int
    reason: str


def _mid_price(bid: float, ask: float, last_price: float) -> float:
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2, 2)

    if last_price > 0:
        return round(last_price, 2)

    return 0.0


def _days_to_expiration(expiration: str) -> int:
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    today = datetime.today().date()
    return max((exp_date - today).days, 0)


def _select_expiration(ticker_obj: yf.Ticker, min_dte: int = 30, max_dte: int = 45) -> Optional[str]:
    expirations = ticker_obj.options

    if not expirations:
        return None

    candidates = []

    for exp in expirations:
        dte = _days_to_expiration(exp)
        if min_dte <= dte <= max_dte:
            candidates.append((exp, dte))

    if candidates:
        return sorted(candidates, key=lambda x: abs(x[1] - 35))[0][0]

    fallback = [(exp, _days_to_expiration(exp)) for exp in expirations if _days_to_expiration(exp) > 20]

    if not fallback:
        return None

    return sorted(fallback, key=lambda x: x[1])[0][0]


def _estimate_iv_rank_proxy(ticker: str, current_iv: float) -> tuple[float, str]:
    """
    yfinance does not provide true 52-week historical IV.
    This uses realized volatility as a conservative proxy.

    True IV Rank requires historical implied volatility data from a provider like:
    - OptionMetrics
    - ORATS
    - CBOE DataShop
    - Polygon
    - Tradier
    - Interactive Brokers
    """
    hist = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

    if hist.empty or "Close" not in hist.columns:
        return 50.0, "UNKNOWN_IV_ENV"

    returns = hist["Close"].pct_change().dropna()
    rolling_rv = returns.rolling(20).std() * np.sqrt(252)

    rolling_rv = rolling_rv.dropna()

    if rolling_rv.empty:
        return 50.0, "UNKNOWN_IV_ENV"

    low = float(rolling_rv.min())
    high = float(rolling_rv.max())

    if high == low:
        return 50.0, "UNKNOWN_IV_ENV"

    iv_rank = (current_iv - low) / (high - low) * 100
    iv_rank = max(0.0, min(100.0, iv_rank))

    if iv_rank > 70:
        status = "EXPENSIVE_PREMIUM"
    elif iv_rank < 30:
        status = "CHEAP_PREMIUM"
    else:
        status = "NORMAL_PREMIUM"

    return round(iv_rank, 2), status


def _nearest_delta_contract(chain: pd.DataFrame, target_delta: float, fallback_strike: float) -> pd.Series:
    df = chain.copy()

    if "delta" in df.columns and df["delta"].notna().any():
        df["delta_distance"] = (df["delta"].abs() - target_delta).abs()
        return df.sort_values("delta_distance").iloc[0]

    df["strike_distance"] = (df["strike"] - fallback_strike).abs()
    return df.sort_values("strike_distance").iloc[0]


def _build_call_debit_spread(
        ticker: str,
        price: float,
        account_equity: float,
        risk_per_trade: float,
        min_dte: int = 30,
        max_dte: int = 45,
) -> OptionsRecommendation:
    ticker_obj = yf.Ticker(ticker)
    expiration = _select_expiration(ticker_obj, min_dte=min_dte, max_dte=max_dte)

    if expiration is None:
        return _no_trade("BUY", ticker, "No valid options expiration found.")

    chain = ticker_obj.option_chain(expiration)
    calls = chain.calls.copy()

    if calls.empty:
        return _no_trade("BUY", ticker, "No call options found.")

    calls = calls[calls["strike"] >= price * 0.95].copy()

    if calls.empty:
        return _no_trade("BUY", ticker, "No usable call strikes found.")

    long_leg = _nearest_delta_contract(
        chain=calls,
        target_delta=0.55,
        fallback_strike=price,
    )

    long_strike = float(long_leg["strike"])

    short_candidates = calls[calls["strike"] > long_strike].copy()

    if short_candidates.empty:
        return _no_trade("BUY", ticker, "No valid short call strike found.")

    if "delta" in short_candidates.columns and short_candidates["delta"].notna().any():
        short_leg = _nearest_delta_contract(
            chain=short_candidates,
            target_delta=0.30,
            fallback_strike=long_strike * 1.05,
        )
    else:
        short_candidates["strike_distance"] = (short_candidates["strike"] - (long_strike * 1.05)).abs()
        short_leg = short_candidates.sort_values("strike_distance").iloc[0]

    short_strike = float(short_leg["strike"])

    long_mid = _mid_price(
        float(long_leg.get("bid", 0.0)),
        float(long_leg.get("ask", 0.0)),
        float(long_leg.get("lastPrice", 0.0)),
    )

    short_mid = _mid_price(
        float(short_leg.get("bid", 0.0)),
        float(short_leg.get("ask", 0.0)),
        float(short_leg.get("lastPrice", 0.0)),
    )

    debit = round(long_mid - short_mid, 2)

    if debit <= 0:
        return _no_trade("BUY", ticker, "Invalid debit calculation. Option quotes may be stale.")

    spread_width = short_strike - long_strike
    max_loss = debit * 100
    max_profit = (spread_width - debit) * 100
    breakeven = long_strike + debit
    reward_risk = max_profit / max_loss if max_loss > 0 else 0.0

    current_iv = float(long_leg.get("impliedVolatility", 0.0))
    iv_rank_proxy, iv_status = _estimate_iv_rank_proxy(ticker, current_iv)

    max_risk_dollars = account_equity * risk_per_trade
    estimated_contracts = int(max_risk_dollars // max_loss) if max_loss > 0 else 0

    if estimated_contracts < 1:
        quality = "REJECT_TOO_EXPENSIVE"
        reason = "Spread max loss exceeds allowed risk budget."
    elif iv_rank_proxy > 70:
        quality = "REJECT_IV_TOO_HIGH"
        reason = "Bullish signal exists, but premium is expensive. Avoid opening debit spread."
    elif reward_risk < 1.0:
        quality = "REJECT_POOR_REWARD_RISK"
        reason = "Reward/risk is below 1.0."
    else:
        quality = "VALID"
        reason = "Bullish signal with acceptable spread economics."

    return OptionsRecommendation(
        stock_signal="BUY",
        options_action="CONSIDER_BULLISH_OPTIONS_TRADE",
        structure="Call Debit Spread",
        trade_quality=quality,
        ticker=ticker,
        expiration=expiration,
        dte=_days_to_expiration(expiration),
        long_strike=long_strike,
        short_strike=short_strike,
        estimated_debit=debit,
        max_loss=round(max_loss, 2),
        max_profit=round(max_profit, 2),
        breakeven=round(breakeven, 2),
        reward_risk=round(reward_risk, 2),
        iv_rank_proxy=iv_rank_proxy,
        iv_status=iv_status,
        max_risk_dollars=round(max_risk_dollars, 2),
        estimated_contracts=estimated_contracts,
        reason=reason,
    )


def _build_put_debit_spread(
        ticker: str,
        price: float,
        account_equity: float,
        risk_per_trade: float,
        min_dte: int = 30,
        max_dte: int = 45,
) -> OptionsRecommendation:
    ticker_obj = yf.Ticker(ticker)
    expiration = _select_expiration(ticker_obj, min_dte=min_dte, max_dte=max_dte)

    if expiration is None:
        return _no_trade("SELL", ticker, "No valid options expiration found.")

    chain = ticker_obj.option_chain(expiration)
    puts = chain.puts.copy()

    if puts.empty:
        return _no_trade("SELL", ticker, "No put options found.")

    puts = puts[puts["strike"] <= price * 1.05].copy()

    if puts.empty:
        return _no_trade("SELL", ticker, "No usable put strikes found.")

    long_leg = _nearest_delta_contract(
        chain=puts,
        target_delta=0.55,
        fallback_strike=price,
    )

    long_strike = float(long_leg["strike"])

    short_candidates = puts[puts["strike"] < long_strike].copy()

    if short_candidates.empty:
        return _no_trade("SELL", ticker, "No valid short put strike found.")

    if "delta" in short_candidates.columns and short_candidates["delta"].notna().any():
        short_leg = _nearest_delta_contract(
            chain=short_candidates,
            target_delta=0.30,
            fallback_strike=long_strike * 0.95,
        )
    else:
        short_candidates["strike_distance"] = (short_candidates["strike"] - (long_strike * 0.95)).abs()
        short_leg = short_candidates.sort_values("strike_distance").iloc[0]

    short_strike = float(short_leg["strike"])

    long_mid = _mid_price(
        float(long_leg.get("bid", 0.0)),
        float(long_leg.get("ask", 0.0)),
        float(long_leg.get("lastPrice", 0.0)),
    )

    short_mid = _mid_price(
        float(short_leg.get("bid", 0.0)),
        float(short_leg.get("ask", 0.0)),
        float(short_leg.get("lastPrice", 0.0)),
    )

    debit = round(long_mid - short_mid, 2)

    if debit <= 0:
        return _no_trade("SELL", ticker, "Invalid debit calculation. Option quotes may be stale.")

    spread_width = long_strike - short_strike
    max_loss = debit * 100
    max_profit = (spread_width - debit) * 100
    breakeven = long_strike - debit
    reward_risk = max_profit / max_loss if max_loss > 0 else 0.0

    current_iv = float(long_leg.get("impliedVolatility", 0.0))
    iv_rank_proxy, iv_status = _estimate_iv_rank_proxy(ticker, current_iv)

    max_risk_dollars = account_equity * risk_per_trade
    estimated_contracts = int(max_risk_dollars // max_loss) if max_loss > 0 else 0

    if estimated_contracts < 1:
        quality = "REJECT_TOO_EXPENSIVE"
        reason = "Spread max loss exceeds allowed risk budget."
    elif iv_rank_proxy > 70:
        quality = "REJECT_IV_TOO_HIGH"
        reason = "Bearish hedge exists, but premium is expensive."
    elif reward_risk < 1.0:
        quality = "REJECT_POOR_REWARD_RISK"
        reason = "Reward/risk is below 1.0."
    else:
        quality = "VALID"
        reason = "Bearish signal with acceptable spread economics."

    return OptionsRecommendation(
        stock_signal="SELL",
        options_action="CONSIDER_BEARISH_OPTIONS_TRADE",
        structure="Put Debit Spread",
        trade_quality=quality,
        ticker=ticker,
        expiration=expiration,
        dte=_days_to_expiration(expiration),
        long_strike=long_strike,
        short_strike=short_strike,
        estimated_debit=debit,
        max_loss=round(max_loss, 2),
        max_profit=round(max_profit, 2),
        breakeven=round(breakeven, 2),
        reward_risk=round(reward_risk, 2),
        iv_rank_proxy=iv_rank_proxy,
        iv_status=iv_status,
        max_risk_dollars=round(max_risk_dollars, 2),
        estimated_contracts=estimated_contracts,
        reason=reason,
    )


def _no_trade(stock_signal: str, ticker: str, reason: str) -> OptionsRecommendation:
    return OptionsRecommendation(
        stock_signal=stock_signal,
        options_action="NO_OPTIONS_TRADE",
        structure="No trade",
        trade_quality="NO_TRADE",
        ticker=ticker,
        expiration="N/A",
        dte=0,
        long_strike=0.0,
        short_strike=0.0,
        estimated_debit=0.0,
        max_loss=0.0,
        max_profit=0.0,
        breakeven=0.0,
        reward_risk=0.0,
        iv_rank_proxy=0.0,
        iv_status="N/A",
        max_risk_dollars=0.0,
        estimated_contracts=0,
        reason=reason,
    )


def recommend_options_trade(
        stock_signal: str,
        ticker: str,
        account_equity: float,
        risk_per_trade: float,
        price: float,
        atr: float,
        min_dte: int = 30,
        max_dte: int = 45,
) -> OptionsRecommendation:
    if stock_signal == "BUY":
        return _build_call_debit_spread(
            ticker=ticker,
            price=price,
            account_equity=account_equity,
            risk_per_trade=risk_per_trade,
            min_dte=min_dte,
            max_dte=max_dte,
        )

    if stock_signal == "BEARISH_ENTRY":
        return _build_put_debit_spread(
            ticker=ticker,
            price=price,
            account_equity=account_equity,
            risk_per_trade=risk_per_trade,
            min_dte=min_dte,
            max_dte=max_dte,
        )

    if stock_signal == "EXIT_LONG":
        return _no_trade(
            stock_signal=stock_signal,
            ticker=ticker,
            reason=f"{ticker} generated a long exit signal. Close the long; do not open a bearish options trade from this event.",
        )

    return _no_trade(
        stock_signal=stock_signal,
        ticker=ticker,
        reason=f"{ticker} has no actionable stock signal. Do not force an options trade.",
    )


def print_options_recommendation(rec: OptionsRecommendation) -> None:
    if rec.options_action == "NO_OPTIONS_TRADE":
        return

    print("\nExecutable Options Trade")
    print("------------------------")
    print(f"Ticker: {rec.ticker}")
    print(f"Stock Signal: {rec.stock_signal}")
    print(f"Trade Quality: {rec.trade_quality}")
    print(f"Structure: {rec.structure}")
    print(f"Expiration: {rec.expiration} ({rec.dte} DTE)")
    print(f"Buy Strike: {rec.long_strike}")
    print(f"Sell Strike: {rec.short_strike}")
    print(f"Estimated Debit: ${rec.estimated_debit:.2f}")
    print(f"Max Loss: ${rec.max_loss:,.2f}")
    print(f"Max Profit: ${rec.max_profit:,.2f}")
    print(f"Breakeven: ${rec.breakeven:.2f}")
    print(f"Reward/Risk: {rec.reward_risk:.2f}")
    print(f"IV Rank Proxy: {rec.iv_rank_proxy:.2f}")
    print(f"IV Status: {rec.iv_status}")
    print(f"Max Risk Budget: ${rec.max_risk_dollars:,.2f}")
    print(f"Estimated Contracts: {rec.estimated_contracts}")
    print(f"Reason: {rec.reason}")
