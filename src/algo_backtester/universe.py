from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from src.algo_backtester.data_loader import validate_ohlcv

ETF_SYMBOLS = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
}


@dataclass(frozen=True)
class UniverseEligibility:
    ticker: str
    is_eligible: bool
    status: str
    reason: str
    avg_dollar_volume: float
    expiration: str
    dte: int
    atm_call_open_interest: int
    atm_put_open_interest: int
    atm_call_volume: int
    atm_put_volume: int
    atm_call_spread_pct: float
    atm_put_spread_pct: float
    earnings_date: str
    rejection_reasons: tuple[str, ...]


def _days_to_expiration(expiration: str) -> int:
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    return max((exp_date - datetime.today().date()).days, 0)


def _select_expiration(
        ticker_obj: yf.Ticker,
        min_dte: int,
        max_dte: int,
) -> Optional[str]:
    expirations = ticker_obj.options

    if not expirations:
        return None

    candidates = []

    for expiration in expirations:
        dte = _days_to_expiration(expiration)
        if min_dte <= dte <= max_dte:
            candidates.append((expiration, dte))

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: abs(item[1] - ((min_dte + max_dte) // 2)))[0][0]


def _spread_pct(row: pd.Series) -> float:
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)

    if bid <= 0 or ask <= 0 or ask < bid:
        return 999.0

    mid = (bid + ask) / 2
    if mid <= 0:
        return 999.0

    return ((ask - bid) / mid) * 100


def _nearest_atm_contract(chain: pd.DataFrame, spot_price: float) -> Optional[pd.Series]:
    if chain.empty:
        return None

    df = chain.copy()
    df["strike_distance"] = (df["strike"] - spot_price).abs()
    return df.sort_values("strike_distance").iloc[0]


def _normalize_date(value) -> Optional[date]:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.date()

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return pd.Timestamp(value).date()
        except Exception:
            return None

    return None


def _next_earnings_date(ticker_obj: yf.Ticker) -> Optional[date]:
    try:
        earnings_dates = ticker_obj.get_earnings_dates(limit=1)
        if isinstance(earnings_dates, pd.DataFrame):
            if len(earnings_dates.index) > 0:
                index_value = earnings_dates.index[0]
                normalized = _normalize_date(index_value)
                if normalized is not None:
                    return normalized
    except Exception:
        pass

    calendar = getattr(ticker_obj, "calendar", None)

    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
        for column in ["Earnings Date", "Earnings Date Start", "Earnings Date End"]:
            if column in calendar.columns:
                normalized = _normalize_date(calendar[column].iloc[0])
                if normalized is not None:
                    return normalized

    if isinstance(calendar, dict):
        raw_value = calendar.get("Earnings Date")
        if isinstance(raw_value, (list, tuple)) and raw_value:
            return _normalize_date(raw_value[0])
        return _normalize_date(raw_value)

    return None


def evaluate_universe_eligibility(
        ticker: str,
        raw_df: pd.DataFrame,
        min_avg_dollar_volume: float,
        min_atm_open_interest: int,
        min_atm_option_volume: int,
        max_atm_bid_ask_spread_pct: float,
        min_dte: int,
        max_dte: int,
        earnings_buffer_days: int,
        ticker_obj: yf.Ticker | None = None,
) -> UniverseEligibility:
    df = validate_ohlcv(raw_df)
    ticker_obj = ticker_obj or yf.Ticker(ticker)

    avg_dollar_volume = float((df["Close"] * df["Volume"]).tail(20).mean())
    rejection_reasons: list[str] = []

    if avg_dollar_volume < min_avg_dollar_volume:
        rejection_reasons.append(
            f"20-day average dollar volume ${avg_dollar_volume:,.0f} is below ${min_avg_dollar_volume:,.0f}."
        )

    expiration = _select_expiration(
        ticker_obj=ticker_obj,
        min_dte=min_dte,
        max_dte=max_dte,
    )

    if expiration is None:
        rejection_reasons.append(f"No listed options expiration found in the {min_dte}-{max_dte} DTE window.")
        return UniverseEligibility(
            ticker=ticker,
            is_eligible=False,
            status="FILTERED_OUT",
            reason=" | ".join(rejection_reasons),
            avg_dollar_volume=avg_dollar_volume,
            expiration="N/A",
            dte=0,
            atm_call_open_interest=0,
            atm_put_open_interest=0,
            atm_call_volume=0,
            atm_put_volume=0,
            atm_call_spread_pct=999.0,
            atm_put_spread_pct=999.0,
            earnings_date="UNKNOWN",
            rejection_reasons=tuple(rejection_reasons),
        )

    dte = _days_to_expiration(expiration)
    chain = ticker_obj.option_chain(expiration)
    spot_price = float(df["Close"].iloc[-1])
    call_contract = _nearest_atm_contract(chain.calls, spot_price)
    put_contract = _nearest_atm_contract(chain.puts, spot_price)

    if call_contract is None or put_contract is None:
        rejection_reasons.append("ATM call/put liquidity could not be evaluated from the options chain.")
        return UniverseEligibility(
            ticker=ticker,
            is_eligible=False,
            status="FILTERED_OUT",
            reason=" | ".join(rejection_reasons),
            avg_dollar_volume=avg_dollar_volume,
            expiration=expiration,
            dte=dte,
            atm_call_open_interest=0,
            atm_put_open_interest=0,
            atm_call_volume=0,
            atm_put_volume=0,
            atm_call_spread_pct=999.0,
            atm_put_spread_pct=999.0,
            earnings_date="UNKNOWN",
            rejection_reasons=tuple(rejection_reasons),
        )

    call_oi = int(call_contract.get("openInterest", 0) or 0)
    put_oi = int(put_contract.get("openInterest", 0) or 0)
    call_volume = int(call_contract.get("volume", 0) or 0)
    put_volume = int(put_contract.get("volume", 0) or 0)
    call_spread_pct = round(_spread_pct(call_contract), 2)
    put_spread_pct = round(_spread_pct(put_contract), 2)

    if min(call_oi, put_oi) < min_atm_open_interest:
        rejection_reasons.append(
            "ATM call/put open interest is too low for consistent liquidity."
        )

    if min(call_volume, put_volume) < min_atm_option_volume:
        rejection_reasons.append(
            "ATM call/put daily option volume is too low for consistent liquidity."
        )

    if max(call_spread_pct, put_spread_pct) > max_atm_bid_ask_spread_pct:
        rejection_reasons.append(
            f"ATM option bid/ask spread is wider than {max_atm_bid_ask_spread_pct:.1f}%."
        )

    if ticker.upper() in ETF_SYMBOLS:
        earnings_date = None
        earnings_date_str = "N/A"
    else:
        earnings_date = _next_earnings_date(ticker_obj)
        earnings_date_str = earnings_date.isoformat() if earnings_date else "UNKNOWN"

        if earnings_date is not None:
            days_to_earnings = (earnings_date - datetime.today().date()).days
            if 0 <= days_to_earnings <= earnings_buffer_days:
                rejection_reasons.append(
                    f"Earnings date {earnings_date.isoformat()} is within the {earnings_buffer_days}-day buffer."
                )

    is_eligible = not rejection_reasons

    if is_eligible:
        reason = (
            f"Eligible universe member. Avg dollar volume ${avg_dollar_volume:,.0f}, "
            f"{expiration} ({dte} DTE), ATM call/put OI {call_oi}/{put_oi}."
        )
    else:
        reason = " | ".join(rejection_reasons)

    return UniverseEligibility(
        ticker=ticker,
        is_eligible=is_eligible,
        status="ELIGIBLE" if is_eligible else "FILTERED_OUT",
        reason=reason,
        avg_dollar_volume=avg_dollar_volume,
        expiration=expiration,
        dte=dte,
        atm_call_open_interest=call_oi,
        atm_put_open_interest=put_oi,
        atm_call_volume=call_volume,
        atm_put_volume=put_volume,
        atm_call_spread_pct=call_spread_pct,
        atm_put_spread_pct=put_spread_pct,
        earnings_date=earnings_date_str,
        rejection_reasons=tuple(rejection_reasons),
    )
