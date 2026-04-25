from datetime import datetime, timedelta

import pandas as pd

import src.algo_backtester.scanner as scanner_module
from src.algo_backtester.config import BacktestConfig
from src.algo_backtester.universe import evaluate_universe_eligibility


class FakeOptionChain:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame):
        self.calls = calls
        self.puts = puts


class FakeTicker:
    def __init__(self, option_chain: FakeOptionChain, expirations: list[str], earnings_dates: pd.DataFrame | None = None):
        self._option_chain = option_chain
        self.options = expirations
        self._earnings_dates = earnings_dates if earnings_dates is not None else pd.DataFrame()

    def option_chain(self, expiration: str):
        return self._option_chain

    def get_earnings_dates(self, limit: int = 1):
        return self._earnings_dates


def make_raw_df(close: float = 100.0, volume: int = 500_000) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=260)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [close] * len(dates),
            "High": [close * 1.01] * len(dates),
            "Low": [close * 0.99] * len(dates),
            "Close": [close] * len(dates),
            "Volume": [volume] * len(dates),
        }
    )


def make_option_chain(
        open_interest: int = 1_000,
        option_volume: int = 100,
        bid: float = 4.9,
        ask: float = 5.1,
) -> FakeOptionChain:
    contracts = pd.DataFrame(
        {
            "strike": [100.0, 105.0],
            "openInterest": [open_interest, open_interest],
            "volume": [option_volume, option_volume],
            "bid": [bid, bid],
            "ask": [ask, ask],
        }
    )
    return FakeOptionChain(calls=contracts.copy(), puts=contracts.copy())


def test_evaluate_universe_eligibility_accepts_liquid_name():
    expiration = (datetime.today().date() + timedelta(days=35)).isoformat()
    eligibility = evaluate_universe_eligibility(
        ticker="AAPL",
        raw_df=make_raw_df(close=200.0, volume=250_000),
        min_avg_dollar_volume=20_000_000.0,
        min_atm_open_interest=500,
        min_atm_option_volume=50,
        max_atm_bid_ask_spread_pct=12.0,
        min_dte=30,
        max_dte=45,
        earnings_buffer_days=5,
        ticker_obj=FakeTicker(
            option_chain=make_option_chain(),
            expirations=[expiration],
        ),
    )

    assert eligibility.is_eligible is True
    assert eligibility.status == "ELIGIBLE"
    assert eligibility.dte >= 30


def test_evaluate_universe_eligibility_rejects_near_earnings():
    expiration = (datetime.today().date() + timedelta(days=35)).isoformat()
    earnings_idx = pd.DatetimeIndex([pd.Timestamp.today().normalize() + pd.Timedelta(days=2)])
    eligibility = evaluate_universe_eligibility(
        ticker="AAPL",
        raw_df=make_raw_df(close=200.0, volume=250_000),
        min_avg_dollar_volume=20_000_000.0,
        min_atm_open_interest=500,
        min_atm_option_volume=50,
        max_atm_bid_ask_spread_pct=12.0,
        min_dte=30,
        max_dte=45,
        earnings_buffer_days=5,
        ticker_obj=FakeTicker(
            option_chain=make_option_chain(),
            expirations=[expiration],
            earnings_dates=pd.DataFrame(index=earnings_idx),
        ),
    )

    assert eligibility.is_eligible is False
    assert "Earnings date" in eligibility.reason


def test_scan_watchlist_returns_ineligible_without_running_backtest(monkeypatch):
    raw_df = make_raw_df(close=200.0, volume=10_000)

    monkeypatch.setattr(scanner_module, "load_yfinance_data", lambda ticker, start, end=None: raw_df)
    monkeypatch.setattr(
        scanner_module,
        "evaluate_universe_eligibility",
        lambda **kwargs: type(
            "Eligibility",
            (),
            {
                "is_eligible": False,
                "status": "FILTERED_OUT",
                "reason": "Rejected by liquidity filter.",
                "dte": 35,
                "avg_dollar_volume": 2_000_000.0,
                "earnings_date": "UNKNOWN",
            },
        )(),
    )

    class ExplodingBacktester:
        def __init__(self, config):
            raise AssertionError("Backtester should not run for ineligible tickers.")

    monkeypatch.setattr(scanner_module, "TrendPullbackBacktester", ExplodingBacktester)

    results = scanner_module.scan_watchlist(
        tickers=["AAPL"],
        config=BacktestConfig(),
        enforce_universe_filter=True,
    )

    assert results[0]["Signal"] == "INELIGIBLE"
    assert results[0]["UniverseStatus"] == "FILTERED_OUT"
    assert results[0]["Reason"] == "Rejected by liquidity filter."


def test_evaluate_universe_eligibility_skips_earnings_for_etf():
    expiration = (datetime.today().date() + timedelta(days=35)).isoformat()
    earnings_idx = pd.DatetimeIndex([pd.Timestamp.today().normalize() + pd.Timedelta(days=1)])

    eligibility = evaluate_universe_eligibility(
        ticker="SPY",
        raw_df=make_raw_df(close=500.0, volume=500_000),
        min_avg_dollar_volume=20_000_000.0,
        min_atm_open_interest=50,
        min_atm_option_volume=5,
        max_atm_bid_ask_spread_pct=20.0,
        min_dte=30,
        max_dte=45,
        earnings_buffer_days=3,
        ticker_obj=FakeTicker(
            option_chain=make_option_chain(),
            expirations=[expiration],
            earnings_dates=pd.DataFrame(index=earnings_idx),
        ),
    )

    assert eligibility.is_eligible is True
    assert eligibility.earnings_date == "N/A"
