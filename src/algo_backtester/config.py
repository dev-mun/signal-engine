from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 3000.0

    stop_loss: float = 0.08
    take_profit: float = 0.20
    trailing_stop: float = 0.08
    max_hold_days: int = 60

    risk_per_trade: float = 0.015
    atr_multiple: float = 2.0

    min_avg_dollar_volume: float = 20_000_000.0
    min_atm_open_interest: int = 50
    min_atm_option_volume: int = 5
    max_atm_bid_ask_spread_pct: float = 20.0
    option_min_dte: int = 30
    option_max_dte: int = 45
    earnings_buffer_days: int = 3
