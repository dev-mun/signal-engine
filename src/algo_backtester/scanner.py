from pathlib import Path
from typing import Iterable

from algo_backtester.backtester import TrendPullbackBacktester
from algo_backtester.config import BacktestConfig
from algo_backtester.data_loader import load_yfinance_data
from algo_backtester.options_engine import recommend_options_trade
from algo_backtester.universe import evaluate_universe_eligibility


def classify_setup(signal: str, rsi: float) -> str:
    if signal in {"BUY", "BEARISH_ENTRY"}:
        return "ACTIONABLE"

    if signal == "EXIT_LONG":
        return "EXIT"

    if rsi >= 70:
        return "EXTENDED"

    if 55 <= rsi < 70:
        return "NEAR_SETUP"

    if rsi < 40:
        return "WEAK"

    return "WAIT"


def distance_to_setup(signal: str, rsi: float) -> str:
    if signal in {"BUY", "BEARISH_ENTRY"}:
        return "Actionable now"

    if signal == "EXIT_LONG":
        return "Exit long position next open"

    if rsi >= 70:
        return f"Too hot (-{rsi - 60:.1f} RSI to setup)"

    if 60 <= rsi < 70:
        return f"Needs mild pullback (-{rsi - 60:.1f} RSI)"

    if 55 <= rsi < 60:
        return "Very close"

    if 40 <= rsi < 55:
        return f"Needs strength (+{55 - rsi:.1f} RSI)"

    return "Too weak"


def scan_ticker(
        ticker: str,
        config: BacktestConfig,
        start: str = "2018-01-01",
        enforce_universe_filter: bool = True,
) -> dict:
    raw_df = load_yfinance_data(ticker=ticker, start=start, end=None)
    latest_price = float(raw_df["Close"].iloc[-1])

    eligibility = evaluate_universe_eligibility(
        ticker=ticker,
        raw_df=raw_df,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        min_atm_open_interest=config.min_atm_open_interest,
        min_atm_option_volume=config.min_atm_option_volume,
        max_atm_bid_ask_spread_pct=config.max_atm_bid_ask_spread_pct,
        min_dte=config.option_min_dte,
        max_dte=config.option_max_dte,
        earnings_buffer_days=config.earnings_buffer_days,
    )

    if enforce_universe_filter and not eligibility.is_eligible:
        return {
            "Ticker": ticker,
            "UniverseStatus": eligibility.status,
            "UniverseReason": eligibility.reason,
            "Signal": "INELIGIBLE",
            "SetupStatus": "FILTERED_OUT",
            "DistanceToSetup": "Rejected by universe filter",
            "Price": latest_price,
            "RSI": 0.0,
            "ATR": 0.0,
            "Equity": 0.0,
            "OptionsAction": "NO_OPTIONS_TRADE",
            "Structure": "No trade",
            "DTE": eligibility.dte,
            "Reason": eligibility.reason,
            "OptionsReason": eligibility.reason,
            "AvgDollarVolume": eligibility.avg_dollar_volume,
            "EarningsDate": eligibility.earnings_date,
        }

    bt = TrendPullbackBacktester(config=config)
    _, _, _, signals_df = bt.run(raw_df)

    latest = signals_df.iloc[-1]

    signal = str(latest["Signal"])
    price = float(latest["Close"])
    rsi = float(latest["RSI"])
    atr = float(latest["ATR"])
    equity = float(latest["Equity"])
    reason = str(latest["Reason"])

    setup_status = classify_setup(signal, rsi)
    setup_distance = distance_to_setup(signal, rsi)

    options_rec = recommend_options_trade(
        stock_signal=signal,
        ticker=ticker,
        account_equity=equity,
        risk_per_trade=config.risk_per_trade,
        price=price,
        atr=atr,
        min_dte=config.option_min_dte,
        max_dte=config.option_max_dte,
    )

    return {
        "Ticker": ticker,
        "UniverseStatus": eligibility.status,
        "UniverseReason": eligibility.reason,
        "Signal": signal,
        "SetupStatus": setup_status,
        "DistanceToSetup": setup_distance,
        "Price": price,
        "RSI": rsi,
        "ATR": atr,
        "Equity": equity,
        "OptionsAction": options_rec.options_action,
        "Structure": options_rec.structure,
        "DTE": options_rec.dte,
        "Reason": reason,
        "OptionsReason": options_rec.reason,
        "AvgDollarVolume": eligibility.avg_dollar_volume,
        "EarningsDate": eligibility.earnings_date,
    }


def scan_watchlist(
        tickers: Iterable[str],
        config: BacktestConfig,
        start: str = "2018-01-01",
        enforce_universe_filter: bool = True,
) -> list[dict]:
    results = []

    for ticker in tickers:
        clean_ticker = ticker.strip().upper()

        if not clean_ticker:
            continue

        try:
            result = scan_ticker(
                ticker=clean_ticker,
                config=config,
                start=start,
                enforce_universe_filter=enforce_universe_filter,
            )
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "Ticker": clean_ticker,
                    "UniverseStatus": "ERROR",
                    "UniverseReason": str(exc),
                    "Signal": "ERROR",
                    "SetupStatus": "ERROR",
                    "DistanceToSetup": "ERROR",
                    "Price": 0.0,
                    "RSI": 0.0,
                    "ATR": 0.0,
                    "Equity": 0.0,
                    "OptionsAction": "ERROR",
                    "Structure": "N/A",
                    "DTE": "N/A",
                    "Reason": str(exc),
                    "OptionsReason": str(exc),
                    "AvgDollarVolume": 0.0,
                    "EarningsDate": "UNKNOWN",
                }
            )

    return results


def save_scan_results(results: list[dict], output_dir: str = "reports") -> None:
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    if df.empty:
        return

    date_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    file_path = output_path / f"watchlist_scan_{date_str}.csv"

    df.to_csv(file_path, index=False)

    print(f"\nSaved Watchlist Scan: {file_path}")


def print_scan_results(results: list[dict]) -> None:
    actionable = [r for r in results if r["Signal"] in {"BUY", "BEARISH_ENTRY"}]
    near_setups = [r for r in results if r["SetupStatus"] == "NEAR_SETUP"]
    exits = [r for r in results if r["Signal"] == "EXIT_LONG"]
    filtered = [r for r in results if r["Signal"] == "INELIGIBLE"]
    errors = [r for r in results if r["Signal"] == "ERROR"]

    print("\nWatchlist Scan")
    print("--------------")
    print(
        f'{"Ticker":<8} '
        f'{"Universe":<14} '
        f'{"Signal":<15} '
        f'{"Setup":<12} '
        f'{"Price":>10} '
        f'{"RSI":>8} '
        f'{"ATR":>8}  '
        f"Distance"
    )
    print("-" * 126)

    for r in results:
        if r["Signal"] == "ERROR":
            continue

        print(
            f'{r["Ticker"]:<8} '
            f'{r["UniverseStatus"]:<14} '
            f'{r["Signal"]:<15} '
            f'{r["SetupStatus"]:<12} '
            f'{r["Price"]:>10.2f} '
            f'{r["RSI"]:>8.2f} '
            f'{r["ATR"]:>8.2f}  '
            f'{r["DistanceToSetup"]}'
        )

    print("\nActionable Signals")
    print("------------------")

    if not actionable:
        print("No actionable bullish or bearish entry signals.")
    else:
        for r in actionable:
            print(
                f'{r["Ticker"]}: {r["Signal"]} | '
                f'{r["OptionsAction"]} | '
                f'{r["Structure"]} | '
                f'Price={r["Price"]:.2f} | ATR={r["ATR"]:.2f}'
            )

    print("\nExit Signals")
    print("------------")

    if not exits:
        print("No exit-long signals.")
    else:
        for r in exits:
            print(
                f'{r["Ticker"]}: EXIT_LONG | '
                f'Price={r["Price"]:.2f} | ATR={r["ATR"]:.2f} | '
                f'{r["Reason"]}'
            )

    print("\nFiltered Out")
    print("------------")

    if not filtered:
        print("No tickers were filtered out by the universe rules.")
    else:
        for r in filtered:
            print(f'{r["Ticker"]}: {r["UniverseReason"]}')

    print("\nNear Setups")
    print("-----------")

    if not near_setups:
        print("No near setups.")
    else:
        for r in near_setups:
            print(
                f'{r["Ticker"]}: '
                f'RSI={r["RSI"]:.2f} | '
                f'Price={r["Price"]:.2f} | '
                f'ATR={r["ATR"]:.2f} | '
                f'{r["DistanceToSetup"]}'
            )

    if errors:
        print("\nErrors")
        print("------")
        for r in errors:
            print(f'{r["Ticker"]}: {r["Reason"]}')
