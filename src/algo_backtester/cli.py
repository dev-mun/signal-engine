import argparse
from pathlib import Path

from algo_backtester.backtester import TrendPullbackBacktester
from algo_backtester.config import BacktestConfig
from algo_backtester.data_loader import load_csv, load_yfinance_data, make_demo_data
from algo_backtester.metrics import buy_and_hold_curve, performance_summary, print_summary
from algo_backtester.options_engine import (
    print_options_recommendation,
    recommend_options_trade,
)
from algo_backtester.plotting import plot_results
from algo_backtester.reporting import print_latest_signal, save_reports
from algo_backtester.scanner import (
    print_scan_results,
    save_scan_results,
    scan_watchlist,
)
from algo_backtester.trade_plan import (
    build_signal_interpretation,
    build_trade_plan,
    print_signal_interpretation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="signal-engine")

    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-reports", action="store_true")
    parser.add_argument("--output-dir", type=str, default="reports")

    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--stop-loss", type=float, default=0.08)
    parser.add_argument("--take-profit", type=float, default=0.20)
    parser.add_argument("--trailing-stop", type=float, default=0.08)
    parser.add_argument("--max-hold-days", type=int, default=60)
    parser.add_argument("--risk-per-trade", type=float, default=0.015)
    parser.add_argument("--atr-multiple", type=float, default=2.0)
    parser.add_argument("--min-avg-dollar-volume", type=float, default=20_000_000.0)
    parser.add_argument("--min-atm-open-interest", type=int, default=50)
    parser.add_argument("--min-atm-option-volume", type=int, default=5)
    parser.add_argument("--max-atm-bid-ask-spread-pct", type=float, default=20.0)
    parser.add_argument("--option-min-dte", type=int, default=30)
    parser.add_argument("--option-max-dte", type=int, default=45)
    parser.add_argument("--earnings-buffer-days", type=int, default=3)
    parser.add_argument("--no-universe-filter", action="store_true")

    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        help="Comma-separated tickers to scan, example: SPY,QQQ,NVDA,AAPL,MSFT",
    )

    return parser.parse_args()


def load_input_data(args: argparse.Namespace):
    if args.csv:
        label = Path(args.csv).stem
        return label, load_csv(args.csv)

    if args.ticker:
        ticker = args.ticker.upper()
        return ticker, load_yfinance_data(
            ticker=ticker,
            start=args.start,
            end=args.end,
        )

    label = "DEMO_DATA"
    print("Running demo mode because no --ticker or --csv was provided.")
    print("For fresh market data, run: python main.py --ticker SPY")
    return label, make_demo_data(rows=500)


def main() -> None:
    args = parse_args()

    config = BacktestConfig(
        initial_cash=args.initial_cash,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        trailing_stop=args.trailing_stop,
        max_hold_days=args.max_hold_days,
        risk_per_trade=args.risk_per_trade,
        atr_multiple=args.atr_multiple,
        min_avg_dollar_volume=args.min_avg_dollar_volume,
        min_atm_open_interest=args.min_atm_open_interest,
        min_atm_option_volume=args.min_atm_option_volume,
        max_atm_bid_ask_spread_pct=args.max_atm_bid_ask_spread_pct,
        option_min_dte=args.option_min_dte,
        option_max_dte=args.option_max_dte,
        earnings_buffer_days=args.earnings_buffer_days,
    )

    if args.scan:
        tickers = [t.strip().upper() for t in args.scan.split(",") if t.strip()]
        results = scan_watchlist(
            tickers=tickers,
            config=config,
            start=args.start,
            enforce_universe_filter=not args.no_universe_filter,
        )
        print_scan_results(results)
        save_scan_results(results, output_dir=args.output_dir)
        return

    label, raw_df = load_input_data(args)

    bt = TrendPullbackBacktester(config=config)
    strategy_df, equity_df, trades_df, signals_df = bt.run(raw_df)

    summary = performance_summary(
        equity_df=equity_df,
        trades_df=trades_df,
        initial_cash=args.initial_cash,
    )

    print_summary(summary)

    latest = print_latest_signal(label, signals_df)
    options_rec = recommend_options_trade(
        stock_signal=latest["Signal"],
        ticker=label,
        account_equity=latest["Equity"],
        risk_per_trade=config.risk_per_trade,
        price=latest["Close"],
        atr=latest["ATR"],
        min_dte=config.option_min_dte,
        max_dte=config.option_max_dte,
    )

    trade_plan = build_trade_plan(
        signal=latest["Signal"],
        entry_price=latest["Close"],
        stop_loss_pct=config.stop_loss,
        take_profit_pct=config.take_profit,
        trailing_stop_pct=config.trailing_stop,
    )
    interpretation = build_signal_interpretation(
        latest_signal=latest,
        trade_plan=trade_plan,
        options_rec=options_rec,
    )

    print_signal_interpretation(interpretation)

    print_options_recommendation(options_rec)

    print("\nRecent Trades")
    print("-------------")
    if trades_df.empty:
        print("No trades generated with current rules.")
    else:
        print(trades_df.tail(20).to_string(index=False))

    if args.save_reports:
        save_reports(
            label=label,
            equity_df=equity_df,
            trades_df=trades_df,
            signals_df=signals_df,
            output_dir=args.output_dir,
        )

    if not args.no_plot:
        try:
            bh_df = buy_and_hold_curve(
                raw_df=raw_df,
                equity_index=equity_df.index,
                initial_cash=args.initial_cash,
            )
            plot_results(
                label=label,
                price_df=strategy_df,
                signals_df=signals_df,
                equity_df=equity_df,
                bh_df=bh_df,
                take_profit_pct=config.take_profit,
            )
        except Exception as exc:
            print(f"Skipping buy-and-hold chart: {exc}")
