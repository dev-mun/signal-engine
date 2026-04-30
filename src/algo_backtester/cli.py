import argparse
from pathlib import Path
import sys

from algo_backtester.backtester import TrendPullbackBacktester
from algo_backtester.backtests.ema_rsi_backtester import (
    EmaRsiBacktestConfig,
    EmaRsiPullbackBacktester,
    scan_watchlist as scan_watchlist_ema_rsi,
)
from algo_backtester.backtests.four_hour_trend_backtester import (
    FourHourTrendBacktester,
    FourHourTrendConfig,
    make_demo_intraday_data,
    prepare_four_hour_data,
    scan_watchlist as scan_watchlist_four_hour,
)
from algo_backtester.config import BacktestConfig
from algo_backtester.data_loader import load_csv, load_yfinance_data, make_demo_data
from algo_backtester.metrics import buy_and_hold_curve, performance_summary, print_summary
from algo_backtester.journal import update_paper_trading_journal
from algo_backtester.options_engine import (
    print_options_recommendation,
    recommend_options_trade,
)
from algo_backtester.plotting import plot_results
from algo_backtester.reports.ema_rsi_report import (
    performance_summary as ema_rsi_performance_summary,
    plot_results as plot_ema_rsi_results,
    print_latest_signal as print_ema_rsi_latest_signal,
    print_scan_results as print_ema_rsi_scan_results,
    print_summary as print_ema_rsi_summary,
    save_reports as save_ema_rsi_reports,
    save_scan_results as save_ema_rsi_scan_results,
)
from algo_backtester.reports.four_hour_report import (
    performance_summary as four_hour_performance_summary,
    plot_results as plot_four_hour_results,
    print_latest_signal as print_four_hour_latest_signal,
    print_scan_results as print_four_hour_scan_results,
    print_summary as print_four_hour_summary,
    save_reports as save_four_hour_reports,
    save_scan_results as save_four_hour_scan_results,
)
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
    parser.add_argument(
        "--strategy",
        type=str,
        default="trend-pullback",
        choices=["trend-pullback", "ema-rsi", "four-hour-trend"],
    )
    parser.add_argument("--interval", type=str, default="1h")

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
    parser.add_argument("--journal", action="store_true")

    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        help="Comma-separated tickers to scan, example: SPY,QQQ,NVDA,AAPL,MSFT",
    )

    return parser.parse_args()


def _provided_cli_flags() -> set[str]:
    return {
        token.split("=")[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }


def _ema_rsi_config(args: argparse.Namespace) -> EmaRsiBacktestConfig:
    provided_flags = _provided_cli_flags()

    return EmaRsiBacktestConfig(
        initial_cash=args.initial_cash,
        stop_loss=args.stop_loss if "--stop-loss" in provided_flags else 0.07,
        take_profit=args.take_profit if "--take-profit" in provided_flags else 0.15,
        trailing_stop=args.trailing_stop if "--trailing-stop" in provided_flags else 0.08,
        max_hold_days=args.max_hold_days if "--max-hold-days" in provided_flags else 45,
        risk_per_trade=args.risk_per_trade if "--risk-per-trade" in provided_flags else 0.01,
        atr_multiple=args.atr_multiple if "--atr-multiple" in provided_flags else 2.0,
    )


def _four_hour_config(args: argparse.Namespace) -> FourHourTrendConfig:
    provided_flags = _provided_cli_flags()

    return FourHourTrendConfig(
        initial_cash=args.initial_cash,
        stop_loss=args.stop_loss if "--stop-loss" in provided_flags else 0.04,
        take_profit=args.take_profit if "--take-profit" in provided_flags else 0.08,
        trailing_stop=args.trailing_stop if "--trailing-stop" in provided_flags else 0.05,
        max_hold_candles=args.max_hold_days if "--max-hold-days" in provided_flags else 12,
        risk_per_trade=args.risk_per_trade if "--risk-per-trade" in provided_flags else 0.0075,
        atr_multiple=args.atr_multiple if "--atr-multiple" in provided_flags else 1.5,
        interval=args.interval,
    )


def _load_four_hour_input_data(args: argparse.Namespace):
    if args.csv:
        label = Path(args.csv).stem
        return label, load_csv(args.csv)

    if args.ticker:
        ticker = args.ticker.upper()
        return ticker, prepare_four_hour_data(
            ticker=ticker,
            interval=args.interval,
        )

    label = "DEMO_INTRADAY"
    print("Running intraday demo mode because no --ticker or --csv was provided.")
    print("For fresh intraday data, run: python main.py --ticker SPY --strategy four-hour-trend --interval 1h")
    return label, make_demo_intraday_data(rows=1600)


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

    if args.strategy == "four-hour-trend":
        four_hour_config = _four_hour_config(args)

        if args.scan:
            tickers = [t.strip().upper() for t in args.scan.split(",") if t.strip()]
            results = scan_watchlist_four_hour(
                tickers=tickers,
                interval=args.interval,
                config=four_hour_config,
            )
            print_four_hour_scan_results(results)
            save_four_hour_scan_results(results, output_dir=args.output_dir)
            return

        try:
            label, raw_df = _load_four_hour_input_data(args)
            if raw_df.empty:
                print(f"Unable to run four-hour trend strategy for {label}: no usable intraday data returned.")
                return

            bt = FourHourTrendBacktester(config=four_hour_config)
            strategy_df, equity_df, trades_df, signals_df = bt.run(raw_df)
        except Exception as exc:
            print(f"Unable to run four-hour trend strategy: {exc}")
            return

        summary = four_hour_performance_summary(
            equity_df=equity_df,
            trades_df=trades_df,
            initial_cash=four_hour_config.initial_cash,
        )
        print_four_hour_summary(summary)
        print_four_hour_latest_signal(label, signals_df)

        print("\nRecent Trades")
        print("-------------")
        if trades_df.empty:
            print("No trades generated with current rules.")
        else:
            print(trades_df.tail(20).to_string(index=False))

        if args.save_reports:
            save_four_hour_reports(
                label=label,
                equity_df=equity_df,
                trades_df=trades_df,
                signals_df=signals_df,
                output_dir=args.output_dir,
            )

        if not args.no_plot:
            try:
                bh_df = buy_and_hold_curve(
                    raw_df=strategy_df,
                    equity_index=equity_df.index,
                    initial_cash=four_hour_config.initial_cash,
                )
                plot_four_hour_results(
                    label=label,
                    price_df=strategy_df,
                    signals_df=signals_df,
                    equity_df=equity_df,
                    bh_df=bh_df,
                    take_profit_pct=four_hour_config.take_profit,
                )
            except Exception as exc:
                print(f"Skipping buy-and-hold chart: {exc}")

        return

    if args.strategy == "ema-rsi":
        ema_config = _ema_rsi_config(args)

        if args.scan:
            tickers = [t.strip().upper() for t in args.scan.split(",") if t.strip()]
            results = scan_watchlist_ema_rsi(
                tickers=tickers,
                start=args.start,
                end=args.end,
                config=ema_config,
            )
            print_ema_rsi_scan_results(results)
            save_ema_rsi_scan_results(results, output_dir=args.output_dir)
            return

        label, raw_df = load_input_data(args)
        bt = EmaRsiPullbackBacktester(config=ema_config)
        strategy_df, equity_df, trades_df, signals_df = bt.run(raw_df)

        summary = ema_rsi_performance_summary(
            equity_df=equity_df,
            trades_df=trades_df,
            initial_cash=ema_config.initial_cash,
        )
        print_ema_rsi_summary(summary)

        print_ema_rsi_latest_signal(label, signals_df)

        print("\nRecent Trades")
        print("-------------")
        if trades_df.empty:
            print("No trades generated with current rules.")
        else:
            print(trades_df.tail(20).to_string(index=False))

        if args.save_reports:
            save_ema_rsi_reports(
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
                    initial_cash=ema_config.initial_cash,
                )
                plot_ema_rsi_results(
                    label=label,
                    price_df=strategy_df,
                    signals_df=signals_df,
                    equity_df=equity_df,
                    bh_df=bh_df,
                    take_profit_pct=ema_config.take_profit,
                )
            except Exception as exc:
                print(f"Skipping buy-and-hold chart: {exc}")

        return

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
        if args.journal:
            update_paper_trading_journal(results=results, output_dir=args.output_dir)
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
