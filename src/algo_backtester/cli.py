from __future__ import annotations

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
from algo_backtester.backtests.options_momentum_backtester import (
    OptionsMomentumConfig,
    analyze_ticker as analyze_options_momentum_ticker,
    scan_watchlist as scan_watchlist_options_momentum,
)
from algo_backtester.backtests.swing_options_debit_spread_backtester import (
    SwingOptionsDebitSpreadConfig,
    analyze_ticker as analyze_swing_options_debit_spread_ticker,
    scan_watchlist as scan_watchlist_swing_options_debit_spread,
)
from algo_backtester.backtests.swing_options_backtester import (
    SMALL_ACCOUNT_OPTIONS_PROFILE,
    SwingOptionsConfig,
    SWING_OPTIONS_ACCOUNT_TIERS,
    SWING_OPTIONS_PREMIUM_BUDGET_MODES,
    apply_execution_profile_labels,
    analyze_ticker as analyze_swing_options_ticker,
    scan_watchlist as scan_watchlist_swing_options,
)
from algo_backtester.backtests.rsi_bollinger_backtester import (
    RsiBollingerBacktestConfig,
    RsiBollingerBacktester,
    scan_watchlist as scan_watchlist_rsi_bollinger,
)
from algo_backtester.backtests.rsi_bollinger_v2_backtester import (
    RsiBollingerV2BacktestConfig,
    RsiBollingerV2Backtester,
    resolve_ticker_config,
    scan_watchlist as scan_watchlist_rsi_bollinger_v2,
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
from algo_backtester.reports.options_momentum_report import (
    print_scan_results as print_options_momentum_scan_results,
    print_ticker_plan as print_options_momentum_ticker_plan,
    save_reports as save_options_momentum_reports,
    save_scan_results as save_options_momentum_scan_results,
)
from algo_backtester.reports.swing_options_report import (
    print_scan_results as print_swing_options_scan_results,
    print_ticker_plan as print_swing_options_ticker_plan,
    save_reports as save_swing_options_reports,
    save_scan_results as save_swing_options_scan_results,
)
from algo_backtester.reports.swing_options_debit_spread_report import (
    print_scan_results as print_swing_options_debit_spread_scan_results,
    print_ticker_plan as print_swing_options_debit_spread_ticker_plan,
    save_reports as save_swing_options_debit_spread_reports,
    save_scan_results as save_swing_options_debit_spread_scan_results,
)
from algo_backtester.reports.rsi_bollinger_report import (
    performance_summary as rsi_bollinger_performance_summary,
    plot_results as plot_rsi_bollinger_results,
    print_latest_signal as print_rsi_bollinger_latest_signal,
    print_scan_results as print_rsi_bollinger_scan_results,
    print_summary as print_rsi_bollinger_summary,
    save_reports as save_rsi_bollinger_reports,
    save_scan_results as save_rsi_bollinger_scan_results,
)
from algo_backtester.reports.rsi_bollinger_v2_report import (
    performance_summary as rsi_bollinger_v2_performance_summary,
    plot_results as plot_rsi_bollinger_v2_results,
    print_latest_signal as print_rsi_bollinger_v2_latest_signal,
    print_scan_results as print_rsi_bollinger_v2_scan_results,
    print_summary as print_rsi_bollinger_v2_summary,
    save_reports as save_rsi_bollinger_v2_reports,
    save_scan_results as save_rsi_bollinger_v2_scan_results,
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
from algo_backtester.watchlists import (
    DEFAULT_STRATEGY_PROFILES,
    WATCHLIST_PROFILES,
    parse_scan_universe,
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
        choices=[
            "trend-pullback",
            "ema-rsi",
            "four-hour-trend",
            "rsi-bollinger",
            "rsi-bollinger-v2",
            "options-momentum",
            "swing-options",
            "swing-options-debit-spread",
        ],
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
        nargs="?",
        const="",
        default=None,
        help="Optional comma-separated tickers to scan, example: SPY,QQQ,NVDA,AAPL,MSFT",
    )
    parser.add_argument("--profile", type=str, default=None, choices=sorted(WATCHLIST_PROFILES))

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


def _rsi_bollinger_config(args: argparse.Namespace) -> RsiBollingerBacktestConfig:
    provided_flags = _provided_cli_flags()

    return RsiBollingerBacktestConfig(
        initial_cash=args.initial_cash,
        stop_loss=args.stop_loss if "--stop-loss" in provided_flags else 0.05,
        take_profit=args.take_profit if "--take-profit" in provided_flags else 0.06,
        trailing_stop=args.trailing_stop if "--trailing-stop" in provided_flags else 0.04,
        max_hold_days=args.max_hold_days if "--max-hold-days" in provided_flags else 10,
        risk_per_trade=args.risk_per_trade if "--risk-per-trade" in provided_flags else 0.0075,
        atr_multiple=args.atr_multiple if "--atr-multiple" in provided_flags else 1.5,
    )


def _rsi_bollinger_v2_config(args: argparse.Namespace) -> RsiBollingerV2BacktestConfig:
    provided_flags = _provided_cli_flags()

    return RsiBollingerV2BacktestConfig(
        initial_cash=args.initial_cash,
        stop_loss=args.stop_loss if "--stop-loss" in provided_flags else 0.04,
        take_profit=args.take_profit if "--take-profit" in provided_flags else 0.04,
        trailing_stop=args.trailing_stop if "--trailing-stop" in provided_flags else 0.03,
        max_hold_days=args.max_hold_days if "--max-hold-days" in provided_flags else 7,
        risk_per_trade=args.risk_per_trade if "--risk-per-trade" in provided_flags else 0.005,
        atr_multiple=args.atr_multiple if "--atr-multiple" in provided_flags else 1.25,
        rsi_threshold=42.0,
        volume_multiplier=0.6,
        band_tolerance=1.03,
        require_confirmation=False,
    )


def _options_momentum_config(args: argparse.Namespace) -> OptionsMomentumConfig:
    provided_flags = _provided_cli_flags()

    return OptionsMomentumConfig(
        initial_cash=args.initial_cash,
        risk_per_trade=args.risk_per_trade if "--risk-per-trade" in provided_flags else 0.015,
        max_contracts=1,
        min_dte=14,
        max_dte=30,
        interval=args.interval,
    )


def _swing_options_config(args: argparse.Namespace) -> SwingOptionsConfig:
    provided_flags = _provided_cli_flags()

    return SwingOptionsConfig(
        initial_cash=args.initial_cash if "--initial-cash" in provided_flags else SWING_OPTIONS_ACCOUNT_TIERS["mid_account"],
        risk_per_trade=args.risk_per_trade if "--risk-per-trade" in provided_flags else SWING_OPTIONS_PREMIUM_BUDGET_MODES["standard"],
        max_contracts=1,
        interval=args.interval,
        min_dte=30,
        max_dte=60,
        preferred_dte=45,
        time_stop_days=5,
        max_hold_days=args.max_hold_days if "--max-hold-days" in provided_flags else 15,
    )


def _swing_options_debit_spread_config(args: argparse.Namespace) -> SwingOptionsDebitSpreadConfig:
    provided_flags = _provided_cli_flags()

    return SwingOptionsDebitSpreadConfig(
        account_size=args.initial_cash if "--initial-cash" in provided_flags else 3_000.0,
        max_contracts=1,
        max_debit=150.0,
        preferred_debit_min=50.0,
        preferred_debit_max=125.0,
        min_dte=30,
        max_dte=60,
        preferred_dte=45,
        max_hold_days=args.max_hold_days if "--max-hold-days" in provided_flags else 15,
        time_stop_days=5,
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


def _parse_explicit_scan_tickers(scan_arg: str | None) -> list[str]:
    if not scan_arg:
        return []
    return [ticker.strip().upper() for ticker in scan_arg.split(",") if ticker.strip()]


def _resolve_scan_request(args: argparse.Namespace) -> tuple[list[str], str]:
    tickers = parse_scan_universe(
        scan_arg=args.scan,
        strategy=args.strategy,
        profile=args.profile,
    )

    if _parse_explicit_scan_tickers(args.scan):
        return tickers, "custom (explicit tickers override)"

    if args.profile:
        return tickers, args.profile

    default_profile = DEFAULT_STRATEGY_PROFILES.get(args.strategy)
    if default_profile is None:
        raise ValueError(f"No default watchlist profile configured for strategy '{args.strategy}'.")

    return tickers, default_profile


def _print_resolved_watchlist(profile_label: str, tickers: list[str]) -> None:
    print("\nResolved Watchlist")
    print("------------------")
    print(f"Profile: {profile_label}")
    print(f"Tickers: {', '.join(tickers) if tickers else '(none)'}")


def main() -> None:
    args = parse_args()

    if args.strategy == "swing-options-debit-spread":
        debit_spread_config = _swing_options_debit_spread_config(args)

        if args.scan is not None:
            tickers, profile_label = _resolve_scan_request(args)
            _print_resolved_watchlist(profile_label, tickers)
            results = scan_watchlist_swing_options_debit_spread(
                tickers=tickers,
                start=args.start,
                end=args.end,
                config=debit_spread_config,
            )
            print_swing_options_debit_spread_scan_results(results)
            save_swing_options_debit_spread_scan_results(results, output_dir=args.output_dir)
            if args.journal:
                update_paper_trading_journal(results=results, output_dir=args.output_dir)
            return

        if not args.ticker:
            print("swing-options-debit-spread deep dive requires --ticker.")
            return

        analysis = analyze_swing_options_debit_spread_ticker(
            ticker=args.ticker.upper(),
            start=args.start,
            end=args.end,
            config=debit_spread_config,
        )
        print_swing_options_debit_spread_ticker_plan(analysis)
        if args.save_reports:
            save_swing_options_debit_spread_reports(analysis=analysis, output_dir=args.output_dir)
        if args.journal:
            update_paper_trading_journal(results=[analysis["result"]], output_dir=args.output_dir)
        return

    if args.strategy == "swing-options":
        swing_options_config = _swing_options_config(args)

        if args.scan is not None:
            tickers, profile_label = _resolve_scan_request(args)
            _print_resolved_watchlist(profile_label, tickers)
            results = scan_watchlist_swing_options(
                tickers=tickers,
                start=args.start,
                end=args.end,
                config=swing_options_config,
            )
            account_profile = SMALL_ACCOUNT_OPTIONS_PROFILE if profile_label == SMALL_ACCOUNT_OPTIONS_PROFILE else "standard"
            results = [apply_execution_profile_labels(result=result, account_profile=account_profile) for result in results]
            print_swing_options_scan_results(results)
            save_swing_options_scan_results(results, output_dir=args.output_dir)
            if args.journal:
                update_paper_trading_journal(results=results, output_dir=args.output_dir)
            return

        if not args.ticker:
            print("swing-options deep dive requires --ticker.")
            return

        analysis = analyze_swing_options_ticker(
            ticker=args.ticker.upper(),
            start=args.start,
            end=args.end,
            config=swing_options_config,
        )
        print_swing_options_ticker_plan(analysis)
        if args.save_reports:
            save_swing_options_reports(analysis=analysis, output_dir=args.output_dir)
        if args.journal:
            update_paper_trading_journal(results=[analysis["result"]], output_dir=args.output_dir)
        return

    if args.strategy == "options-momentum":
        options_momentum_config = _options_momentum_config(args)

        if args.scan is not None:
            tickers, profile_label = _resolve_scan_request(args)
            _print_resolved_watchlist(profile_label, tickers)
            results = scan_watchlist_options_momentum(
                tickers=tickers,
                start=args.start,
                end=args.end,
                config=options_momentum_config,
            )
            print_options_momentum_scan_results(results)
            save_options_momentum_scan_results(results, output_dir=args.output_dir)
            if args.journal:
                update_paper_trading_journal(results=results, output_dir=args.output_dir)
            return

        if not args.ticker:
            print("options-momentum deep dive requires --ticker.")
            return

        analysis = analyze_options_momentum_ticker(
            ticker=args.ticker.upper(),
            start=args.start,
            end=args.end,
            config=options_momentum_config,
        )
        print_options_momentum_ticker_plan(analysis)
        if args.save_reports:
            save_options_momentum_reports(analysis=analysis, output_dir=args.output_dir)
        if args.journal:
            update_paper_trading_journal(results=[analysis["result"]], output_dir=args.output_dir)
        return

    if args.strategy == "rsi-bollinger-v2":
        base_rsi_bollinger_v2_config = _rsi_bollinger_v2_config(args)

        if args.scan is not None:
            tickers, profile_label = _resolve_scan_request(args)
            _print_resolved_watchlist(profile_label, tickers)
            results = scan_watchlist_rsi_bollinger_v2(
                tickers=tickers,
                start=args.start,
                end=args.end,
                config=base_rsi_bollinger_v2_config,
            )
            print_rsi_bollinger_v2_scan_results(results)
            save_rsi_bollinger_v2_scan_results(results, output_dir=args.output_dir)
            return

        label, raw_df = load_input_data(args)
        profile_used, rsi_bollinger_v2_config = resolve_ticker_config(
            ticker=label,
            config=base_rsi_bollinger_v2_config,
        )
        bt = RsiBollingerV2Backtester(config=rsi_bollinger_v2_config)
        strategy_df, equity_df, trades_df, signals_df = bt.run(raw_df)

        summary = rsi_bollinger_v2_performance_summary(
            equity_df=equity_df,
            trades_df=trades_df,
            initial_cash=rsi_bollinger_v2_config.initial_cash,
        )
        print_rsi_bollinger_v2_summary(summary)
        print(f"Profile Used: {profile_used}")
        print_rsi_bollinger_v2_latest_signal(label, signals_df)

        print("\nRecent Trades")
        print("-------------")
        if trades_df.empty:
            print("No trades generated with current rules.")
        else:
            print(trades_df.tail(20).to_string(index=False))

        if args.save_reports:
            save_rsi_bollinger_v2_reports(
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
                    initial_cash=rsi_bollinger_v2_config.initial_cash,
                )
                plot_rsi_bollinger_v2_results(
                    label=label,
                    price_df=strategy_df,
                    signals_df=signals_df,
                    equity_df=equity_df,
                    bh_df=bh_df,
                    take_profit_pct=rsi_bollinger_v2_config.take_profit,
                )
            except Exception as exc:
                print(f"Skipping buy-and-hold chart: {exc}")

        return

    if args.strategy == "rsi-bollinger":
        rsi_bollinger_config = _rsi_bollinger_config(args)

        if args.scan is not None:
            tickers, profile_label = _resolve_scan_request(args)
            _print_resolved_watchlist(profile_label, tickers)
            results = scan_watchlist_rsi_bollinger(
                tickers=tickers,
                start=args.start,
                end=args.end,
                config=rsi_bollinger_config,
            )
            print_rsi_bollinger_scan_results(results)
            save_rsi_bollinger_scan_results(results, output_dir=args.output_dir)
            return

        label, raw_df = load_input_data(args)
        bt = RsiBollingerBacktester(config=rsi_bollinger_config)
        strategy_df, equity_df, trades_df, signals_df = bt.run(raw_df)

        summary = rsi_bollinger_performance_summary(
            equity_df=equity_df,
            trades_df=trades_df,
            initial_cash=rsi_bollinger_config.initial_cash,
        )
        print_rsi_bollinger_summary(summary)
        print_rsi_bollinger_latest_signal(label, signals_df)

        print("\nRecent Trades")
        print("-------------")
        if trades_df.empty:
            print("No trades generated with current rules.")
        else:
            print(trades_df.tail(20).to_string(index=False))

        if args.save_reports:
            save_rsi_bollinger_reports(
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
                    initial_cash=rsi_bollinger_config.initial_cash,
                )
                plot_rsi_bollinger_results(
                    label=label,
                    price_df=strategy_df,
                    signals_df=signals_df,
                    equity_df=equity_df,
                    bh_df=bh_df,
                    take_profit_pct=rsi_bollinger_config.take_profit,
                )
            except Exception as exc:
                print(f"Skipping buy-and-hold chart: {exc}")

        return

    if args.strategy == "four-hour-trend":
        four_hour_config = _four_hour_config(args)

        if args.scan is not None:
            tickers, profile_label = _resolve_scan_request(args)
            _print_resolved_watchlist(profile_label, tickers)
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

        if args.scan is not None:
            tickers, profile_label = _resolve_scan_request(args)
            _print_resolved_watchlist(profile_label, tickers)
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

    if args.scan is not None:
        tickers, profile_label = _resolve_scan_request(args)
        _print_resolved_watchlist(profile_label, tickers)
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
