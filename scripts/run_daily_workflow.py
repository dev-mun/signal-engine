from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algo_backtester.backtests.ema_rsi_backtester import EmaRsiBacktestConfig, scan_watchlist as scan_watchlist_ema_rsi
from algo_backtester.backtests.four_hour_trend_backtester import FourHourTrendConfig, scan_watchlist as scan_watchlist_four_hour
from algo_backtester.backtests.rsi_bollinger_v2_backtester import (
    RsiBollingerV2BacktestConfig,
    scan_watchlist as scan_watchlist_rsi_bollinger_v2,
)
from algo_backtester.backtests.swing_options_debit_spread_backtester import (
    SwingOptionsDebitSpreadConfig,
    scan_watchlist as scan_watchlist_swing_options_debit_spread,
)
from algo_backtester.journal import update_paper_trading_journal
from algo_backtester.reports.daily_summary_report import (
    build_daily_summary,
    save_daily_summary,
    select_top_setup,
)
from algo_backtester.reports.ema_rsi_report import print_scan_results as print_ema_rsi_scan_results
from algo_backtester.reports.ema_rsi_report import save_scan_results as save_ema_rsi_scan_results
from algo_backtester.reports.four_hour_report import print_scan_results as print_four_hour_scan_results
from algo_backtester.reports.four_hour_report import save_scan_results as save_four_hour_scan_results
from algo_backtester.reports.rsi_bollinger_v2_report import print_scan_results as print_rsi_bollinger_v2_scan_results
from algo_backtester.reports.rsi_bollinger_v2_report import save_scan_results as save_rsi_bollinger_v2_scan_results
from algo_backtester.reports.swing_options_debit_spread_report import (
    print_scan_results as print_swing_options_debit_spread_scan_results,
)
from algo_backtester.reports.swing_options_debit_spread_report import (
    save_scan_results as save_swing_options_debit_spread_scan_results,
)
from algo_backtester.watchlists import DEFAULT_STRATEGY_PROFILES, get_default_watchlist_for_strategy, get_watchlist

SUPPORTED_STRATEGIES = [
    "ema-rsi",
    "four-hour-trend",
    "rsi-bollinger-v2",
    "swing-options-debit-spread",
]
LARGE_CAP_DEBIT_PROFILE = "small_account_debit_spreads"
GROWTH_DEBIT_PROFILE = "small_account_growth"
DEBIT_PROFILE_RUNS = [LARGE_CAP_DEBIT_PROFILE, GROWTH_DEBIT_PROFILE]

SCAN_RELATIVE_PATHS = {
    "ema-rsi": "ema_rsi/watchlist_scan_{date}.csv",
    "four-hour-trend": "four_hour/watchlist_scan_{date}.csv",
    "rsi-bollinger-v2": "rsi_bollinger_v2/watchlist_scan_{date}.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily scan workflow and generate a plain-English summary.")
    parser.add_argument("--date", type=str, default=str(date.today()))
    parser.add_argument("--no-journal", action="store_true")
    parser.add_argument("--auto-open-summary", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--skip-strategy-on-error", action="store_true")
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(SUPPORTED_STRATEGIES),
        help="Comma-separated strategy list.",
    )
    parser.add_argument("--output-dir", type=str, default="reports")
    return parser.parse_args()


def _parse_strategies(value: str) -> list[str]:
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [strategy for strategy in strategies if strategy not in SUPPORTED_STRATEGIES]
    if invalid:
        valid = ", ".join(SUPPORTED_STRATEGIES)
        raise ValueError(f"Unsupported strategies: {', '.join(invalid)}. Valid choices: {valid}")
    return strategies


def _ema_config() -> EmaRsiBacktestConfig:
    return EmaRsiBacktestConfig(
        initial_cash=10_000.0,
        stop_loss=0.07,
        take_profit=0.15,
        trailing_stop=0.08,
        max_hold_days=45,
        risk_per_trade=0.01,
        atr_multiple=2.0,
    )


def _four_hour_config() -> FourHourTrendConfig:
    return FourHourTrendConfig(
        initial_cash=10_000.0,
        stop_loss=0.04,
        take_profit=0.08,
        trailing_stop=0.05,
        max_hold_candles=12,
        risk_per_trade=0.0075,
        atr_multiple=1.5,
        interval="1h",
    )


def _rsi_bollinger_v2_config() -> RsiBollingerV2BacktestConfig:
    return RsiBollingerV2BacktestConfig(
        initial_cash=10_000.0,
        stop_loss=0.04,
        take_profit=0.04,
        trailing_stop=0.03,
        max_hold_days=7,
        risk_per_trade=0.005,
        atr_multiple=1.25,
        rsi_threshold=42.0,
        volume_multiplier=0.6,
        band_tolerance=1.03,
        require_confirmation=False,
    )


def _payload_key(strategy: str, profile: str | None = None) -> str:
    if strategy == "swing-options-debit-spread" and profile:
        return f"{strategy}:{profile}"
    return strategy


def _save_workflow_scan(strategy: str, report_date: str, results: list[dict], output_dir: str, profile: str | None = None) -> Path:
    if strategy == "swing-options-debit-spread" and profile:
        file_path = Path(output_dir) / "swing_options_debit_spread" / f"scan_{profile}_{report_date}.csv"
    else:
        template = SCAN_RELATIVE_PATHS[strategy]
        file_path = Path(output_dir) / template.format(date=report_date)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(file_path, index=False)
    return file_path


def _scan_strategy(strategy: str, output_dir: str, report_date: str, profile: str | None = None) -> dict[str, dict]:
    effective_profile = profile or DEFAULT_STRATEGY_PROFILES[strategy]
    tickers = get_watchlist(effective_profile) if profile is not None else get_default_watchlist_for_strategy(strategy)

    if strategy == "ema-rsi":
        results = scan_watchlist_ema_rsi(tickers=tickers, config=_ema_config())
        print_ema_rsi_scan_results(results)
        save_ema_rsi_scan_results(results, output_dir=output_dir)
    elif strategy == "four-hour-trend":
        results = scan_watchlist_four_hour(tickers=tickers, interval="1h", config=_four_hour_config())
        print_four_hour_scan_results(results)
        save_four_hour_scan_results(results, output_dir=output_dir)
    elif strategy == "rsi-bollinger-v2":
        results = scan_watchlist_rsi_bollinger_v2(tickers=tickers, config=_rsi_bollinger_v2_config())
        print_rsi_bollinger_v2_scan_results(results)
        save_rsi_bollinger_v2_scan_results(results, output_dir=output_dir)
    elif strategy == "swing-options-debit-spread":
        results = scan_watchlist_swing_options_debit_spread(tickers=tickers, config=SwingOptionsDebitSpreadConfig())
        print_swing_options_debit_spread_scan_results(results)
        _save_workflow_scan(strategy=strategy, report_date=report_date, results=results, output_dir=output_dir, profile=effective_profile)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if strategy == "swing-options-debit-spread":
        return {
            _payload_key(strategy, effective_profile): {
                "strategy": strategy,
                "profile": effective_profile,
                "tickers": tickers,
                "results": results,
            }
        }

    return {
        _payload_key(strategy): {
            "strategy": strategy,
            "profile": effective_profile,
            "tickers": tickers,
            "results": results,
        }
    }


def _scan_report_path(strategy: str, report_date: str, output_dir: str, profile: str | None = None) -> Path:
    if strategy == "swing-options-debit-spread" and profile:
        return Path(output_dir) / "swing_options_debit_spread" / f"scan_{profile}_{report_date}.csv"
    template = SCAN_RELATIVE_PATHS[strategy]
    return Path(output_dir) / template.format(date=report_date)


def _load_existing_scan_payload(strategies: list[str], report_date: str, output_dir: str) -> dict[str, dict]:
    payload: dict[str, dict] = {}

    for strategy in strategies:
        if strategy == "swing-options-debit-spread":
            for profile in DEBIT_PROFILE_RUNS:
                file_path = _scan_report_path(strategy=strategy, report_date=report_date, output_dir=output_dir, profile=profile)
                if not file_path.exists():
                    raise FileNotFoundError("Missing scan reports. Run full workflow first.")

                df = pd.read_csv(file_path)
                payload[_payload_key(strategy, profile)] = {
                    "strategy": strategy,
                    "profile": profile,
                    "tickers": get_watchlist(profile),
                    "results": df.to_dict(orient="records"),
                }
            continue

        file_path = _scan_report_path(strategy=strategy, report_date=report_date, output_dir=output_dir)
        if not file_path.exists():
            raise FileNotFoundError("Missing scan reports. Run full workflow first.")

        df = pd.read_csv(file_path)
        payload[_payload_key(strategy)] = {
            "strategy": strategy,
            "profile": DEFAULT_STRATEGY_PROFILES[strategy],
            "tickers": get_default_watchlist_for_strategy(strategy),
            "results": df.to_dict(orient="records"),
        }

    return payload


def _top_setup_line(scan_payload: dict[str, dict]) -> str:
    top_setup = select_top_setup(scan_payload)
    if top_setup is None:
        return "None"
    return str(top_setup["display"])


def _open_summary(markdown_path: Path) -> None:
    try:
        subprocess.run(["open", str(markdown_path)], check=True)
    except Exception:
        print(f"Unable to auto-open summary. Markdown remains at: {markdown_path}")


def _print_terminal_summary(paths: dict[str, Path], summary: dict, scan_payload: dict[str, dict]) -> None:
    actionable_count = int(summary["actionable_count"])
    regime = str(summary.get("market_regime", {}).get("regime", "UNKNOWN") or "UNKNOWN")
    print("\nDaily Workflow Summary")
    print("----------------------")
    print(f"Markdown: {paths['markdown']}")
    print(f"Top Decision: {summary['executive_decision']}")
    print(f"Actionable Count: {actionable_count}")
    print(f"Market Regime: {regime}")
    print(f"Large-Cap Debit Profile: {LARGE_CAP_DEBIT_PROFILE}")
    print(f"Growth Debit Profile: {GROWTH_DEBIT_PROFILE}")
    print(f"Top Setup: {_top_setup_line(scan_payload)}")
    if actionable_count == 0:
        print("Decision: No trade. Re-run after next market close.")
    else:
        print("Decision: Review candidate before market open. Do not place order unless live chain confirms.")


def run_workflow(
    *,
    report_date: str,
    strategies: list[str],
    output_dir: str,
    no_journal: bool,
    summary_only: bool,
    skip_strategy_on_error: bool,
) -> tuple[dict[str, dict], list[dict]]:
    failures: list[dict] = []

    if summary_only:
        return _load_existing_scan_payload(strategies=strategies, report_date=report_date, output_dir=output_dir), failures

    scan_payload: dict[str, dict] = {}
    for strategy in strategies:
        try:
            if strategy == "swing-options-debit-spread":
                for profile in DEBIT_PROFILE_RUNS:
                    payloads = _scan_strategy(
                        strategy=strategy,
                        output_dir=output_dir,
                        report_date=report_date,
                        profile=profile,
                    )
                    for payload_key, payload in payloads.items():
                        scan_payload[payload_key] = payload
                        if not no_journal:
                            update_paper_trading_journal(results=payload["results"], output_dir=output_dir)
            else:
                payloads = _scan_strategy(strategy=strategy, output_dir=output_dir, report_date=report_date)
                for payload_key, payload in payloads.items():
                    scan_payload[payload_key] = payload
                    if not no_journal:
                        update_paper_trading_journal(results=payload["results"], output_dir=output_dir)
        except Exception as exc:
            if not skip_strategy_on_error:
                raise
            failures.append({"strategy": strategy, "error": str(exc)})

    return scan_payload, failures


def main() -> None:
    args = parse_args()
    strategies = _parse_strategies(args.strategies)

    try:
        scan_payload, failures = run_workflow(
            report_date=args.date,
            strategies=strategies,
            output_dir=args.output_dir,
            no_journal=args.no_journal,
            summary_only=args.summary_only,
            skip_strategy_on_error=args.skip_strategy_on_error,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    summary = build_daily_summary(scan_payload=scan_payload, report_date=args.date, failures=failures)
    paths = save_daily_summary(summary=summary, output_dir=args.output_dir)
    _print_terminal_summary(paths=paths, summary=summary, scan_payload=scan_payload)

    if args.auto_open_summary:
        _open_summary(paths["markdown"])


if __name__ == "__main__":
    main()
