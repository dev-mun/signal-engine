import importlib.util
from pathlib import Path

import pandas as pd

import algo_backtester.cli as cli_module
import algo_backtester.config_loader as config_loader_module
from algo_backtester.backtester import TrendPullbackBacktester
from algo_backtester.backtests.ema_rsi_backtester import EmaRsiPullbackBacktester
from algo_backtester.backtests.four_hour_trend_backtester import FourHourTrendBacktester, make_demo_intraday_data
import algo_backtester.backtests.rsi_bollinger_v2_backtester as rsi_bollinger_v2_module
from algo_backtester.backtests.rsi_bollinger_backtester import RsiBollingerBacktester
from algo_backtester.backtests.rsi_bollinger_v2_backtester import (
    RsiBollingerV2BacktestConfig,
    RsiBollingerV2Backtester,
    resolve_ticker_config,
    run_parameter_sweep,
)
from algo_backtester.data_loader import make_demo_data, validate_ohlcv
from algo_backtester.reports.rsi_bollinger_v2_report import latest_signal, print_scan_results, save_reports
from algo_backtester.strategies.rsi_bollinger_v2 import (
    add_indicators,
    close_position_in_range,
    should_buy,
    trend_quality_passes,
)


def _load_sweep_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sweep_rsi_bollinger_v2.py"
    spec = importlib.util.spec_from_file_location("sweep_rsi_bollinger_v2", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rsi_bollinger_v2_indicators_are_created():
    raw_df = make_demo_data(rows=500)
    df = add_indicators(validate_ohlcv(raw_df))

    assert not df.empty
    assert "RSI14" in df.columns
    assert "ATR14" in df.columns
    assert "BB_MIDDLE" in df.columns
    assert "BB_UPPER" in df.columns
    assert "BB_LOWER" in df.columns


def test_rsi_bollinger_v2_defaults_are_updated():
    config = RsiBollingerV2BacktestConfig()

    assert config.take_profit == 0.05
    assert config.trailing_stop == 0.04
    assert config.rsi_threshold == 38.0
    assert config.band_tolerance == 1.02
    assert config.close_position_min == 0.35


def test_rsi_bollinger_v2_backtest_returns_expected_frames():
    raw_df = make_demo_data(rows=500)
    bt = RsiBollingerV2Backtester()

    _, equity_df, trades_df, signals_df = bt.run(raw_df)

    assert not equity_df.empty
    assert not signals_df.empty
    assert "Equity" in equity_df.columns
    assert "Signal" in signals_df.columns
    assert trades_df is not None


def test_rsi_bollinger_v2_latest_signal_has_expected_fields():
    raw_df = make_demo_data(rows=500)
    bt = RsiBollingerV2Backtester()

    _, _, _, signals_df = bt.run(raw_df)
    signal = latest_signal(signals_df)

    assert "Signal" in signal
    assert "Reason" in signal
    assert "Close" in signal
    assert "BB_LOWER" in signal
    assert signal["Signal"] in {"BUY", "SELL", "HOLD", "HOLD_POSITION"}


def test_trend_quality_filter_requires_ema_stack_and_location():
    row = pd.Series(
        {
            "Close": 101.0,
            "EMA50": 100.0,
            "EMA200": 99.0,
        }
    )
    weak_row = pd.Series(
        {
            "Close": 101.0,
            "EMA50": 98.0,
            "EMA200": 99.0,
        }
    )

    assert trend_quality_passes(row)
    assert not trend_quality_passes(weak_row)


def test_close_position_min_filter_blocks_falling_knife():
    row = pd.Series(
        {
            "Open": 98.0,
            "High": 100.0,
            "Low": 90.0,
            "Close": 92.0,
            "EMA50": 95.0,
            "EMA200": 90.0,
            "RSI": 34.0,
            "BB_MIDDLE": 101.0,
            "BB_LOWER": 91.0,
            "Volume": 100.0,
            "AVG_VOL20": 100.0,
        }
    )
    prev = pd.Series({"Close": 91.0})

    assert not close_position_in_range(row, close_position_min=0.35)
    buy_now, _ = should_buy(
        row=row,
        prev=prev,
        rsi_threshold=38.0,
        volume_multiplier=0.6,
        band_tolerance=1.02,
        close_position_min=0.35,
        require_confirmation=False,
    )
    assert not buy_now


def test_rsi_bollinger_v2_scan_result_includes_strategy(monkeypatch):
    raw_df = make_demo_data(rows=500)

    monkeypatch.setattr(rsi_bollinger_v2_module, "load_yfinance_data", lambda **kwargs: raw_df)

    result = rsi_bollinger_v2_module.scan_ticker(ticker="SPY")

    assert result["Ticker"] == "SPY"
    assert result["Strategy"] == "rsi-bollinger-v2"
    assert result["Profile"] == "default"
    assert "Setup" in result
    assert "Reason" in result


def test_rsi_bollinger_v2_reports_save_to_separate_directory(tmp_path: Path):
    raw_df = make_demo_data(rows=500)
    bt = RsiBollingerV2Backtester()
    _, equity_df, trades_df, signals_df = bt.run(raw_df)

    save_reports(
        label="SPY",
        equity_df=equity_df,
        trades_df=trades_df,
        signals_df=signals_df,
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "rsi_bollinger_v2" / "SPY_equity.csv").exists()
    assert (tmp_path / "rsi_bollinger_v2" / "SPY_trades.csv").exists()
    assert not (tmp_path / "rsi_bollinger" / "SPY_equity.csv").exists()


def test_existing_strategies_still_import_and_run():
    daily_demo = make_demo_data(rows=500)

    _, trend_equity_df, _, _ = TrendPullbackBacktester().run(daily_demo)
    _, ema_equity_df, _, _ = EmaRsiPullbackBacktester().run(daily_demo)
    _, four_hour_equity_df, _, _ = FourHourTrendBacktester().run(make_demo_intraday_data(rows=1400))
    _, v1_equity_df, _, _ = RsiBollingerBacktester().run(daily_demo)

    assert not trend_equity_df.empty
    assert not ema_equity_df.empty
    assert not four_hour_equity_df.empty
    assert not v1_equity_df.empty


def test_sweep_script_defaults_to_multi_ticker_universe(monkeypatch):
    sweep_module = _load_sweep_script_module()
    monkeypatch.setattr("sys.argv", ["sweep_rsi_bollinger_v2.py"])

    args = sweep_module.parse_args()

    tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
    assert tickers == sweep_module.DEFAULT_SWEEP_UNIVERSE


def test_rank_sweep_results_prioritizes_requested_filters():
    sweep_module = _load_sweep_script_module()
    input_df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "params": "set-1",
                "total_return": 12.0,
                "Sharpe": 0.55,
                "max_drawdown": -8.0,
                "completed_trades": 24,
                "trades_per_year": 4.0,
                "win_rate": 60.0,
                "profit_factor": 1.4,
            },
            {
                "ticker": "BBB",
                "params": "set-2",
                "total_return": 20.0,
                "Sharpe": 0.7,
                "max_drawdown": -7.0,
                "completed_trades": 30,
                "trades_per_year": 5.0,
                "win_rate": 58.0,
                "profit_factor": 1.2,
            },
            {
                "ticker": "CCC",
                "params": "set-3",
                "total_return": 5.0,
                "Sharpe": 0.3,
                "max_drawdown": -9.0,
                "completed_trades": 22,
                "trades_per_year": 3.2,
                "win_rate": 55.0,
                "profit_factor": 1.35,
            },
        ]
    )

    ranked_df = sweep_module.rank_sweep_results(input_df)

    assert ranked_df.iloc[0]["ticker"] == "AAA"
    assert bool(ranked_df.iloc[0]["meets_profit_factor"])
    assert bool(ranked_df.iloc[0]["meets_sharpe"])
    assert int(ranked_df.iloc[0]["rank"]) == 1
    assert bool(ranked_df.iloc[0]["meets_completed_trades"])


def test_parameter_sweep_includes_close_position_min():
    raw_df = make_demo_data(rows=260)
    original_product = rsi_bollinger_v2_module.product
    rsi_bollinger_v2_module.product = lambda *args: original_product([35], [0.03], [0.04], [0.03], [5], [0.5], [1.01], [0.30, 0.35, 0.40])

    try:
        sweep_df = run_parameter_sweep(ticker="SPY", raw_df=raw_df)
    finally:
        rsi_bollinger_v2_module.product = original_product

    assert not sweep_df.empty
    assert "close_position_min" in sweep_df.columns
    assert set(sweep_df["close_position_min"].unique()) == {0.3, 0.35, 0.4}


def test_parameter_sweep_accepts_custom_grid():
    raw_df = make_demo_data(rows=260)

    sweep_df = run_parameter_sweep(
        ticker="SPY",
        raw_df=raw_df,
        sweep_grid={
            "rsi_threshold": [38, 40],
            "stop_loss": [0.04],
            "take_profit": [0.05],
            "trailing_stop": [0.04],
            "max_hold_days": [7],
            "volume_multiplier": [0.6],
            "band_tolerance": [1.02],
            "close_position_min": [0.35],
        },
    )

    assert len(sweep_df) == 2
    assert set(sweep_df["rsi_threshold"].unique()) == {38, 40}
    assert set(sweep_df["stop_loss"].unique()) == {0.04}


def test_load_rsi_bollinger_v2_profiles_reads_config_file(tmp_path: Path):
    config_path = tmp_path / "rsi_bollinger_v2_profiles.json"
    config_path.write_text(
        """
        {
          "default": {
            "rsi_threshold": 38,
            "stop_loss": 0.04,
            "take_profit": 0.05,
            "trailing_stop": 0.04,
            "max_hold_days": 7,
            "volume_multiplier": 0.6,
            "band_tolerance": 1.02,
            "close_position_min": 0.35
          },
          "SMCI": {
            "rsi_threshold": 40,
            "stop_loss": 0.05,
            "take_profit": 0.06,
            "trailing_stop": 0.05,
            "max_hold_days": 5,
            "volume_multiplier": 0.8,
            "band_tolerance": 1.01,
            "close_position_min": 0.30
          }
        }
        """,
        encoding="utf-8",
    )

    profiles = config_loader_module.load_rsi_bollinger_v2_profiles(config_path=config_path)

    assert "default" in profiles
    assert "SMCI" in profiles
    assert profiles["SMCI"]["rsi_threshold"] == 40


def test_rsi_bollinger_v2_profile_default_fallback_works():
    profile_name, config = resolve_ticker_config("SPY")

    assert profile_name == "default"
    assert config.rsi_threshold == 38.0
    assert config.take_profit == 0.05


def test_rsi_bollinger_v2_profile_ticker_override_works():
    profile_name, config = resolve_ticker_config("NVDA")

    assert profile_name == "NVDA"
    assert config.take_profit == 0.04
    assert config.max_hold_days == 10
    assert config.volume_multiplier == 0.8


def test_rsi_bollinger_v2_missing_config_falls_back_cleanly(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "missing_profiles.json"
    monkeypatch.setattr(config_loader_module, "DEFAULT_RSI_BOLLINGER_V2_PROFILES_PATH", config_path)

    profiles = config_loader_module.load_rsi_bollinger_v2_profiles()
    profile = config_loader_module.get_rsi_bollinger_v2_profile("META")

    assert "default" in profiles
    assert profile["take_profit"] == 0.06
    assert profile["close_position_min"] == 0.30


def test_rsi_bollinger_v2_scan_output_shows_profile_column(capsys):
    print_scan_results(
        [
            {
                "Ticker": "SPY",
                "Strategy": "rsi-bollinger-v2",
                "Profile": "default",
                "Signal": "HOLD",
                "Setup": "WAIT",
                "Price": 500.0,
                "RSI": 45.0,
                "ATR": 5.0,
                "Distance": "Waiting for lower band test",
                "Reason": "No actionable RSI Bollinger V2 setup.",
            }
        ]
    )

    captured = capsys.readouterr()
    assert "Profile" in captured.out
    assert "default" in captured.out


def test_rsi_bollinger_v2_single_ticker_output_shows_profile_used(monkeypatch, capsys):
    raw_df = make_demo_data(rows=500)
    monkeypatch.setattr(cli_module, "load_yfinance_data", lambda **kwargs: raw_df)
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--ticker", "NVDA", "--strategy", "rsi-bollinger-v2", "--no-plot"],
    )

    cli_module.main()

    captured = capsys.readouterr()
    assert "Profile Used: NVDA" in captured.out
