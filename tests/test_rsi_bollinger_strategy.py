from pathlib import Path

from algo_backtester.backtester import TrendPullbackBacktester
from algo_backtester.backtests.ema_rsi_backtester import EmaRsiPullbackBacktester
import algo_backtester.backtests.rsi_bollinger_backtester as rsi_bollinger_module
from algo_backtester.backtests.four_hour_trend_backtester import FourHourTrendBacktester, make_demo_intraday_data
from algo_backtester.backtests.rsi_bollinger_backtester import RsiBollingerBacktester
from algo_backtester.data_loader import make_demo_data, validate_ohlcv
from algo_backtester.reports.rsi_bollinger_report import latest_signal, save_reports
from algo_backtester.strategies.rsi_bollinger_mean_reversion import add_indicators


def test_rsi_bollinger_indicators_are_created():
    raw_df = make_demo_data(rows=500)
    df = add_indicators(validate_ohlcv(raw_df))

    assert not df.empty
    assert "RSI14" in df.columns
    assert "ATR14" in df.columns
    assert "BB_MIDDLE" in df.columns
    assert "BB_UPPER" in df.columns
    assert "BB_LOWER" in df.columns
    assert "EMA50" in df.columns
    assert "EMA200" in df.columns


def test_rsi_bollinger_backtest_returns_expected_frames():
    raw_df = make_demo_data(rows=500)
    bt = RsiBollingerBacktester()

    _, equity_df, trades_df, signals_df = bt.run(raw_df)

    assert not equity_df.empty
    assert not signals_df.empty
    assert "Equity" in equity_df.columns
    assert "Signal" in signals_df.columns
    assert trades_df is not None


def test_rsi_bollinger_latest_signal_has_expected_fields():
    raw_df = make_demo_data(rows=500)
    bt = RsiBollingerBacktester()

    _, _, _, signals_df = bt.run(raw_df)
    signal = latest_signal(signals_df)

    assert "Signal" in signal
    assert "Reason" in signal
    assert "Close" in signal
    assert "BB_LOWER" in signal
    assert signal["Signal"] in {"BUY", "SELL", "HOLD", "HOLD_POSITION"}


def test_rsi_bollinger_scan_result_includes_strategy(monkeypatch):
    raw_df = make_demo_data(rows=500)

    monkeypatch.setattr(rsi_bollinger_module, "load_yfinance_data", lambda **kwargs: raw_df)

    result = rsi_bollinger_module.scan_ticker(ticker="SPY")

    assert result["Ticker"] == "SPY"
    assert result["Strategy"] == "rsi-bollinger"
    assert "Setup" in result
    assert "Reason" in result


def test_rsi_bollinger_reports_save_to_separate_directory(tmp_path: Path):
    raw_df = make_demo_data(rows=500)
    bt = RsiBollingerBacktester()
    _, equity_df, trades_df, signals_df = bt.run(raw_df)

    save_reports(
        label="SPY",
        equity_df=equity_df,
        trades_df=trades_df,
        signals_df=signals_df,
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "rsi_bollinger" / "SPY_equity.csv").exists()
    assert (tmp_path / "rsi_bollinger" / "SPY_trades.csv").exists()
    assert not (tmp_path / "SPY_equity.csv").exists()


def test_existing_strategies_still_import_and_run():
    daily_demo = make_demo_data(rows=500)

    _, trend_equity_df, _, _ = TrendPullbackBacktester().run(daily_demo)
    _, ema_equity_df, _, _ = EmaRsiPullbackBacktester().run(daily_demo)
    _, four_hour_equity_df, _, _ = FourHourTrendBacktester().run(make_demo_intraday_data(rows=1400))

    assert not trend_equity_df.empty
    assert not ema_equity_df.empty
    assert not four_hour_equity_df.empty
