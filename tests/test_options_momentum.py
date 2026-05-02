from pathlib import Path

import pandas as pd

import algo_backtester.backtests.options_momentum_backtester as options_momentum_module
from algo_backtester.journal import TRADE_JOURNAL_SHEET, update_paper_trading_journal
from algo_backtester.options_utils import build_long_call_plan
from algo_backtester.reports.options_momentum_report import print_scan_results, print_ticker_plan, save_reports
from algo_backtester.strategies.options_momentum import SourceSignalSnapshot
from algo_backtester.watchlists import get_default_watchlist_for_strategy, get_watchlist


def _qualified_snapshot(source_strategy: str = "four-hour-trend") -> SourceSignalSnapshot:
    return SourceSignalSnapshot(
        ticker="NVDA",
        source_strategy=source_strategy,
        signal="BUY",
        setup="ACTIONABLE",
        price=200.0,
        rsi=55.0,
        atr=6.0,
        signal_date="2026-05-01",
        reason="Qualified source signal.",
        trend_quality=True,
        volume_confirmed=True,
        earnings_risk=False,
        acceptable_volatility=True,
        score=4.5,
        qualified=True,
        notes="Qualified for options overlay.",
    )


def _rejected_snapshot(source_strategy: str = "ema-rsi") -> SourceSignalSnapshot:
    return SourceSignalSnapshot(
        ticker="NVDA",
        source_strategy=source_strategy,
        signal="HOLD",
        setup="WAIT",
        price=200.0,
        rsi=62.0,
        atr=6.0,
        signal_date="2026-05-01",
        reason="No actionable setup.",
        trend_quality=True,
        volume_confirmed=False,
        earnings_risk=False,
        acceptable_volatility=True,
        score=2.0,
        qualified=False,
        notes="Source signal is not BUY. | Volume confirmation failed.",
    )


def test_options_momentum_profile_loads():
    assert get_watchlist("options_momentum_core") == ["SPY", "QQQ", "NVDA", "AVGO", "META", "AAPL", "MSFT"]
    assert get_default_watchlist_for_strategy("options-momentum") == get_watchlist("options_momentum_core")


def test_options_plan_generated():
    plan = build_long_call_plan(
        ticker="NVDA",
        source_strategy="four-hour-trend",
        price=200.0,
        atr=6.0,
        signal_date="2026-05-01",
    )

    assert plan.option_type == "CALL"
    assert 14 <= plan.dte <= 30
    assert 0.60 <= plan.delta_target <= 0.70
    assert plan.estimated_premium > 0
    assert plan.max_loss == round(plan.estimated_premium * 100, 2)


def test_options_momentum_scan_runs(monkeypatch):
    monkeypatch.setattr(
        options_momentum_module,
        "evaluate_source_strategies",
        lambda **kwargs: [_rejected_snapshot("ema-rsi"), _qualified_snapshot("four-hour-trend")],
    )

    result = options_momentum_module.scan_ticker("NVDA")

    assert result["Strategy"] == "options-momentum"
    assert result["SourceStrategy"] == "four-hour-trend"
    assert result["Signal"] == "BUY"
    assert result["OptionType"] == "CALL"
    assert result["EstPremium"] > 0


def test_options_momentum_source_strategy_filtering(monkeypatch):
    monkeypatch.setattr(
        options_momentum_module,
        "evaluate_source_strategies",
        lambda **kwargs: [_rejected_snapshot("ema-rsi"), _rejected_snapshot("four-hour-trend")],
    )

    result = options_momentum_module.scan_ticker("NVDA")

    assert result["Signal"] == "HOLD"
    assert result["EstPremium"] == 0.0
    assert result["OptionsAction"] == "NO_OPTIONS_TRADE"


def test_options_momentum_report_output(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        options_momentum_module,
        "evaluate_source_strategies",
        lambda **kwargs: [_rejected_snapshot("ema-rsi"), _qualified_snapshot("four-hour-trend")],
    )
    analysis = options_momentum_module.analyze_ticker("NVDA")

    print_ticker_plan(analysis)
    print_scan_results([analysis["result"]])
    save_reports(analysis=analysis, output_dir=str(tmp_path))

    captured = capsys.readouterr()
    assert "Source Signal Summary" in captured.out
    assert "Options Momentum Watchlist Scan" in captured.out
    assert (tmp_path / "options_momentum" / "NVDA_options_plan.csv").exists()
    assert (tmp_path / "options_momentum" / "NVDA_source_signals.csv").exists()


def test_options_momentum_journal_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        options_momentum_module,
        "evaluate_source_strategies",
        lambda **kwargs: [_rejected_snapshot("ema-rsi"), _qualified_snapshot("four-hour-trend")],
    )
    result = options_momentum_module.scan_ticker("NVDA")

    workbook_path = update_paper_trading_journal(results=[result], output_dir=str(tmp_path))
    trade_df = pd.read_excel(workbook_path, sheet_name=TRADE_JOURNAL_SHEET)

    assert len(trade_df) == 1
    assert trade_df.iloc[0]["Options Structure"] == f"Long Call {result['Strike']:.2f}C"
