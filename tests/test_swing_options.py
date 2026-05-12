from pathlib import Path

import pandas as pd

import algo_backtester.backtests.swing_options_backtester as swing_options_module
from algo_backtester.journal import TRADE_JOURNAL_SHEET, update_paper_trading_journal
from algo_backtester.reports.swing_options_report import print_scan_results, print_ticker_plan, save_reports
from algo_backtester.backtests.swing_options_backtester import _finalize_signal_conversion, apply_execution_profile_labels
from algo_backtester.strategies.swing_options import SwingSourceSignal, build_long_call_plan
from algo_backtester.watchlists import get_default_watchlist_for_strategy, get_watchlist


def _context(close: float, ema20: float, ema50: float, ema200: float, rsi: float, atr: float, volume: float, avg_volume: float, signal_date: str = "2026-05-01") -> dict:
    latest = pd.Series(
        {
            "Close": close,
            "EMA20": ema20,
            "EMA50": ema50,
            "EMA200": ema200,
            "RSI": rsi,
            "ATR": atr,
            "Volume": volume,
            "AverageVolume20": avg_volume,
        }
    )
    prev = pd.Series({"Close": close - 2})
    return {"signal_date": signal_date, "latest": latest, "prev": prev}


def _actionable_evaluation() -> dict:
    return {
        "sources": [
            SwingSourceSignal("ema-rsi", "BUY", "ACTIONABLE", 60.0, 54.0, 1.8, True, True, True, True, "EMA RSI BUY"),
            SwingSourceSignal("four-hour-trend", "BUY", "ACTIONABLE", 60.5, 57.0, 1.6, True, True, True, True, "4H BUY"),
            SwingSourceSignal("rsi-bollinger-v2", "HOLD", "NEAR_SETUP", 59.5, 40.0, 1.7, True, True, True, True, "V2 recovery"),
        ],
        "contexts": {
            "ema-rsi": _context(close=60.0, ema20=58.5, ema50=55.0, ema200=50.0, rsi=54.0, atr=1.8, volume=2_000_000, avg_volume=1_000_000),
            "four-hour-trend": _context(close=60.5, ema20=59.0, ema50=56.0, ema200=51.0, rsi=57.0, atr=1.6, volume=1_800_000, avg_volume=1_000_000),
            "rsi-bollinger-v2": _context(close=59.5, ema20=58.0, ema50=55.0, ema200=50.0, rsi=40.0, atr=1.7, volume=1_700_000, avg_volume=1_000_000),
        },
    }


def _watchlist_evaluation() -> dict:
    return {
        "sources": [
            SwingSourceSignal("ema-rsi", "HOLD", "NEAR_SETUP", 200.0, 58.0, 6.0, True, True, True, True, "EMA supportive"),
            SwingSourceSignal("four-hour-trend", "BUY", "ACTIONABLE", 201.0, 56.0, 5.5, True, True, True, True, "4H BUY"),
            SwingSourceSignal("rsi-bollinger-v2", "HOLD", "WAIT", 199.0, 48.0, 5.6, True, True, True, False, "V2 neutral"),
        ],
        "contexts": {
            "ema-rsi": _context(close=200.0, ema20=196.0, ema50=188.0, ema200=175.0, rsi=58.0, atr=6.0, volume=2_000_000, avg_volume=1_000_000),
            "four-hour-trend": _context(close=201.0, ema20=197.0, ema50=189.0, ema200=176.0, rsi=56.0, atr=5.5, volume=1_800_000, avg_volume=1_000_000),
            "rsi-bollinger-v2": _context(close=199.0, ema20=195.0, ema50=188.0, ema200=174.0, rsi=48.0, atr=5.6, volume=1_700_000, avg_volume=1_000_000),
        },
    }


def _wait_evaluation() -> dict:
    return {
        "sources": [
            SwingSourceSignal("ema-rsi", "HOLD", "WAIT", 100.0, 50.0, 2.0, True, True, False, False, "Waiting"),
            SwingSourceSignal("four-hour-trend", "HOLD", "WAIT", 100.0, 49.0, 2.1, True, False, False, False, "Waiting"),
            SwingSourceSignal("rsi-bollinger-v2", "HOLD", "WAIT", 100.0, 47.0, 2.0, True, False, False, False, "Waiting"),
        ],
        "contexts": {
            "ema-rsi": _context(close=100.0, ema20=99.5, ema50=98.0, ema200=95.0, rsi=50.0, atr=2.0, volume=900_000, avg_volume=1_000_000),
            "four-hour-trend": _context(close=100.0, ema20=99.0, ema50=98.0, ema200=95.0, rsi=49.0, atr=2.1, volume=900_000, avg_volume=1_000_000),
            "rsi-bollinger-v2": _context(close=100.0, ema20=99.0, ema50=98.0, ema200=95.0, rsi=47.0, atr=2.0, volume=900_000, avg_volume=1_000_000),
        },
    }


def _avoid_evaluation() -> dict:
    return {
        "sources": [
            SwingSourceSignal("ema-rsi", "HOLD", "WEAK", 90.0, 33.0, 5.0, False, False, False, False, "Weak"),
            SwingSourceSignal("four-hour-trend", "HOLD", "WAIT", 90.0, 42.0, 5.2, False, False, False, False, "Weak"),
            SwingSourceSignal("rsi-bollinger-v2", "HOLD", "WEAK_TREND", 90.0, 45.0, 5.1, False, False, False, False, "Weak"),
        ],
        "contexts": {
            "ema-rsi": _context(close=90.0, ema20=92.0, ema50=91.0, ema200=95.0, rsi=33.0, atr=5.0, volume=500_000, avg_volume=1_000_000),
            "four-hour-trend": _context(close=90.0, ema20=91.0, ema50=90.0, ema200=94.0, rsi=42.0, atr=5.2, volume=500_000, avg_volume=1_000_000),
            "rsi-bollinger-v2": _context(close=90.0, ema20=91.0, ema50=90.0, ema200=94.0, rsi=45.0, atr=5.1, volume=500_000, avg_volume=1_000_000),
        },
    }


def _extended_evaluation() -> dict:
    data = _actionable_evaluation()
    data["contexts"]["ema-rsi"] = _context(close=70.0, ema20=63.0, ema50=58.0, ema200=52.0, rsi=74.0, atr=2.0, volume=2_000_000, avg_volume=1_000_000)
    return data


def _weak_trend_evaluation() -> dict:
    data = _actionable_evaluation()
    data["contexts"]["ema-rsi"] = _context(close=58.0, ema20=57.5, ema50=55.0, ema200=60.0, rsi=52.0, atr=1.8, volume=2_000_000, avg_volume=1_000_000)
    return data


def test_swing_options_watchlist_profile_loads():
    assert get_watchlist("swing_options_core") == ["SPY", "QQQ", "NVDA", "AVGO", "META", "AAPL", "MSFT", "AMD", "SLV"]
    assert get_watchlist("small_account_options") == ["SPY", "QQQ", "AAPL", "AMD", "SLV"]
    assert get_default_watchlist_for_strategy("swing-options") == get_watchlist("swing_options_core")


def test_score_calculation_and_actionable_classification(monkeypatch):
    monkeypatch.setattr(swing_options_module, "evaluate_source_signals", lambda **kwargs: _actionable_evaluation())
    analysis = swing_options_module.analyze_ticker(
        ticker="NVDA",
        config=swing_options_module.SwingOptionsConfig(initial_cash=50_000.0, risk_per_trade=0.02),
    )

    assert analysis["result"]["Score"] >= 80
    assert analysis["result"]["Setup"] == "ACTIONABLE"
    assert analysis["result"]["Signal"] == "BUY"


def test_watchlist_classification_on_premium_risk_cap(monkeypatch):
    monkeypatch.setattr(swing_options_module, "evaluate_source_signals", lambda **kwargs: _watchlist_evaluation())
    analysis = swing_options_module.analyze_ticker(
        ticker="NVDA",
        config=swing_options_module.SwingOptionsConfig(initial_cash=10_000.0, risk_per_trade=0.015),
    )

    assert analysis["result"]["Score"] >= 80
    assert analysis["result"]["Setup"] == "WATCHLIST"
    assert analysis["result"]["Signal"] == "HOLD"


def test_actionable_conversion_audit_shows_premium_budget_block():
    evaluation = _watchlist_evaluation()
    conversion = _finalize_signal_conversion(
        ticker="NVDA",
        signal_date="2026-05-01",
        score=82.0,
        raw_setup="ACTIONABLE",
        sources=evaluation["sources"],
        selected_source_strategy="four-hour-trend",
        latest_close=200.0,
        latest_atr=6.0,
        supporting_signals=2,
        config=swing_options_module.SwingOptionsConfig(initial_cash=10_000.0, risk_per_trade=0.015),
    )

    assert conversion["audit"]["PlanCreated"] is True
    assert conversion["audit"]["PremiumOverBudget"] is True
    assert conversion["audit"]["FinalSetup"] == "WATCHLIST"
    assert conversion["audit"]["FinalSignal"] == "HOLD"
    assert conversion["audit"]["BlockReason"] == "PREMIUM_OVER_BUDGET"


def test_high_quality_watchlist_can_convert_to_buy():
    evaluation = _actionable_evaluation()
    conversion = _finalize_signal_conversion(
        ticker="NVDA",
        signal_date="2026-05-01",
        score=75.0,
        raw_setup="WATCHLIST",
        sources=evaluation["sources"],
        selected_source_strategy="ema-rsi",
        latest_close=45.0,
        latest_atr=0.9,
        supporting_signals=2,
        config=swing_options_module.SwingOptionsConfig(),
    )

    assert conversion["audit"]["PreFinalSetup"] == "WATCHLIST"
    assert conversion["audit"]["PlanCreated"] is True
    assert conversion["audit"]["PremiumOverBudget"] is False
    assert conversion["audit"]["FinalSetup"] == "WATCHLIST"
    assert conversion["audit"]["FinalSignal"] == "BUY"
    assert conversion["audit"]["BlockReason"] == "CONVERTED_TO_BUY"


def test_wait_and_avoid_classifications(monkeypatch):
    monkeypatch.setattr(swing_options_module, "evaluate_source_signals", lambda **kwargs: _wait_evaluation())
    wait_analysis = swing_options_module.analyze_ticker(ticker="NVDA")
    monkeypatch.setattr(swing_options_module, "evaluate_source_signals", lambda **kwargs: _avoid_evaluation())
    avoid_analysis = swing_options_module.analyze_ticker(ticker="NVDA")

    assert wait_analysis["result"]["Setup"] == "WAIT"
    assert avoid_analysis["result"]["Setup"] in {"AVOID", "WEAK_TREND"}


def test_long_call_plan_generation():
    plan = build_long_call_plan(ticker="NVDA", price=120.0, atr=3.0, signal_date="2026-05-01", score=88.0)

    assert plan.option_type == "CALL"
    assert 30 <= plan.dte <= 60
    assert 0.55 <= plan.delta_target <= 0.70
    assert plan.estimated_premium > 0
    assert plan.max_loss == round(plan.estimated_premium * 100, 2)


def test_affordability_fallback_selects_lower_cost_contract():
    plan = build_long_call_plan(
        ticker="NVDA",
        price=140.0,
        atr=3.5,
        signal_date="2026-05-01",
        score=88.0,
        premium_budget=225.0,
        preferred_target_dte=45,
    )

    assert plan.max_loss <= 225.0
    assert "Affordable fallback" in plan.notes


def test_no_trade_when_extended(monkeypatch):
    monkeypatch.setattr(swing_options_module, "evaluate_source_signals", lambda **kwargs: _extended_evaluation())
    analysis = swing_options_module.analyze_ticker(
        ticker="NVDA",
        config=swing_options_module.SwingOptionsConfig(initial_cash=50_000.0, risk_per_trade=0.02),
    )

    assert analysis["result"]["Setup"] == "EXTENDED"
    assert analysis["result"]["Signal"] == "HOLD"


def test_no_trade_when_weak_trend(monkeypatch):
    monkeypatch.setattr(swing_options_module, "evaluate_source_signals", lambda **kwargs: _weak_trend_evaluation())
    analysis = swing_options_module.analyze_ticker(
        ticker="NVDA",
        config=swing_options_module.SwingOptionsConfig(initial_cash=50_000.0, risk_per_trade=0.02),
    )

    assert analysis["result"]["Setup"] == "WEAK_TREND"
    assert analysis["result"]["Signal"] == "HOLD"


def test_small_account_execution_labels():
    labeled = apply_execution_profile_labels(
        result={
            "Ticker": "AAPL",
            "Signal": "BUY",
            "MaxLoss": 120.0,
            "Reason": "score=83.00; setup=WATCHLIST; source=ema-rsi; estimated premium is approximate.",
        },
        account_profile="small_account_options",
    )

    assert labeled["AccountProfile"] == "small_account_options"
    assert labeled["SmallAccountEligible"] == "YES"
    assert labeled["PremiumStatus"] == "OK"
    assert "preferred $75-$125 premium range" in labeled["Reason"]


def test_small_account_execution_labels_flag_expensive_trade():
    labeled = apply_execution_profile_labels(
        result={
            "Ticker": "QQQ",
            "Signal": "BUY",
            "MaxLoss": 175.0,
            "Reason": "score=86.00; setup=ACTIONABLE; source=four-hour-trend; estimated premium is approximate.",
        },
        account_profile="small_account_options",
    )

    assert labeled["SmallAccountEligible"] == "NO"
    assert labeled["PremiumStatus"] == "TOO_EXPENSIVE"
    assert "$150 small-account cap" in labeled["Reason"]


def test_small_account_execution_labels_hold_rows_show_na():
    labeled = apply_execution_profile_labels(
        result={
            "Ticker": "SPY",
            "Signal": "HOLD",
            "MaxLoss": 0.0,
            "Reason": "score=32.00; setup=EXTENDED; source=ema-rsi; estimated premium is approximate.",
        },
        account_profile="small_account_options",
    )

    assert labeled["SmallAccountEligible"] == "NO"
    assert labeled["PremiumStatus"] == "N/A"
    assert "No option plan generated because signal is HOLD." in labeled["Reason"]


def test_report_output_path_and_ticker_output(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(swing_options_module, "evaluate_source_signals", lambda **kwargs: _actionable_evaluation())
    analysis = swing_options_module.analyze_ticker(
        ticker="NVDA",
        config=swing_options_module.SwingOptionsConfig(initial_cash=50_000.0, risk_per_trade=0.02),
    )

    labeled_result = apply_execution_profile_labels(analysis["result"], account_profile="small_account_options")

    print_ticker_plan(analysis)
    print_scan_results([labeled_result])
    save_reports(analysis=analysis, output_dir=str(tmp_path))

    captured = capsys.readouterr()
    assert "Swing Options Score" in captured.out
    assert "small_account_options" in captured.out
    assert "Planner/scanner only" in captured.out
    assert (tmp_path / "swing_options" / "NVDA_options_plan.csv").exists()
    assert (tmp_path / "swing_options" / "NVDA_source_signals.csv").exists()
    assert (tmp_path / "swing_options" / "NVDA_latest_signal.csv").exists()


def test_journal_compatibility(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(swing_options_module, "evaluate_source_signals", lambda **kwargs: _actionable_evaluation())
    result = swing_options_module.analyze_ticker(
        ticker="NVDA",
        config=swing_options_module.SwingOptionsConfig(initial_cash=50_000.0, risk_per_trade=0.02),
    )["result"]

    workbook_path = update_paper_trading_journal(results=[result], output_dir=str(tmp_path))
    trade_df = pd.read_excel(workbook_path, sheet_name=TRADE_JOURNAL_SHEET)

    assert len(trade_df) == 1
    assert trade_df.iloc[0]["Options Structure"] == f"Long Call {result['Strike']:.2f}C"
