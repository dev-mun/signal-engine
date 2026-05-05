from pathlib import Path

import pandas as pd

import algo_backtester.backtests.swing_options_debit_spread_backtester as spread_module
from algo_backtester.reports.swing_options_debit_spread_report import (
    print_scan_results,
    print_ticker_plan,
    save_reports,
)
from algo_backtester.strategies.swing_options_debit_spread import build_bull_call_debit_spread
from algo_backtester.watchlists import get_default_watchlist_for_strategy, get_watchlist


def _underlying_buy_result() -> dict:
    return {
        "result": {
            "Ticker": "AMD",
            "Strategy": "swing-options",
            "Signal": "BUY",
            "Setup": "WATCHLIST",
            "Score": 82.0,
            "SourceSummary": "ema-rsi:BUY/ACTIONABLE; four-hour-trend:BUY/ACTIONABLE",
            "Price": 110.0,
            "RSI": 54.0,
            "ATR": 3.0,
            "SignalDate": "2026-05-01",
            "PlannedExecutionDate": "2026-05-04",
            "Equity": 25_000.0,
        },
        "sources": [
            {
                "strategy": "ema-rsi",
                "signal": "BUY",
                "setup": "ACTIONABLE",
                "price": 110.0,
                "rsi": 54.0,
                "atr": 3.0,
                "trend_quality": True,
                "volume_confirmed": True,
                "recent_momentum": True,
                "bullish_support": True,
                "notes": "Qualified",
            }
        ],
    }


def _underlying_hold_result() -> dict:
    payload = _underlying_buy_result()
    payload["result"] = dict(payload["result"])
    payload["result"]["Signal"] = "HOLD"
    payload["result"]["Setup"] = "WAIT"
    return payload


def test_debit_spread_watchlist_profile_loads():
    assert get_watchlist("small_account_debit_spreads") == ["SPY", "QQQ", "AAPL", "AMD"]
    assert get_default_watchlist_for_strategy("swing-options-debit-spread") == get_watchlist("small_account_debit_spreads")


def test_build_debit_spread_plan():
    plan = build_bull_call_debit_spread(
        ticker="AMD",
        price=110.0,
        atr=3.0,
        signal_date="2026-05-01",
        score=82.0,
    )

    assert plan.option_structure.startswith("Bull Call Debit Spread")
    assert 30 <= plan.dte <= 60
    assert plan.est_debit > 0
    assert plan.max_loss == round(plan.est_debit * 100, 2)


def test_scan_runs_for_buy_candidate(monkeypatch):
    monkeypatch.setattr(spread_module, "analyze_swing_options_ticker", lambda **kwargs: _underlying_buy_result())
    result = spread_module.scan_ticker("AMD")

    assert result["Strategy"] == "swing-options-debit-spread"
    assert result["PremiumStatus"] in {"OK", "ACCEPTABLE", "TOO_EXPENSIVE", "BAD_REWARD_RISK"}
    assert result["OptionStructure"].startswith("Bull Call Debit Spread")
    assert result["Reason"] == "Debit spread plan generated from confirmed swing-options BUY signal."


def test_hold_candidate_has_no_option_plan(monkeypatch):
    monkeypatch.setattr(spread_module, "analyze_swing_options_ticker", lambda **kwargs: _underlying_hold_result())
    monkeypatch.setattr(
        spread_module,
        "evaluate_source_signals",
        lambda **kwargs: {
            "contexts": {
                "ema-rsi": {
                    "latest": pd.Series({"Close": 110.0, "EMA50": 100.0, "RSI": 54.0}),
                    "prev": pd.Series({"Close": 109.0}),
                }
            }
        },
    )
    result = spread_module.scan_ticker("AMD")

    assert result["Signal"] == "HOLD"
    assert result["PremiumStatus"] in {"N/A", "OK", "ACCEPTABLE"}
    assert result["SmallAccountEligible"] == "NO"
    assert "remained non-actionable" in result["Reason"]


def test_debit_spread_report_output(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(spread_module, "analyze_swing_options_ticker", lambda **kwargs: _underlying_buy_result())
    analysis = spread_module.analyze_ticker("AMD")

    print_ticker_plan(analysis)
    print_scan_results([analysis["result"]])
    save_reports(analysis=analysis, output_dir=str(tmp_path))

    captured = capsys.readouterr()
    assert "PROXY DEBIT SPREAD VALIDATION ONLY" in captured.out
    assert (tmp_path / "swing_options_debit_spread" / "AMD_options_plan.csv").exists()
    assert (tmp_path / "swing_options_debit_spread" / "AMD_source_signals.csv").exists()


def test_proxy_backtest_summary():
    signal_df = pd.DataFrame(
        [
            {"SignalDate": "2026-01-05", "Signal": "BUY"},
            {"SignalDate": "2026-01-12", "Signal": "BUY"},
        ]
    )
    affordable_df = pd.DataFrame(
        [
            {"SignalDate": "2026-01-05"},
        ]
    )
    trades_df = pd.DataFrame(
        [
            {"Ticker": "AMD", "SignalDate": "2026-01-05", "MoveQuality": "SUITABLE", "HoldDays": 3, "ProxyPnLPct": 50.0, "ProxyPnLDollars": 60.0, "ProxyWinner": "YES"},
            {"Ticker": "AAPL", "SignalDate": "2026-01-20", "MoveQuality": "FAILED", "HoldDays": 2, "ProxyPnLPct": -50.0, "ProxyPnLDollars": -55.0, "ProxyWinner": "NO"},
        ]
    )

    summary_df = spread_module.build_debit_spread_summary(signal_df, affordable_df, trades_df)
    monthly_df = spread_module.build_debit_spread_monthly_report(signal_df, affordable_df, trades_df)

    assert int(summary_df.iloc[0]["TotalSignals"]) == 2
    assert int(summary_df.iloc[0]["AffordableTrades"]) == 1
    assert float(summary_df.iloc[0]["WinRateProxy"]) == 50.0
    assert list(monthly_df["Month"]) == ["2026-01"]


def test_tuned_watchlist_candidate_converts_to_buy():
    sources = [
        spread_module.SwingSourceSignal("ema-rsi", "HOLD", "NEAR_SETUP", 110.0, 54.0, 3.0, True, True, True, True, "x"),
        spread_module.SwingSourceSignal("four-hour-trend", "HOLD", "NEEDS_PULLBACK", 110.0, 55.0, 3.0, True, True, True, True, "x"),
        spread_module.SwingSourceSignal("rsi-bollinger-v2", "HOLD", "WAIT", 110.0, 52.0, 3.0, True, True, True, False, "x"),
    ]
    plan = build_bull_call_debit_spread(ticker="AMD", price=110.0, atr=3.0, signal_date="2026-05-01", score=72.0)

    signal, setup, reason = spread_module._resolve_debit_spread_signal(
        raw_setup="WATCHLIST",
        strict_signal="HOLD",
        score=72.0,
        plan=plan,
        sources=sources,
        close_price=110.0,
        ema50=100.0,
        rsi=54.0,
        prev_close=109.0,
        mode="tuned",
    )

    assert signal == "BUY"
    assert setup == "WATCHLIST"
    assert reason == "TUNED_WATCHLIST_BUY"


def test_tuned_conversion_reason_text():
    plan = build_bull_call_debit_spread(ticker="AMD", price=110.0, atr=3.0, signal_date="2026-05-01", score=72.0)

    reason = spread_module._reason_text(
        signal="BUY",
        plan=plan,
        conversion_reason="TUNED_NEAR_ACTIONABLE_2_SOURCES",
        raw_setup="WATCHLIST",
    )

    assert reason == (
        "Base swing-options signal was HOLD. Tuned debit-spread conversion upgraded this setup "
        "to BUY due to near-actionable bullish source alignment."
    )


def test_blocked_reason_text_appends_blocker():
    plan = build_bull_call_debit_spread(ticker="AMD", price=110.0, atr=3.0, signal_date="2026-05-01", score=72.0)

    reason = spread_module._reason_text(
        signal="HOLD",
        plan=plan,
        conversion_reason="HARD_BLOCKER",
        raw_setup="EXTENDED",
    )

    assert reason == (
        "No debit spread plan generated because the underlying swing-options signal remained non-actionable. "
        "Blocked because the setup is EXTENDED."
    )
