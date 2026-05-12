from pathlib import Path
from dataclasses import replace

import pandas as pd

import algo_backtester.backtests.swing_options_debit_spread_backtester as spread_module
from algo_backtester.market_regime import MarketRegimeSnapshot
from algo_backtester.options_signal_quality import OptionsSignalAssessment
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
            },
            {
                "strategy": "four-hour-trend",
                "signal": "BUY",
                "setup": "ACTIONABLE",
                "price": 110.0,
                "rsi": 55.0,
                "atr": 3.0,
                "trend_quality": True,
                "volume_confirmed": True,
                "recent_momentum": True,
                "bullish_support": True,
                "notes": "Qualified",
            },
            {
                "strategy": "rsi-bollinger-v2",
                "signal": "HOLD",
                "setup": "WAIT",
                "price": 110.0,
                "rsi": 52.0,
                "atr": 3.0,
                "trend_quality": True,
                "volume_confirmed": True,
                "recent_momentum": True,
                "bullish_support": False,
                "notes": "Qualified",
            },
        ],
    }


def _underlying_hold_result() -> dict:
    payload = _underlying_buy_result()
    payload["result"] = dict(payload["result"])
    payload["result"]["Signal"] = "HOLD"
    payload["result"]["Setup"] = "WAIT"
    return payload


def _affordable_plan(ticker: str = "AMD"):
    base_plan = build_bull_call_debit_spread(
        ticker=ticker,
        price=110.0 if ticker == "AMD" else 284.18,
        atr=3.0 if ticker == "AMD" else 6.5,
        signal_date="2026-05-01",
        score=82.0,
    )
    return replace(
        base_plan,
        est_debit=1.1,
        max_loss=110.0,
        max_profit=240.0,
        reward_risk=2.18,
        premium_status="OK",
        small_account_eligible=True,
        notes=f"{base_plan.notes} Test affordable override.",
    )


def _source_contexts(close_price: float = 110.0) -> dict:
    return {
        "contexts": {
            "ema-rsi": {
                "latest": pd.Series(
                    {
                        "Close": close_price,
                        "EMA20": close_price - 1.0,
                        "EMA50": close_price - 5.0,
                        "EMA200": close_price - 10.0,
                        "RSI": 54.0,
                        "ATR": 3.0,
                        "Volume": 2_000_000.0,
                        "AverageVolume20": 1_500_000.0,
                    }
                ),
                "prev": pd.Series({"Close": close_price - 1.0}),
            }
        }
    }


def _market_regime_snapshot(regime: str = "BULLISH") -> MarketRegimeSnapshot:
    return MarketRegimeSnapshot(
        market_regime=regime,
        regime_reason=f"{regime} regime test snapshot.",
        spy_close=500.0,
        spy_sma20=495.0,
        spy_sma50=490.0,
        qqq_close=400.0,
        qqq_sma20=395.0,
        qqq_sma50=390.0,
        vix_close=15.0,
        vix_trend="FLAT_OR_DECLINING" if regime == "BULLISH" else "RISING",
    )


def _assessment(**overrides) -> OptionsSignalAssessment:
    payload = {
        "market_regime": "BULLISH",
        "regime_reason": "Bullish regime test snapshot.",
        "timeframe_confirmation": {"aligned": True, "four_hour_only": False, "reason": "Aligned"},
        "daily_trend": "BULLISH",
        "four_hour_trend": "BULLISH",
        "setup_score": 88.0,
        "setup_rating": "A_SETUP",
        "evaluated_setup": "ACTIONABLE",
        "no_trade_reasons": [],
        "warnings": [],
        "final_decision": "Review before market open. Only enter if continuation structure remains intact and liquidity is acceptable.",
    }
    payload.update(overrides)
    return OptionsSignalAssessment(**payload)


def test_debit_spread_watchlist_profile_loads():
    assert get_watchlist("small_account_debit_spreads") == ["SPY", "QQQ", "AAPL", "AMD", "SLV"]
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
    assert plan.est_long_call_ask > 0
    assert plan.est_short_call_bid > 0
    assert plan.est_debit > 0
    assert plan.max_loss == round(plan.est_debit * 100, 2)
    assert plan.approximation_confidence in {"LOW", "MEDIUM", "HIGH"}


def test_pricing_approximation_is_directionally_realistic_for_aapl():
    plan = build_bull_call_debit_spread(
        ticker="AAPL",
        price=284.18,
        atr=6.5,
        signal_date="2026-05-05",
        score=82.0,
    )

    assert plan.long_strike == 285.0
    assert plan.short_strike in {295.0, 300.0}
    assert plan.est_long_call_ask >= 7.0
    assert plan.est_short_call_bid >= 1.75
    assert 4.5 <= plan.est_debit <= 6.25
    assert plan.max_loss >= 450.0


def test_scan_runs_for_buy_candidate(monkeypatch):
    monkeypatch.setattr(spread_module, "analyze_swing_options_ticker", lambda **kwargs: _underlying_buy_result())
    monkeypatch.setattr(spread_module, "build_bull_call_debit_spread", lambda **kwargs: _affordable_plan("AMD"))
    monkeypatch.setattr(spread_module, "analyze_market_regime", lambda **kwargs: _market_regime_snapshot("BULLISH"))
    monkeypatch.setattr(spread_module, "evaluate_source_signals", lambda **kwargs: _source_contexts())
    result = spread_module.scan_ticker("AMD")

    assert result["Strategy"] == "swing-options-debit-spread"
    assert result["PremiumStatus"] in {"OK", "ACCEPTABLE", "TOO_EXPENSIVE", "BAD_REWARD_RISK"}
    assert result["OptionStructure"].startswith("Bull Call Debit Spread")
    assert result["ApproximationConfidence"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["MarketRegime"] == "BULLISH"
    assert result["SetupRating"] in {"A_SETUP", "B_SETUP", "WATCHLIST", "NO_TRADE"}
    assert result["FinalScore"] == result["SetupScore"]
    assert result["ActionState"] == "ACTIONABLE"
    assert result["Reason"] == "Debit spread plan generated from confirmed swing-options BUY signal."


def test_hold_candidate_has_no_option_plan(monkeypatch):
    monkeypatch.setattr(spread_module, "analyze_swing_options_ticker", lambda **kwargs: _underlying_hold_result())
    monkeypatch.setattr(spread_module, "analyze_market_regime", lambda **kwargs: _market_regime_snapshot("BULLISH"))
    monkeypatch.setattr(spread_module, "evaluate_source_signals", lambda **kwargs: _source_contexts())
    result = spread_module.scan_ticker("AMD")

    assert result["Signal"] == "HOLD"
    assert result["PremiumStatus"] in {"N/A", "OK", "ACCEPTABLE", "TOO_EXPENSIVE"}
    assert result["SmallAccountEligible"] == "NO"
    assert result["FinalScore"] == result["SetupScore"]
    assert result["ActionState"] in {"WATCHLIST", "NO_TRADE", "IGNORE"}
    assert "remained non-actionable" in result["Reason"]


def test_debit_spread_report_output(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(spread_module, "analyze_swing_options_ticker", lambda **kwargs: _underlying_buy_result())
    monkeypatch.setattr(spread_module, "build_bull_call_debit_spread", lambda **kwargs: _affordable_plan("AMD"))
    monkeypatch.setattr(spread_module, "analyze_market_regime", lambda **kwargs: _market_regime_snapshot("BULLISH"))
    monkeypatch.setattr(spread_module, "evaluate_source_signals", lambda **kwargs: _source_contexts())
    analysis = spread_module.analyze_ticker("AMD")

    print_ticker_plan(analysis)
    print_scan_results([analysis["result"]])
    save_reports(analysis=analysis, output_dir=str(tmp_path))

    captured = capsys.readouterr()
    assert "PROXY DEBIT SPREAD VALIDATION ONLY" in captured.out
    assert "Approximation Confidence:" in captured.out
    assert "Final Score:" in captured.out
    assert "Action State:" in captured.out
    assert "Market Regime:" in captured.out
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
    plan = _affordable_plan("AMD")

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
        assessment=_assessment(),
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
        assessment=_assessment(warnings=[], evaluated_setup="WATCHLIST"),
    )

    assert reason == (
        "No debit spread plan generated because the underlying swing-options signal remained non-actionable. "
        "Blocked because the setup is EXTENDED."
    )


def test_no_trade_filter_for_bearish_market_regime():
    four_hour_source = spread_module.SwingSourceSignal(
        "four-hour-trend",
        "BUY",
        "ACTIONABLE",
        110.0,
        55.0,
        3.0,
        True,
        True,
        True,
        True,
        "x",
    )
    assessment = spread_module.evaluate_long_options_setup(
        signal_date="2026-05-01",
        market_regime_snapshot=_market_regime_snapshot("BEARISH"),
        daily_close=110.0,
        ema20=108.0,
        ema50=105.0,
        ema200=100.0,
        rsi=55.0,
        atr=3.0,
        avg_dollar_volume=50_000_000.0,
        volume_confirmed=True,
        recent_momentum=True,
        liquidity="A",
        earnings_date="PLACEHOLDER_OK",
        reward_risk=2.0,
        four_hour_source=four_hour_source,
    )

    assert assessment.evaluated_setup == "NO_TRADE"
    assert "Market regime strongly conflicts with a bullish debit spread." in assessment.no_trade_reasons
