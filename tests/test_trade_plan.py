from algo_backtester.options_engine import OptionsRecommendation
from algo_backtester.trade_plan import build_signal_interpretation, build_trade_plan


def make_no_trade_options_rec(signal: str) -> OptionsRecommendation:
    return OptionsRecommendation(
        stock_signal=signal,
        options_action="NO_OPTIONS_TRADE",
        structure="No trade",
        trade_quality="NO_TRADE",
        ticker="AAPL",
        expiration="N/A",
        dte=0,
        long_strike=0.0,
        short_strike=0.0,
        estimated_debit=0.0,
        max_loss=0.0,
        max_profit=0.0,
        breakeven=0.0,
        reward_risk=0.0,
        iv_rank_proxy=0.0,
        iv_status="N/A",
        max_risk_dollars=0.0,
        estimated_contracts=0,
        reason="No valid options trade.",
    )


def test_build_trade_plan_for_buy_returns_exact_levels():
    plan = build_trade_plan(
        signal="BUY",
        entry_price=271.06,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.08,
    )

    assert plan["HasActionableTrade"] is True
    assert plan["EstimatedEntryReference"] == 271.06
    assert plan["ActualEntry"] == "Next market open"
    assert plan["StopLoss"] == 249.38
    assert plan["TakeProfit"] == 325.27
    assert plan["InitialTrailingStop"] == 249.38
    assert plan["RiskPerShare"] == 21.68
    assert plan["RewardPerShare"] == 54.21
    assert plan["RewardRisk"] == 2.5


def test_build_signal_interpretation_for_buy_includes_planned_levels():
    latest_signal = {
        "Signal": "BUY",
        "Reason": "Trend pullback bullish entry",
        "Close": 271.06,
        "EntryPrice": 0.0,
        "InitialStopPrice": 0.0,
    }
    trade_plan = build_trade_plan(
        signal="BUY",
        entry_price=271.06,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.08,
    )
    options_rec = make_no_trade_options_rec("BUY")

    interpretation = build_signal_interpretation(latest_signal, trade_plan, options_rec)

    assert interpretation["Meaning"] == "A bullish pullback setup has been confirmed."
    assert "next session open" in interpretation["BacktesterSimulates"]
    assert "Estimated Buy Reference: $271.06" in interpretation["PlannedTradeLevels"]
    assert "Stop Loss: $249.38" in interpretation["PlannedTradeLevels"]
    assert "Reward/Risk: 2.50" in interpretation["RiskReward"]


def test_build_signal_interpretation_for_bearish_entry_is_truthful_about_backtest():
    latest_signal = {
        "Signal": "BEARISH_ENTRY",
        "Reason": "Trend pullback bearish entry",
        "Close": 180.0,
        "EntryPrice": 0.0,
        "InitialStopPrice": 0.0,
    }

    interpretation = build_signal_interpretation(
        latest_signal=latest_signal,
        trade_plan=build_trade_plan(
            signal="BEARISH_ENTRY",
            entry_price=180.0,
            stop_loss_pct=0.08,
            take_profit_pct=0.20,
            trailing_stop_pct=0.08,
        ),
        options_rec=make_no_trade_options_rec("BEARISH_ENTRY"),
    )

    assert "does not yet simulate short stock positions" in interpretation["BacktesterSimulates"]
    assert interpretation["HowOptionsFit"] == "This bearish thesis can be expressed with a put debit spread."
    assert "Estimated Short Reference: $180.00" in interpretation["PlannedTradeLevels"]
    assert "Stop Loss: $194.40" in interpretation["PlannedTradeLevels"]
    assert "Reward/Risk: 2.50" in interpretation["RiskReward"]


def test_build_trade_plan_for_bearish_entry_returns_exact_levels():
    plan = build_trade_plan(
        signal="BEARISH_ENTRY",
        entry_price=180.0,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.08,
    )

    assert plan["HasActionableTrade"] is True
    assert plan["EstimatedEntryReference"] == 180.0
    assert plan["ActualEntry"] == "Next market open"
    assert plan["StopLoss"] == 194.4
    assert plan["TakeProfit"] == 144.0
    assert plan["InitialTrailingStop"] == 194.4
    assert plan["RiskPerShare"] == 14.4
    assert plan["RewardPerShare"] == 36.0
    assert plan["RewardRisk"] == 2.5
