from pathlib import Path

import json

from algo_backtester.reports.daily_summary_report import (
    build_daily_summary,
    render_daily_summary_markdown,
    save_daily_summary,
    select_top_setup,
)


def _sample_scan_payload() -> dict[str, dict]:
    return {
        "ema-rsi": {
            "strategy": "ema-rsi",
            "profile": "broad_market",
            "tickers": ["AAPL", "SPY"],
            "results": [
                {
                    "Ticker": "AAPL",
                    "Strategy": "ema-rsi",
                    "Signal": "HOLD",
                    "Setup": "WATCHLIST",
                    "ActionState": "WATCHLIST",
                    "Price": 210.0,
                    "RSI": 57.0,
                    "ATR": 3.2,
                    "Reason": "Near setup on daily trend pullback.",
                },
                {
                    "Ticker": "SPY",
                    "Strategy": "ema-rsi",
                    "Signal": "HOLD",
                    "Setup": "EXTENDED",
                    "ActionState": "IGNORE",
                    "Price": 600.0,
                    "RSI": 71.0,
                    "ATR": 5.0,
                    "Reason": "Extended above the pullback zone.",
                },
            ],
        },
        "swing-options-debit-spread:small_account_debit_spreads": {
            "strategy": "swing-options-debit-spread",
            "profile": "small_account_debit_spreads",
            "tickers": ["AAPL", "AMD"],
            "results": [
                {
                    "Ticker": "QQQ",
                    "Strategy": "swing-options-debit-spread",
                    "Signal": "BUY",
                    "Setup": "WATCHLIST",
                    "Price": 500.0,
                    "RSI": 55.0,
                    "ATR": 4.0,
                    "OptionStructure": "Bull Call Debit Spread 500/515",
                    "EstDebit": 1.1,
                    "MaxLoss": 110.0,
                    "RewardRisk": 2.1,
                    "SmallAccountEligible": "YES",
                    "PremiumStatus": "OK",
                    "ActionState": "ACTIONABLE",
                    "MarketRegime": "BULLISH",
                    "RegimeReason": "SPY and QQQ above trend while VIX is flat to lower.",
                    "FinalScore": 91.0,
                    "SetupScore": 91.0,
                    "SetupRating": "A_SETUP",
                    "FinalDecision": "Review before market open.",
                    "Reason": "Large-cap benchmark candidate.",
                },
                {
                    "Ticker": "AMD",
                    "Strategy": "swing-options-debit-spread",
                    "Signal": "HOLD",
                    "Setup": "EXTENDED",
                    "Price": 145.0,
                    "RSI": 73.0,
                    "ATR": 4.1,
                    "PremiumStatus": "TOO_EXPENSIVE",
                    "ActionState": "IGNORE",
                    "Reason": "Blocked because the setup is EXTENDED.",
                },
            ],
        },
        "swing-options-debit-spread:small_account_growth": {
            "strategy": "swing-options-debit-spread",
            "profile": "small_account_growth",
            "tickers": ["AAPL", "PLTR"],
            "results": [
                {
                    "Ticker": "AAPL",
                    "Strategy": "swing-options-debit-spread",
                    "Signal": "BUY",
                    "Setup": "WATCHLIST",
                    "Price": 210.0,
                    "RSI": 57.0,
                    "ATR": 3.2,
                    "OptionStructure": "Bull Call Debit Spread 210/220",
                    "EstDebit": 1.0,
                    "MaxLoss": 100.0,
                    "RewardRisk": 2.4,
                    "SmallAccountEligible": "YES",
                    "PremiumStatus": "OK",
                    "ActionState": "ACTIONABLE",
                    "MarketRegime": "BULLISH",
                    "RegimeReason": "SPY and QQQ above trend while VIX is flat to lower.",
                    "FinalScore": 88.0,
                    "SetupScore": 88.0,
                    "SetupRating": "A_SETUP",
                    "FinalDecision": "Review before market open.",
                    "Reason": "Base swing-options signal was HOLD. Tuned conversion upgraded the setup.",
                },
            ],
        },
    }


def test_build_daily_summary_identifies_debit_spread_setup():
    summary = build_daily_summary(scan_payload=_sample_scan_payload(), report_date="2026-05-05")

    assert summary["executive_decision"] == "One valid setup: AAPL debit spread"
    assert summary["market_state"] == "actionable"
    assert summary["actionable_count"] == 2
    assert summary["top_setup"]["ticker"] == "AAPL"
    assert summary["top_setup"]["conviction"] == "Medium"
    assert summary["small_account_options"][0]["ticker"] == "AAPL"
    assert summary["large_cap_debit_context"][0]["ticker"] == "QQQ"
    assert summary["watchlist_names"][0]["ticker"] == "AAPL"
    assert summary["breadth_snapshot"] == {
        "actionable": 2,
        "watchlist": 1,
        "extended": 2,
        "weak": 0,
        "avoid": 0,
    }
    assert summary["market_regime"]["regime"] == "BULLISH"
    assert any(row["ticker"] == "SPY" for row in summary["ignore_list"])
    assert summary["key_no_trade_reasons"][0]["reason"] == "EXTENDED"
    assert summary["debit_spread_context"] is not None
    assert summary["no_trade_reason"] is None
    assert summary["paper_execution_checklist"]
    assert "Planner output is not an executable order. It is only a candidate generator." in summary["manual_chain_confirmation_rules"]


def test_no_trade_reason_only_when_no_actionable():
    payload = _sample_scan_payload()
    payload["swing-options-debit-spread:small_account_growth"]["results"][0]["Signal"] = "HOLD"
    payload["swing-options-debit-spread:small_account_growth"]["results"][0]["ActionState"] = "WATCHLIST"
    payload["swing-options-debit-spread:small_account_debit_spreads"]["results"][0]["Signal"] = "HOLD"
    payload["swing-options-debit-spread:small_account_debit_spreads"]["results"][0]["ActionState"] = "WATCHLIST"
    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")

    assert summary["actionable_count"] == 0
    assert summary["no_trade_reason"] is not None


def test_select_top_setup_prefers_debit_spread():
    top_setup = select_top_setup(_sample_scan_payload())
    assert top_setup is not None
    assert top_setup["display"] == "AAPL | swing-options-debit-spread | BUY | 210/220 debit spread | Max Risk $100"
    assert top_setup["structure"] == "210/220 Bull Call Debit Spread"


def test_render_and_save_daily_summary(tmp_path: Path):
    summary = build_daily_summary(scan_payload=_sample_scan_payload(), report_date="2026-05-05")
    markdown = render_daily_summary_markdown(summary)

    assert "# Daily Trading Summary - 2026-05-05" in markdown
    assert "## Executive Decision" in markdown
    assert "## Top Setup" in markdown
    assert "Ticker: AAPL" in markdown
    assert "Market Regime: BULLISH" in markdown
    assert "Final Score: 88.00" in markdown
    assert "Setup Score: 88.00" in markdown
    assert "Setup Rating: A_SETUP" in markdown
    assert "Conviction: Medium" in markdown
    assert "## Market Regime" in markdown
    assert "Regime: BULLISH" in markdown
    assert "## Breadth Snapshot" in markdown
    assert "## Large-Cap Debit Spread Context" in markdown
    assert "QQQ | Bull Call Debit Spread 500/515" in markdown
    assert "## Small-Account Growth Debit Spread Candidates" in markdown
    assert "AAPL | Bull Call Debit Spread 210/220" in markdown
    assert "## Debit Spread Historical Context" in markdown
    assert "## Manual Live Chain Confirmation Required" in markdown
    assert "## Key No-Trade Reasons" in markdown
    assert "- EXTENDED: 2" in markdown
    assert "Planner output is not an executable order. It is only a candidate generator." in markdown
    assert "## Paper Execution Checklist" in markdown
    assert "PROXY VALIDATION ONLY" in markdown

    paths = save_daily_summary(summary=summary, output_dir=str(tmp_path))
    assert paths["markdown"].exists()
    assert paths["json"].exists()

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["executive_decision"] == "One valid setup: AAPL debit spread"


def test_non_options_top_setup_uses_directional_wording():
    payload = _sample_scan_payload()
    payload["swing-options-debit-spread:small_account_growth"]["results"][0]["Signal"] = "HOLD"
    payload["swing-options-debit-spread:small_account_growth"]["results"][0]["ActionState"] = "WATCHLIST"
    payload["swing-options-debit-spread:small_account_debit_spreads"]["results"][0]["Signal"] = "HOLD"
    payload["swing-options-debit-spread:small_account_debit_spreads"]["results"][0]["ActionState"] = "WATCHLIST"
    payload["four-hour-trend"] = {
        "strategy": "four-hour-trend",
        "profile": "broad_market",
        "tickers": ["CRM"],
        "results": [
            {
                "Ticker": "CRM",
                "Strategy": "four-hour-trend",
                "Signal": "SHORT_SETUP",
                "Setup": "ACTIONABLE",
                "ActionState": "ACTIONABLE",
                "Price": 181.51,
                "MarketRegime": "BULLISH",
                "FinalScore": 75.0,
                "SetupScore": 75.0,
                "SetupRating": "B_SETUP",
                "FinalDecision": "Review before market open. Only enter if setup remains intact.",
                "Reason": "4H bearish continuation setup confirmed.",
            }
        ],
    }

    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")
    markdown = render_daily_summary_markdown(summary)

    assert "Max Risk: Not calculated - directional setup only" in markdown
    assert "Reward/Risk: Not calculated - no options structure generated" in markdown


def test_blocked_wait_setup_moves_to_ignore_only():
    payload = _sample_scan_payload()
    payload["swing-options-debit-spread:small_account_growth"]["results"].append(
        {
            "Ticker": "AAPL",
            "Strategy": "swing-options-debit-spread",
            "Signal": "HOLD",
            "Setup": "WAIT",
            "Price": 210.0,
            "RSI": 57.0,
            "ATR": 3.2,
            "PremiumStatus": "TOO_EXPENSIVE",
            "Reason": "Blocked wait setup.",
        }
    )

    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")
    watchlist_pairs = {(row["ticker"], row["strategy"]) for row in summary["watchlist_names"]}
    ignore_pairs = {(row["ticker"], row["strategy"]) for row in summary["ignore_list"]}

    assert ("AAPL", "swing-options-debit-spread") in ignore_pairs
    assert ("AAPL", "swing-options-debit-spread") not in watchlist_pairs


def test_near_setup_without_blockers_remains_watchlist():
    payload = _sample_scan_payload()
    payload["four-hour-trend"] = {
        "strategy": "four-hour-trend",
        "profile": "broad_market",
        "tickers": ["UBER"],
        "results": [
            {
                "Ticker": "UBER",
                "Strategy": "four-hour-trend",
                "Signal": "HOLD",
                "Setup": "NEAR_SETUP",
                "Price": 73.6,
                "Reason": "Near short setup.",
            }
        ],
    }

    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")
    watchlist_pairs = {(row["ticker"], row["strategy"]) for row in summary["watchlist_names"]}
    ignore_pairs = {(row["ticker"], row["strategy"]) for row in summary["ignore_list"]}

    assert ("UBER", "four-hour-trend") in watchlist_pairs
    assert ("UBER", "four-hour-trend") not in ignore_pairs


def test_no_duplicate_ticker_strategy_pair_across_watchlist_and_ignore():
    payload = _sample_scan_payload()
    payload["swing-options-debit-spread:small_account_growth"]["results"].append(
        {
            "Ticker": "AAPL",
            "Strategy": "swing-options-debit-spread",
            "Signal": "HOLD",
            "Setup": "WAIT",
            "Price": 210.0,
            "PremiumStatus": "BAD_REWARD_RISK",
            "Reason": "Blocked candidate.",
        }
    )

    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")
    watchlist_pairs = {(row["ticker"], row["strategy"]) for row in summary["watchlist_names"]}
    ignore_pairs = {(row["ticker"], row["strategy"]) for row in summary["ignore_list"]}

    assert watchlist_pairs.isdisjoint(ignore_pairs)


def test_watchlist_and_ignore_follow_action_state():
    payload = _sample_scan_payload()
    payload["four-hour-trend"] = {
        "strategy": "four-hour-trend",
        "profile": "broad_market",
        "tickers": ["PLTR", "SHOP"],
        "results": [
            {
                "Ticker": "PLTR",
                "Strategy": "four-hour-trend",
                "Signal": "HOLD",
                "Setup": "WAIT",
                "ActionState": "WATCHLIST",
                "Reason": "Needs more confirmation.",
            },
            {
                "Ticker": "SHOP",
                "Strategy": "four-hour-trend",
                "Signal": "HOLD",
                "Setup": "NO_TRADE",
                "ActionState": "NO_TRADE",
                "NoTradeReasons": ["ATR is too low."],
                "Reason": "No trade.",
            },
        ],
    }

    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")
    watchlist_pairs = {(row["ticker"], row["strategy"]) for row in summary["watchlist_names"]}
    ignore_pairs = {(row["ticker"], row["strategy"]) for row in summary["ignore_list"]}

    assert ("PLTR", "four-hour-trend") in watchlist_pairs
    assert ("SHOP", "four-hour-trend") in ignore_pairs


def test_key_no_trade_reasons_aggregate_full_labels_without_character_truncation():
    payload = {
        "swing-options-debit-spread:small_account_growth": {
            "strategy": "swing-options-debit-spread",
            "profile": "small_account_growth",
            "tickers": ["AAPL", "AMD", "PLTR"],
            "results": [
                {
                    "Ticker": "AAPL",
                    "Strategy": "swing-options-debit-spread",
                    "Signal": "HOLD",
                    "Setup": "NO_TRADE",
                    "ActionState": "NO_TRADE",
                    "NoTradeReasons": "['Price is too extended from EMA20.', 'Expected move exhaustion is already present.']",
                    "Reason": "No trade.",
                },
                {
                    "Ticker": "AMD",
                    "Strategy": "swing-options-debit-spread",
                    "Signal": "HOLD",
                    "Setup": "NO_TRADE",
                    "ActionState": "NO_TRADE",
                    "NoTradeReasons": "Price is too extended from EMA20. | ATR is too low.",
                    "Reason": "No trade.",
                },
                {
                    "Ticker": "PLTR",
                    "Strategy": "swing-options-debit-spread",
                    "Signal": "HOLD",
                    "Setup": "EXTENDED",
                    "ActionState": "IGNORE",
                    "NoTradeReasons": ["Price is too extended from EMA20.", "Price is too extended from EMA20."],
                    "Reason": "No trade.",
                },
            ],
        }
    }

    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")

    assert summary["key_no_trade_reasons"][0] == {"reason": "EXTENDED", "count": 3}
    labels = {row["reason"] for row in summary["key_no_trade_reasons"]}
    assert "EXPECTED_MOVE_EXHAUSTION" in labels
    assert "LOW_ATR" in labels
    assert "E" not in labels
    assert "O" not in labels
    assert "T" not in labels


def test_key_no_trade_reasons_deduplicate_per_ticker():
    payload = {
        "swing-options-debit-spread:small_account_growth": {
            "strategy": "swing-options-debit-spread",
            "profile": "small_account_growth",
            "tickers": ["AAPL"],
            "results": [
                {
                    "Ticker": "AAPL",
                    "Strategy": "swing-options-debit-spread",
                    "Signal": "HOLD",
                    "Setup": "EXTENDED",
                    "ActionState": "IGNORE",
                    "NoTradeReasons": [
                        "Price is too extended from EMA20.",
                        "Price is too extended from EMA20.",
                    ],
                    "Reason": "No trade.",
                }
            ],
        }
    }

    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")
    assert summary["key_no_trade_reasons"][0] == {"reason": "EXTENDED", "count": 1}


def test_key_no_trade_reasons_render_stably_in_markdown():
    payload = _sample_scan_payload()
    payload["swing-options-debit-spread:small_account_growth"]["results"].append(
        {
            "Ticker": "PLTR",
            "Strategy": "swing-options-debit-spread",
            "Signal": "HOLD",
            "Setup": "NO_TRADE",
            "ActionState": "NO_TRADE",
            "NoTradeReasons": "['Expected move exhaustion is already present.']",
            "Reason": "No trade.",
        }
    )

    summary = build_daily_summary(scan_payload=payload, report_date="2026-05-05")
    markdown = render_daily_summary_markdown(summary)

    assert "## Key No-Trade Reasons" in markdown
    assert "- EXPECTED_MOVE_EXHAUSTION:" in markdown
    assert "- E:" not in markdown
