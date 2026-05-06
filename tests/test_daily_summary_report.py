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
                    "Price": 600.0,
                    "RSI": 71.0,
                    "ATR": 5.0,
                    "Reason": "Extended above the pullback zone.",
                },
            ],
        },
        "swing-options-debit-spread": {
            "strategy": "swing-options-debit-spread",
            "profile": "small_account_debit_spreads",
            "tickers": ["AAPL", "AMD"],
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
                    "Reason": "Base swing-options signal was HOLD. Tuned conversion upgraded the setup.",
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
                    "Reason": "Blocked because the setup is EXTENDED.",
                },
            ],
        },
    }


def test_build_daily_summary_identifies_debit_spread_setup():
    summary = build_daily_summary(scan_payload=_sample_scan_payload(), report_date="2026-05-05")

    assert summary["executive_decision"] == "One valid setup: AAPL debit spread"
    assert summary["market_state"] == "actionable"
    assert summary["actionable_count"] == 1
    assert summary["top_setup"]["ticker"] == "AAPL"
    assert summary["top_setup"]["conviction"] == "Medium"
    assert summary["small_account_options"][0]["ticker"] == "AAPL"
    assert summary["watchlist_names"][0]["ticker"] == "AAPL"
    assert summary["breadth_snapshot"] == {
        "actionable": 1,
        "watchlist": 1,
        "extended": 2,
        "weak": 0,
        "avoid": 0,
    }
    assert any(row["ticker"] == "SPY" for row in summary["ignore_list"])
    assert summary["debit_spread_context"] is not None
    assert summary["no_trade_reason"] is None
    assert summary["paper_execution_checklist"]
    assert "Planner output is not an executable order. It is only a candidate generator." in summary["manual_chain_confirmation_rules"]


def test_no_trade_reason_only_when_no_actionable():
    payload = _sample_scan_payload()
    payload["swing-options-debit-spread"]["results"][0]["Signal"] = "HOLD"
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
    assert "Conviction: Medium" in markdown
    assert "## Breadth Snapshot" in markdown
    assert "## Small Account Options" in markdown
    assert "AAPL | Bull Call Debit Spread 210/220" in markdown
    assert "## Debit Spread Historical Context" in markdown
    assert "## Manual Live Chain Confirmation Required" in markdown
    assert "Planner output is not an executable order. It is only a candidate generator." in markdown
    assert "## Paper Execution Checklist" in markdown
    assert "PROXY VALIDATION ONLY" in markdown

    paths = save_daily_summary(summary=summary, output_dir=str(tmp_path))
    assert paths["markdown"].exists()
    assert paths["json"].exists()

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["executive_decision"] == "One valid setup: AAPL debit spread"
