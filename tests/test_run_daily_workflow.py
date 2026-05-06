from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_daily_workflow.py"
SPEC = importlib.util.spec_from_file_location("run_daily_workflow", SCRIPT_PATH)
run_daily_workflow = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(run_daily_workflow)


def test_parse_strategies():
    assert run_daily_workflow._parse_strategies("ema-rsi,swing-options-debit-spread") == [
        "ema-rsi",
        "swing-options-debit-spread",
    ]


def test_parse_strategies_rejects_invalid():
    with pytest.raises(ValueError, match="Unsupported strategies"):
        run_daily_workflow._parse_strategies("ema-rsi,not-real")


def test_top_setup_prefers_small_account_debit_spread():
    scan_payload = {
        "swing-options-debit-spread": {
            "results": [
                {
                    "Ticker": "AAPL",
                    "Signal": "BUY",
                    "SmallAccountEligible": "YES",
                    "OptionStructure": "Bull Call Debit Spread 285/300",
                    "MaxLoss": 85.0,
                    "PremiumStatus": "OK",
                }
            ]
        },
        "four-hour-trend": {
            "results": [
                {"Ticker": "PLTR", "Signal": "SHORT_SETUP", "Setup": "ACTIONABLE", "Price": 135.75}
            ]
        },
        "ema-rsi": {"results": []},
    }

    top_setup = run_daily_workflow._top_setup_line(scan_payload)
    assert top_setup == "AAPL | swing-options-debit-spread | BUY | 285/300 debit spread | Max Risk $85"


def test_auto_open_does_not_crash(monkeypatch, capsys, tmp_path: Path):
    def _raise(*args, **kwargs):
        raise RuntimeError("open failed")

    monkeypatch.setattr(run_daily_workflow.subprocess, "run", _raise)
    markdown_path = tmp_path / "daily_summary.md"
    markdown_path.write_text("test", encoding="utf-8")

    run_daily_workflow._open_summary(markdown_path)
    captured = capsys.readouterr()
    assert "Unable to auto-open summary" in captured.out


def test_summary_only_fails_when_reports_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing scan reports. Run full workflow first."):
        run_daily_workflow.run_workflow(
            report_date="2026-05-05",
            strategies=["ema-rsi"],
            output_dir=str(tmp_path),
            no_journal=True,
            summary_only=True,
            skip_strategy_on_error=False,
        )


def test_skip_strategy_on_error_continues(monkeypatch):
    def _broken_scan(strategy: str, output_dir: str) -> dict:
        if strategy == "ema-rsi":
            raise RuntimeError("scan failed")
        return {
            "strategy": strategy,
            "profile": "test",
            "tickers": ["AAPL"],
            "results": [{"Ticker": "AAPL", "Strategy": strategy, "Signal": "HOLD", "Setup": "WAIT", "Price": 1.0, "Reason": "x"}],
        }

    monkeypatch.setattr(run_daily_workflow, "_scan_strategy", _broken_scan)

    scan_payload, failures = run_daily_workflow.run_workflow(
        report_date="2026-05-05",
        strategies=["ema-rsi", "swing-options-debit-spread"],
        output_dir="reports",
        no_journal=True,
        summary_only=False,
        skip_strategy_on_error=True,
    )

    assert "swing-options-debit-spread" in scan_payload
    assert failures == [{"strategy": "ema-rsi", "error": "scan failed"}]
