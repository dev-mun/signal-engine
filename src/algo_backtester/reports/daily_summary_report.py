from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


NEAR_SETUP_STATUSES = {"WATCHLIST", "WAIT", "NEAR_SETUP", "NEEDS_PULLBACK", "OVERSOLD"}
IGNORE_STATUSES = {"EXTENDED", "WEAK_TREND", "AVOID"}
IGNORE_PREMIUM_STATUSES = {"TOO_EXPENSIVE", "BAD_REWARD_RISK"}
WEAK_STATUSES = {"WEAK", "WEAK_TREND"}
DEBIT_SPREAD_PROXY_CONTEXT = {
    "trades_per_month": 1.08,
    "win_rate_proxy": 59.26,
    "profit_factor_proxy": 3.58,
    "max_drawdown_proxy": -226.0,
    "label": "PROXY VALIDATION ONLY",
}
MANUAL_CHAIN_CONFIRMATION_RULES = [
    "Before any paper or live trade, manually confirm the spread in Fidelity.",
    "Match the same ticker.",
    "Use the same expiration range, with 30-45 DTE preferred.",
    "Use the same long/short strikes or the closest liquid equivalent.",
    "Confirm the actual debit.",
    "Confirm the actual max loss.",
    "Confirm the actual max profit.",
    "Confirm the bid/ask spread.",
    "Confirm volume and open interest.",
    "Confirm reward/risk is at least 1.5.",
    "Confirm total debit stays within the account cap.",
    "Planner output is not an executable order. It is only a candidate generator.",
    "If the real debit differs materially from the estimate, use real broker pricing and either skip the trade or record it as planner mismatch.",
]


def _format_debit_spread_structure(structure: str) -> str:
    prefix = "Bull Call Debit Spread "
    if structure.startswith(prefix):
        strikes = structure[len(prefix):]
        parts = []
        for value in strikes.split("/"):
            clean = value.strip()
            if clean.endswith(".00"):
                clean = clean[:-3]
            parts.append(clean)
        return "/".join(parts)
    return structure


def _conviction_label(result: dict) -> str:
    strategy = str(result.get("Strategy", ""))
    signal = str(result.get("Signal", ""))
    setup = str(result.get("Setup", ""))

    if strategy == "swing-options-debit-spread" and signal == "BUY":
        premium_status = str(result.get("PremiumStatus", ""))
        small_account = str(result.get("SmallAccountEligible", "NO"))
        if small_account == "YES" and premium_status == "OK":
            return "Medium"
        if premium_status == "ACCEPTABLE":
            return "Low/Medium"
        return "Low/Medium"

    if strategy == "four-hour-trend" and signal == "SHORT_SETUP":
        return "Tactical"

    if strategy == "ema-rsi" and signal == "BUY":
        return "High" if setup == "ACTIONABLE" else "Medium"

    if signal == "BUY":
        return "Medium"

    return "Tactical"


def select_top_setup(scan_payload: dict[str, dict]) -> dict | None:
    spread_results = scan_payload.get("swing-options-debit-spread", {}).get("results", [])
    for result in spread_results:
        if result.get("Signal") == "BUY" and str(result.get("SmallAccountEligible", "NO")) == "YES":
            return {
                "ticker": str(result["Ticker"]),
                "strategy": "swing-options-debit-spread",
                "signal": "BUY",
                "structure": f"{_format_debit_spread_structure(str(result.get('OptionStructure', 'N/A')))} Bull Call Debit Spread",
                "display": (
                    f"{result['Ticker']} | swing-options-debit-spread | BUY | "
                    f"{_format_debit_spread_structure(str(result.get('OptionStructure', 'N/A')))} debit spread | "
                    f"Max Risk ${float(result.get('MaxLoss', 0.0) or 0.0):.0f}"
                ),
                "max_risk": float(result.get("MaxLoss", 0.0) or 0.0),
                "reward_risk": float(result.get("RewardRisk", 0.0) or 0.0),
                "status": "Small-account eligible",
                "conviction": _conviction_label(result),
                "reason": str(result.get("Reason", "")),
            }

    four_hour_results = scan_payload.get("four-hour-trend", {}).get("results", [])
    for result in four_hour_results:
        if result.get("Signal") in {"BUY", "SHORT_SETUP"} and str(result.get("Setup", "")) == "ACTIONABLE":
            return {
                "ticker": str(result["Ticker"]),
                "strategy": "four-hour-trend",
                "signal": str(result["Signal"]),
                "structure": str(result.get("Setup", "")),
                "display": (
                    f"{result['Ticker']} | four-hour-trend | {result['Signal']} | "
                    f"{result.get('Setup', '')} | Price {float(result.get('Price', 0.0) or 0.0):.2f}"
                ),
                "max_risk": None,
                "reward_risk": None,
                "status": "Actionable",
                "conviction": _conviction_label(result),
                "reason": str(result.get("Reason", "")),
            }

    ema_results = scan_payload.get("ema-rsi", {}).get("results", [])
    for result in ema_results:
        if result.get("Signal") == "BUY":
            return {
                "ticker": str(result["Ticker"]),
                "strategy": "ema-rsi",
                "signal": "BUY",
                "structure": str(result.get("Setup", "")),
                "display": (
                    f"{result['Ticker']} | ema-rsi | BUY | "
                    f"{result.get('Setup', '')} | Price {float(result.get('Price', 0.0) or 0.0):.2f}"
                ),
                "max_risk": None,
                "reward_risk": None,
                "status": "Actionable",
                "conviction": _conviction_label(result),
                "reason": str(result.get("Reason", "")),
            }

    return None


def _direction_for_result(result: dict) -> str:
    signal = str(result.get("Signal", ""))
    strategy = str(result.get("Strategy", ""))

    if strategy == "swing-options-debit-spread" and signal == "BUY":
        return "bullish options"
    if signal in {"SHORT_SETUP", "SELL_SHORT", "BEARISH_ENTRY"}:
        return "bearish"
    if signal == "BUY":
        return "bullish"
    return "neutral"


def _is_actionable_trade(result: dict) -> bool:
    signal = str(result.get("Signal", ""))
    strategy = str(result.get("Strategy", ""))

    if strategy == "swing-options-debit-spread":
        return signal == "BUY"

    return signal in {"BUY", "SHORT_SETUP", "SELL_SHORT", "BEARISH_ENTRY"}


def _actionable_signal_rows(scan_payload: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for strategy_payload in scan_payload.values():
        for result in strategy_payload["results"]:
            if result.get("Signal") == "ERROR" or not _is_actionable_trade(result):
                continue
            rows.append(
                {
                    "ticker": str(result.get("Ticker", "")),
                    "strategy": str(result.get("Strategy", "")),
                    "direction": _direction_for_result(result),
                    "setup": str(result.get("Setup", "")),
                    "price": float(result.get("Price", 0.0) or 0.0),
                    "conviction": _conviction_label(result),
                    "reason": str(result.get("Reason", "")),
                }
            )
    return rows


def _small_account_options_rows(scan_payload: dict[str, dict]) -> list[dict]:
    strategy_payload = scan_payload.get("swing-options-debit-spread")
    if strategy_payload is None:
        return []

    rows: list[dict] = []
    for result in strategy_payload["results"]:
        if result.get("Signal") != "BUY":
            continue
        rows.append(
            {
                "ticker": str(result.get("Ticker", "")),
                "setup": str(result.get("Setup", "")),
                "spread_structure": str(result.get("OptionStructure", "N/A")),
                "debit": float(result.get("EstDebit", 0.0) or 0.0),
                "max_loss": float(result.get("MaxLoss", 0.0) or 0.0),
                "reward_risk": float(result.get("RewardRisk", 0.0) or 0.0),
                "small_account_eligible": str(result.get("SmallAccountEligible", "NO")),
                "reason": str(result.get("Reason", "")),
            }
        )
    return rows


def _watchlist_rows(scan_payload: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for strategy_payload in scan_payload.values():
        for result in strategy_payload["results"]:
            if result.get("Signal") == "ERROR":
                continue
            if _is_actionable_trade(result):
                continue
            setup = str(result.get("Setup", ""))
            if setup not in NEAR_SETUP_STATUSES:
                continue
            rows.append(
                {
                    "ticker": str(result.get("Ticker", "")),
                    "strategy": str(result.get("Strategy", "")),
                    "setup": setup,
                    "reason": str(result.get("Reason", "")),
                }
            )
    return rows


def _breadth_snapshot(scan_payload: dict[str, dict]) -> dict[str, int]:
    actionable = 0
    watchlist = 0
    extended = 0
    weak = 0
    avoid = 0

    for strategy_payload in scan_payload.values():
        for result in strategy_payload["results"]:
            if result.get("Signal") == "ERROR":
                continue
            setup = str(result.get("Setup", ""))
            if _is_actionable_trade(result):
                actionable += 1
            elif setup in NEAR_SETUP_STATUSES:
                watchlist += 1

            if setup == "EXTENDED":
                extended += 1
            if setup in WEAK_STATUSES:
                weak += 1
            if setup == "AVOID":
                avoid += 1

    return {
        "actionable": actionable,
        "watchlist": watchlist,
        "extended": extended,
        "weak": weak,
        "avoid": avoid,
    }


def _no_trade_reason(actionable_count: int) -> str | None:
    if actionable_count > 0:
        return None
    return (
        "Most setups remain EXTENDED, WAIT, or NEEDS_PULLBACK. "
        "No high-quality entry with acceptable risk was found today. "
        "Capital preservation takes priority."
    )


def _paper_execution_checklist(actionable_count: int) -> list[str]:
    if actionable_count <= 0:
        return []
    return [
        "Open Fidelity chain.",
        "Select 30-45 DTE.",
        "Build the same spread.",
        "Confirm debit is within the account cap.",
        "Confirm reward/risk is at least 1.5.",
        "Confirm liquidity.",
        "Confirm the setup is still valid.",
        "Paper trade only if all checks pass.",
    ]


def _ignore_rows(scan_payload: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for strategy_payload in scan_payload.values():
        for result in strategy_payload["results"]:
            if result.get("Signal") == "ERROR":
                continue
            setup = str(result.get("Setup", ""))
            premium_status = str(result.get("PremiumStatus", ""))
            if setup not in IGNORE_STATUSES and premium_status not in IGNORE_PREMIUM_STATUSES:
                continue
            rows.append(
                {
                    "ticker": str(result.get("Ticker", "")),
                    "strategy": str(result.get("Strategy", "")),
                    "setup": setup,
                    "reason": str(result.get("Reason", "")),
                }
            )
    return rows


def _market_state(scan_payload: dict[str, dict], actionable: list[dict], watchlist: list[dict], ignore: list[dict]) -> str:
    valid_results = [
        result
        for strategy_payload in scan_payload.values()
        for result in strategy_payload["results"]
        if result.get("Signal") != "ERROR"
    ]
    if not valid_results:
        return "mixed"

    if actionable:
        return "actionable"

    extended_count = sum(1 for result in valid_results if str(result.get("Setup", "")) == "EXTENDED")
    weak_count = sum(1 for result in valid_results if str(result.get("Setup", "")) in {"WEAK_TREND", "AVOID"})

    if extended_count >= max(2, len(valid_results) // 2):
        return "extended"
    if weak_count >= max(2, len(valid_results) // 2):
        return "weak"
    if watchlist and not actionable:
        return "pullback forming"
    if ignore and watchlist:
        return "mixed"
    return "mixed"


def _executive_decision(actionable: list[dict], small_account_options: list[dict], watchlist: list[dict]) -> str:
    if len(small_account_options) == 1:
        return f"One valid setup: {small_account_options[0]['ticker']} debit spread"
    if len(small_account_options) > 1:
        tickers = ", ".join(row["ticker"] for row in small_account_options[:3])
        return f"Multiple valid debit spreads: {tickers}"

    bearish = [row for row in actionable if row["direction"] == "bearish"]
    bullish = [row for row in actionable if row["direction"] != "bearish"]

    if bearish and not bullish:
        if len(bearish) == 1:
            return f"Tactical short only: {bearish[0]['ticker']}"
        return "Tactical shorts only"

    if bullish:
        tickers = ", ".join(row["ticker"] for row in bullish[:3])
        return f"Actionable setups present: {tickers}"

    if watchlist:
        return "Wait for pullback"

    return "No trade today"


def _tomorrow_plan(executive_decision: str, actionable: list[dict], small_account_options: list[dict], watchlist: list[dict]) -> list[str]:
    if small_account_options:
        tickers = ", ".join(row["ticker"] for row in small_account_options)
        return [
            f"Deep dive before the open: {tickers}.",
            "Check the live debit spread chain, confirm total debit stays at or below the small-account cap, and keep reward/risk at or above 1.5.",
            "Skip the trade if the signal degrades, the spread widens materially, or liquidity is poor.",
            "One trade allowed only if the setup remains valid at the open.",
        ]

    if actionable:
        tickers = ", ".join(row["ticker"] for row in actionable[:5])
        return [
            f"Deep dive before the open: {tickers}.",
            "Only allow a trade if the actionable signal remains intact after the open.",
            "Skip any name that opens extended or invalidates the setup.",
            "No small-account options trade unless a debit spread signal remains BUY.",
        ]

    if watchlist:
        tickers = ", ".join(row["ticker"] for row in watchlist[:5])
        return [
            f"Monitor watchlist names for confirmation: {tickers}.",
            "Do not force a trade at the open.",
            "Skip names that stay extended, weak, or fail to improve.",
            "No trade allowed unless a fresh actionable signal appears.",
        ]

    return [
        "No trade allowed at the open.",
        "Skip all names unless a fresh actionable signal appears.",
        "Do not force entries in a non-actionable market state.",
    ]


def build_daily_summary(scan_payload: dict[str, dict], report_date: str, failures: list[dict] | None = None) -> dict:
    actionable = _actionable_signal_rows(scan_payload)
    small_account_options = _small_account_options_rows(scan_payload)
    watchlist = _watchlist_rows(scan_payload)
    ignore = _ignore_rows(scan_payload)
    top_setup = select_top_setup(scan_payload)
    breadth_snapshot = _breadth_snapshot(scan_payload)
    market_state = _market_state(scan_payload, actionable, watchlist, ignore)
    executive_decision = _executive_decision(actionable, small_account_options, watchlist)
    tomorrow_plan = _tomorrow_plan(executive_decision, actionable, small_account_options, watchlist)
    workflow_failures = failures or []
    no_trade_reason = _no_trade_reason(len(actionable))
    debit_spread_context = DEBIT_SPREAD_PROXY_CONTEXT if small_account_options else None
    paper_execution_checklist = _paper_execution_checklist(len(actionable))

    return {
        "report_date": report_date,
        "executive_decision": executive_decision,
        "top_setup": top_setup,
        "breadth_snapshot": breadth_snapshot,
        "market_state": market_state,
        "actionable_signals": actionable,
        "small_account_options": small_account_options,
        "debit_spread_context": debit_spread_context,
        "watchlist_names": watchlist,
        "ignore_list": ignore,
        "no_trade_reason": no_trade_reason,
        "tomorrow_plan": tomorrow_plan,
        "manual_chain_confirmation_rules": MANUAL_CHAIN_CONFIRMATION_RULES,
        "paper_execution_checklist": paper_execution_checklist,
        "risk_notes": [
            "No forcing trades.",
            "No trade unless the signal remains valid at the open.",
            "One position at a time.",
            "Respect the small-account max debit cap.",
        ],
        "failures": workflow_failures,
        "actionable_count": len(actionable),
        "scan_payload": scan_payload,
    }


def render_daily_summary_markdown(summary: dict) -> str:
    lines = [
        f"# Daily Trading Summary - {summary['report_date']}",
        "",
        "## Executive Decision",
        summary["executive_decision"],
        "",
        "## Top Setup",
    ]

    top_setup = summary.get("top_setup")
    if top_setup is None:
        lines.append("None")
    else:
        max_risk_text = "N/A" if top_setup["max_risk"] is None else f"${top_setup['max_risk']:.0f}"
        reward_risk_text = "N/A" if top_setup["reward_risk"] is None else f"{top_setup['reward_risk']:.2f}"
        lines.extend(
            [
                f"Ticker: {top_setup['ticker']}",
                f"Strategy: {top_setup['strategy']}",
                f"Structure: {top_setup['structure']}",
                f"Max Risk: {max_risk_text}",
                f"Reward/Risk: {reward_risk_text}",
                f"Status: {top_setup['status']}",
                f"Conviction: {top_setup['conviction']}",
                f"Reason: {top_setup['reason']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Breadth Snapshot",
            "",
            f"Actionable: {summary['breadth_snapshot']['actionable']}",
            f"Watchlist: {summary['breadth_snapshot']['watchlist']}",
            f"Extended: {summary['breadth_snapshot']['extended']}",
            f"Weak/Weak Trend: {summary['breadth_snapshot']['weak']}",
            f"Avoid: {summary['breadth_snapshot']['avoid']}",
            "",
        ]
    )

    lines.extend([
        "## Market State",
        f"Market state: {summary['market_state']}.",
        "",
        "## Actionable Signals",
    ])

    if not summary["actionable_signals"]:
        lines.append("No real actionable signals.")
    else:
        for row in summary["actionable_signals"]:
            lines.append(
                f"- {row['ticker']} | {row['strategy']} | {row['direction']} | {row['setup']} | "
                f"{row['price']:.2f} | Conviction: {row['conviction']} | {row['reason']}"
            )

    lines.extend(["", "## Small Account Options"])
    if not summary["small_account_options"]:
        lines.append("No small-account debit spread BUY candidates.")
    else:
        for row in summary["small_account_options"]:
            lines.append(
                f"- {row['ticker']} | {row['spread_structure']} | Debit {row['debit']:.2f} | "
                f"Max Loss {row['max_loss']:.2f} | Reward/Risk {row['reward_risk']:.2f} | "
                f"Eligible: {row['small_account_eligible']} | {row['reason']}"
            )

    if summary.get("debit_spread_context"):
        context = summary["debit_spread_context"]
        lines.extend(
            [
                "",
                "## Debit Spread Historical Context",
                f"{context['label']}",
                f"- Tuned debit spread proxy trades/month: {context['trades_per_month']:.2f}",
                f"- Proxy win rate: {context['win_rate_proxy']:.2f}%",
                f"- Proxy profit factor: {context['profit_factor_proxy']:.2f}",
                f"- Proxy max drawdown: {context['max_drawdown_proxy']:.0f}",
            ]
        )

    lines.extend(["", "## Manual Live Chain Confirmation Required"])
    for row in summary["manual_chain_confirmation_rules"]:
        lines.append(f"- {row}")

    lines.extend(["", "## Watchlist Names"])
    if not summary["watchlist_names"]:
        lines.append("No near-setup watchlist names.")
    else:
        for row in summary["watchlist_names"]:
            lines.append(f"- {row['ticker']} | {row['strategy']} | {row['setup']} | {row['reason']}")

    lines.extend(["", "## Ignore List"])
    if not summary["ignore_list"]:
        lines.append("No names explicitly on the ignore list.")
    else:
        for row in summary["ignore_list"]:
            lines.append(f"- {row['ticker']} | {row['strategy']} | {row['setup']} | {row['reason']}")

    lines.extend(["", "## Workflow Failures"])
    if not summary.get("failures"):
        lines.append("No workflow failures.")
    else:
        for row in summary["failures"]:
            lines.append(f"- {row['strategy']} | {row['error']}")

    if summary.get("no_trade_reason"):
        lines.extend(["", "## No-Trade Reason", summary["no_trade_reason"]])

    if summary.get("paper_execution_checklist"):
        lines.extend(["", "## Paper Execution Checklist"])
        for row in summary["paper_execution_checklist"]:
            lines.append(f"- {row}")

    lines.extend(["", "## Tomorrow Plan"])
    for row in summary["tomorrow_plan"]:
        lines.append(f"- {row}")

    lines.extend(["", "## Risk Notes"])
    for row in summary["risk_notes"]:
        lines.append(f"- {row}")

    return "\n".join(lines) + "\n"


def save_daily_summary(summary: dict, output_dir: str = "reports") -> dict[str, Path]:
    output_path = Path(output_dir) / "daily" / summary["report_date"]
    output_path.mkdir(parents=True, exist_ok=True)

    markdown_path = output_path / "daily_summary.md"
    json_path = output_path / "daily_summary.json"

    markdown_path.write_text(render_daily_summary_markdown(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "markdown": markdown_path,
        "json": json_path,
    }
