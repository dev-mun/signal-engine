from __future__ import annotations

import json
import ast
from pathlib import Path
from collections import Counter

import pandas as pd


NEAR_SETUP_STATUSES = {"WATCHLIST", "WAIT", "NEAR_SETUP", "NEEDS_PULLBACK", "OVERSOLD"}
IGNORE_STATUSES = {"EXTENDED", "WEAK_TREND", "AVOID", "NO_TRADE"}
IGNORE_PREMIUM_STATUSES = {"TOO_EXPENSIVE", "BAD_REWARD_RISK"}
WEAK_STATUSES = {"WEAK", "WEAK_TREND"}
DEBIT_SPREAD_PROXY_CONTEXT = {
    "trades_per_month": 1.08,
    "win_rate_proxy": 59.26,
    "profit_factor_proxy": 3.58,
    "max_drawdown_proxy": -226.0,
    "label": "PROXY VALIDATION ONLY",
}
LARGE_CAP_DEBIT_PROFILE = "small_account_debit_spreads"
GROWTH_DEBIT_PROFILE = "small_account_growth"
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


def _action_state(result: dict) -> str:
    explicit = str(result.get("ActionState", "")).strip()
    if explicit:
        return explicit

    signal = str(result.get("Signal", ""))
    setup = str(result.get("Setup", ""))
    if signal == "ERROR":
        return "ERROR"
    if signal in {"BUY", "SHORT_SETUP", "SELL", "SELL_SHORT", "BEARISH_ENTRY"}:
        return "ACTIONABLE"
    if setup == "NO_TRADE":
        return "NO_TRADE"
    if setup in {"EXTENDED", "WEAK_TREND", "AVOID"}:
        return "IGNORE"
    if setup in NEAR_SETUP_STATUSES:
        return "WATCHLIST"
    return "WATCHLIST"


def _final_score(result: dict) -> float:
    return float(result.get("FinalScore", result.get("SetupScore", result.get("Score", 0.0))) or 0.0)


def _market_regime_summary(scan_payload: dict[str, dict]) -> dict[str, str]:
    for profile in (GROWTH_DEBIT_PROFILE, LARGE_CAP_DEBIT_PROFILE):
        rows = _debit_profile_results(scan_payload, profile)
        for row in rows:
            regime = str(row.get("MarketRegime", "")).strip()
            if regime:
                return {
                    "regime": regime,
                    "reason": str(row.get("RegimeReason", "")).strip(),
                }

    for strategy_payload in scan_payload.values():
        for row in strategy_payload.get("results", []):
            regime = str(row.get("MarketRegime", "")).strip()
            if regime:
                return {
                    "regime": regime,
                    "reason": str(row.get("RegimeReason", "")).strip(),
                }

    return {"regime": "UNKNOWN", "reason": ""}


def _normalized_no_trade_reasons(scan_payload: dict[str, dict]) -> list[dict[str, int | str]]:
    counter: Counter[str] = Counter()
    reason_map = {
        "Price is too extended from EMA20.": "EXTENDED",
        "Expected move exhaustion is already present.": "EXPECTED_MOVE_EXHAUSTION",
        "ATR is too low.": "LOW_ATR",
        "Average volume is too weak.": "LOW_VOLUME",
        "Spread/liquidity is poor.": "LOW_LIQUIDITY",
        "Market regime strongly conflicts with the setup.": "REGIME_CONFLICT",
        "Earnings are too close.": "EARNINGS_SOON",
        "Daily trend is bearish, so bullish debit spreads are not allowed.": "DAILY_TREND_CONFLICT",
    }

    def _reason_labels(raw_reasons: object) -> list[str]:
        if raw_reasons is None:
            return []
        if isinstance(raw_reasons, str) and not raw_reasons.strip():
            return []
        if isinstance(raw_reasons, (list, tuple, set)) and not raw_reasons:
            return []

        parsed = raw_reasons
        if isinstance(raw_reasons, str):
            text = raw_reasons.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = text
            else:
                parsed = text

        if isinstance(parsed, str):
            if " | " in parsed:
                items = [part.strip() for part in parsed.split(" | ") if part.strip()]
            else:
                items = [parsed.strip()]
        elif isinstance(parsed, (list, tuple, set)):
            items = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            items = [str(parsed).strip()]

        labels: list[str] = []
        for item in items:
            label = reason_map.get(item, item.upper().replace(" ", "_"))
            if label:
                labels.append(label)
        return labels

    for strategy_payload in scan_payload.values():
        for result in strategy_payload["results"]:
            action_state = _action_state(result)
            if action_state not in {"NO_TRADE", "IGNORE"}:
                continue

            row_labels: set[str] = set()

            premium_status = str(result.get("PremiumStatus", "")).strip()
            if premium_status in IGNORE_PREMIUM_STATUSES:
                row_labels.add(premium_status)

            setup = str(result.get("Setup", "")).strip()
            if setup in {"EXTENDED", "WEAK_TREND", "AVOID"}:
                row_labels.add(setup)

            for label in _reason_labels(result.get("NoTradeReasons", [])):
                row_labels.add(label)

            for label in row_labels:
                counter[label] += 1

    return [{"reason": reason, "count": count} for reason, count in counter.most_common(3)]


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


def _debit_profile_results(scan_payload: dict[str, dict], profile_name: str) -> list[dict]:
    rows: list[dict] = []
    for strategy_payload in scan_payload.values():
        if str(strategy_payload.get("strategy", "")) != "swing-options-debit-spread":
            continue
        if str(strategy_payload.get("profile", "")) != profile_name:
            continue
        rows.extend(strategy_payload.get("results", []))
    return rows


def select_top_setup(scan_payload: dict[str, dict]) -> dict | None:
    for result in _debit_profile_results(scan_payload, GROWTH_DEBIT_PROFILE):
        if result.get("Signal") == "BUY" and str(result.get("SmallAccountEligible", "NO")) == "YES":
            return {
                "ticker": str(result["Ticker"]),
                "strategy": "swing-options-debit-spread",
                "profile": GROWTH_DEBIT_PROFILE,
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
                "market_regime": str(result.get("MarketRegime", "")),
                "final_score": _final_score(result),
                "setup_score": float(result.get("SetupScore", result.get("Score", 0.0)) or 0.0),
                "setup_rating": str(result.get("SetupRating", "")),
                "final_decision": str(result.get("FinalDecision", "")),
                "reason": str(result.get("Reason", "")),
            }

    for result in _debit_profile_results(scan_payload, LARGE_CAP_DEBIT_PROFILE):
        if result.get("Signal") == "BUY" and str(result.get("SmallAccountEligible", "NO")) == "YES":
            return {
                "ticker": str(result["Ticker"]),
                "strategy": "swing-options-debit-spread",
                "profile": LARGE_CAP_DEBIT_PROFILE,
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
                "market_regime": str(result.get("MarketRegime", "")),
                "final_score": _final_score(result),
                "setup_score": float(result.get("SetupScore", result.get("Score", 0.0)) or 0.0),
                "setup_rating": str(result.get("SetupRating", "")),
                "final_decision": str(result.get("FinalDecision", "")),
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
                "market_regime": str(result.get("MarketRegime", "")),
                "final_score": _final_score(result),
                "setup_score": float(result.get("SetupScore", result.get("Score", 0.0)) or 0.0),
                "setup_rating": str(result.get("SetupRating", "")),
                "final_decision": str(result.get("FinalDecision", "")),
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
                "market_regime": str(result.get("MarketRegime", "")),
                "final_score": _final_score(result),
                "setup_score": float(result.get("SetupScore", result.get("Score", 0.0)) or 0.0),
                "setup_rating": str(result.get("SetupRating", "")),
                "final_decision": str(result.get("FinalDecision", "")),
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
    return _action_state(result) == "ACTIONABLE"


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
                    "action_state": _action_state(result),
                    "conviction": _conviction_label(result),
                    "market_regime": str(result.get("MarketRegime", "")),
                    "final_score": _final_score(result),
                    "setup_score": float(result.get("SetupScore", result.get("Score", 0.0)) or 0.0),
                    "setup_rating": str(result.get("SetupRating", "")),
                    "final_decision": str(result.get("FinalDecision", "")),
                    "reason": str(result.get("Reason", "")),
                }
            )
    return rows


def _small_account_options_rows(scan_payload: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for result in _debit_profile_results(scan_payload, GROWTH_DEBIT_PROFILE):
        if result.get("Signal") != "BUY":
            continue
        rows.append(
            {
                "ticker": str(result.get("Ticker", "")),
                "profile": GROWTH_DEBIT_PROFILE,
                "setup": str(result.get("Setup", "")),
                "spread_structure": str(result.get("OptionStructure", "N/A")),
                "debit": float(result.get("EstDebit", 0.0) or 0.0),
                "max_loss": float(result.get("MaxLoss", 0.0) or 0.0),
                "reward_risk": float(result.get("RewardRisk", 0.0) or 0.0),
                "small_account_eligible": str(result.get("SmallAccountEligible", "NO")),
                "action_state": _action_state(result),
                "market_regime": str(result.get("MarketRegime", "")),
                "final_score": _final_score(result),
                "setup_score": float(result.get("SetupScore", result.get("Score", 0.0)) or 0.0),
                "setup_rating": str(result.get("SetupRating", "")),
                "final_decision": str(result.get("FinalDecision", "")),
                "reason": str(result.get("Reason", "")),
            }
        )
    return rows


def _large_cap_debit_rows(scan_payload: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for result in _debit_profile_results(scan_payload, LARGE_CAP_DEBIT_PROFILE):
        if result.get("Signal") != "BUY":
            continue
        rows.append(
            {
                "ticker": str(result.get("Ticker", "")),
                "profile": LARGE_CAP_DEBIT_PROFILE,
                "setup": str(result.get("Setup", "")),
                "spread_structure": str(result.get("OptionStructure", "N/A")),
                "debit": float(result.get("EstDebit", 0.0) or 0.0),
                "max_loss": float(result.get("MaxLoss", 0.0) or 0.0),
                "reward_risk": float(result.get("RewardRisk", 0.0) or 0.0),
                "small_account_eligible": str(result.get("SmallAccountEligible", "NO")),
                "action_state": _action_state(result),
                "market_regime": str(result.get("MarketRegime", "")),
                "final_score": _final_score(result),
                "setup_score": float(result.get("SetupScore", result.get("Score", 0.0)) or 0.0),
                "setup_rating": str(result.get("SetupRating", "")),
                "final_decision": str(result.get("FinalDecision", "")),
                "reason": str(result.get("Reason", "")),
            }
        )
    return rows


def _ignore_key(result: dict) -> tuple[str, str]:
    return (str(result.get("Ticker", "")), str(result.get("Strategy", "")))


def _watchlist_rows(scan_payload: dict[str, dict], ignore_pairs: set[tuple[str, str]]) -> list[dict]:
    rows: list[dict] = []
    for strategy_payload in scan_payload.values():
        for result in strategy_payload["results"]:
            action_state = _action_state(result)
            if action_state == "ERROR":
                continue
            if action_state != "WATCHLIST":
                continue
            if _ignore_key(result) in ignore_pairs:
                continue
            setup = str(result.get("Setup", ""))
            premium_status = str(result.get("PremiumStatus", ""))
            if setup in IGNORE_STATUSES or premium_status in IGNORE_PREMIUM_STATUSES:
                continue
            rows.append(
                {
                    "ticker": str(result.get("Ticker", "")),
                    "strategy": str(result.get("Strategy", "")),
                    "action_state": action_state,
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
            action_state = _action_state(result)
            if action_state == "ERROR":
                continue
            setup = str(result.get("Setup", ""))
            if action_state == "ACTIONABLE":
                actionable += 1
            elif action_state == "WATCHLIST":
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
            action_state = _action_state(result)
            if action_state == "ERROR":
                continue
            setup = str(result.get("Setup", ""))
            premium_status = str(result.get("PremiumStatus", ""))
            if action_state not in {"IGNORE", "NO_TRADE"} and premium_status not in IGNORE_PREMIUM_STATUSES:
                continue
            rows.append(
                {
                    "ticker": str(result.get("Ticker", "")),
                    "strategy": str(result.get("Strategy", "")),
                    "action_state": action_state,
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
    large_cap_debit_context = _large_cap_debit_rows(scan_payload)
    ignore = _ignore_rows(scan_payload)
    ignore_pairs = {(row["ticker"], row["strategy"]) for row in ignore}
    watchlist = _watchlist_rows(scan_payload, ignore_pairs=ignore_pairs)
    top_setup = select_top_setup(scan_payload)
    breadth_snapshot = _breadth_snapshot(scan_payload)
    market_regime = _market_regime_summary(scan_payload)
    market_state = _market_state(scan_payload, actionable, watchlist, ignore)
    executive_decision = _executive_decision(actionable, small_account_options, watchlist)
    tomorrow_plan = _tomorrow_plan(executive_decision, actionable, small_account_options, watchlist)
    workflow_failures = failures or []
    no_trade_reason = _no_trade_reason(len(actionable))
    key_no_trade_reasons = _normalized_no_trade_reasons(scan_payload)
    debit_spread_context = DEBIT_SPREAD_PROXY_CONTEXT if (small_account_options or large_cap_debit_context) else None
    paper_execution_checklist = _paper_execution_checklist(len(actionable))

    return {
        "report_date": report_date,
        "executive_decision": executive_decision,
        "top_setup": top_setup,
        "breadth_snapshot": breadth_snapshot,
        "market_regime": market_regime,
        "market_state": market_state,
        "actionable_signals": actionable,
        "large_cap_debit_context": large_cap_debit_context,
        "small_account_options": small_account_options,
        "debit_spread_context": debit_spread_context,
        "watchlist_names": watchlist,
        "ignore_list": ignore,
        "key_no_trade_reasons": key_no_trade_reasons,
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
        if top_setup["strategy"] == "swing-options-debit-spread":
            max_risk_text = "N/A" if top_setup["max_risk"] is None else f"${top_setup['max_risk']:.0f}"
            reward_risk_text = "N/A" if top_setup["reward_risk"] is None else f"{top_setup['reward_risk']:.2f}"
        else:
            max_risk_text = "Not calculated - directional setup only"
            reward_risk_text = "Not calculated - no options structure generated"
        lines.extend(
            [
                f"Ticker: {top_setup['ticker']}",
                f"Strategy: {top_setup['strategy']}",
                f"Structure: {top_setup['structure']}",
                f"Market Regime: {top_setup.get('market_regime') or 'UNKNOWN'}",
                f"Final Score: {top_setup.get('final_score', 0.0):.2f}",
                f"Setup Score: {top_setup.get('setup_score', 0.0):.2f}",
                f"Setup Rating: {top_setup.get('setup_rating') or 'N/A'}",
                f"Max Risk: {max_risk_text}",
                f"Reward/Risk: {reward_risk_text}",
                f"Status: {top_setup['status']}",
                f"Conviction: {top_setup['conviction']}",
                f"Final Decision: {top_setup.get('final_decision') or 'N/A'}",
                f"Reason: {top_setup['reason']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Market Regime",
            f"Regime: {summary['market_regime'].get('regime', 'UNKNOWN')}",
            f"Reason: {summary['market_regime'].get('reason', '') or 'UNKNOWN'}",
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
                f"{row['price']:.2f} | Regime: {row['market_regime'] or 'N/A'} | "
                f"FinalScore: {row['final_score']:.2f} | Rating: {row['setup_rating'] or 'N/A'} | "
                f"Conviction: {row['conviction']} | {row['reason']} | Decision: {row['final_decision']}"
            )

    lines.extend(["", "## Large-Cap Debit Spread Context"])
    if not summary["large_cap_debit_context"]:
        lines.append("No large-cap debit spread BUY candidates.")
    else:
        for row in summary["large_cap_debit_context"]:
            lines.append(
                f"- {row['ticker']} | {row['spread_structure']} | Debit {row['debit']:.2f} | "
                f"Max Loss {row['max_loss']:.2f} | Reward/Risk {row['reward_risk']:.2f} | "
                f"Regime: {row['market_regime'] or 'N/A'} | FinalScore: {row['final_score']:.2f} | "
                f"Rating: {row['setup_rating'] or 'N/A'} | Eligible: {row['small_account_eligible']} | "
                f"{row['reason']} | Decision: {row['final_decision']}"
            )

    lines.extend(["", "## Small-Account Growth Debit Spread Candidates"])
    if not summary["small_account_options"]:
        lines.append("No small-account growth debit spread BUY candidates.")
    else:
        for row in summary["small_account_options"]:
            lines.append(
                f"- {row['ticker']} | {row['spread_structure']} | Debit {row['debit']:.2f} | "
                f"Max Loss {row['max_loss']:.2f} | Reward/Risk {row['reward_risk']:.2f} | "
                f"Regime: {row['market_regime'] or 'N/A'} | FinalScore: {row['final_score']:.2f} | "
                f"Rating: {row['setup_rating'] or 'N/A'} | Eligible: {row['small_account_eligible']} | "
                f"{row['reason']} | Decision: {row['final_decision']}"
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

    lines.extend(["", "## Key No-Trade Reasons"])
    if not summary["key_no_trade_reasons"]:
        lines.append("None")
    else:
        for row in summary["key_no_trade_reasons"]:
            lines.append(f"- {row['reason']}: {row['count']}")

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
