from __future__ import annotations

import json
from pathlib import Path

WATCHLIST_PROFILES = {
    "broad_market": [
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "XLK",
        "SMH",
        "AAPL",
        "MSFT",
        "NVDA",
        "META",
        "AMZN",
        "GOOGL",
        "AVGO",
        "COST",
        "NFLX",
        "CRM",
        "ORCL",
        "ADBE",
        "AMD",
        "TSM",
        "QCOM",
        "TXN",
        "PLTR",
        "COIN",
        "SHOP",
        "UBER",
        "PANW",
        "CRWD",
    ],
    "high_beta": [
        "NVDA",
        "AVGO",
        "META",
        "MSFT",
        "DIA",
        "PANW",
        "TSM",
        "GOOGL",
        "SPY",
    ],
    "options_momentum_core": [
        "SPY",
        "QQQ",
        "NVDA",
        "AVGO",
        "META",
        "AAPL",
        "MSFT",
    ],
    "swing_options_core": [
        "SPY",
        "QQQ",
        "NVDA",
        "AVGO",
        "META",
        "AAPL",
        "MSFT",
        "AMD",
    ],
    "small_account_options": [
        "SPY",
        "QQQ",
        "AAPL",
        "AMD",
    ],
    "custom": [],
}

DEFAULT_STRATEGY_PROFILES = {
    "ema-rsi": "broad_market",
    "four-hour-trend": "broad_market",
    "rsi-bollinger-v2": "high_beta",
    "options-momentum": "options_momentum_core",
    "swing-options": "swing_options_core",
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "watchlists.json"


def _normalize_tickers(tickers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for ticker in tickers:
        clean_ticker = str(ticker).strip().upper()
        if not clean_ticker or clean_ticker in seen:
            continue
        seen.add(clean_ticker)
        normalized.append(clean_ticker)

    return normalized


def _load_config_overrides(config_path: Path | None = None) -> dict[str, list[str]]:
    effective_config_path = config_path or DEFAULT_CONFIG_PATH
    if not effective_config_path.exists():
        return {}

    with effective_config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Watchlist config must be a JSON object: {effective_config_path}")

    overrides: dict[str, list[str]] = {}
    for profile_name, tickers in payload.items():
        if not isinstance(profile_name, str):
            raise ValueError(f"Watchlist config profile names must be strings: {effective_config_path}")
        if not isinstance(tickers, list):
            raise ValueError(f"Watchlist profile '{profile_name}' must map to a list of tickers.")
        overrides[profile_name] = _normalize_tickers(tickers)

    return overrides


def _resolved_profiles() -> dict[str, list[str]]:
    profiles = {name: list(tickers) for name, tickers in WATCHLIST_PROFILES.items()}
    profiles.update(_load_config_overrides())
    return profiles


def _parse_explicit_tickers(scan_arg: str | None) -> list[str]:
    if not scan_arg:
        return []
    return _normalize_tickers(scan_arg.split(","))


def get_watchlist(profile_name: str) -> list[str]:
    profiles = _resolved_profiles()
    if profile_name not in profiles:
        available_profiles = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown watchlist profile '{profile_name}'. Available profiles: {available_profiles}")
    return list(profiles[profile_name])


def get_default_watchlist_for_strategy(strategy: str) -> list[str]:
    profile_name = DEFAULT_STRATEGY_PROFILES.get(strategy)
    if profile_name is None:
        raise ValueError(f"No default watchlist profile configured for strategy '{strategy}'.")
    return get_watchlist(profile_name)


def parse_scan_universe(scan_arg: str | None, strategy: str, profile: str | None) -> list[str]:
    explicit_tickers = _parse_explicit_tickers(scan_arg)
    if explicit_tickers:
        return explicit_tickers

    if profile:
        return get_watchlist(profile)

    return get_default_watchlist_for_strategy(strategy)
