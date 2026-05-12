from __future__ import annotations

import json
from pathlib import Path

# Metals are routed separately from high-beta equity momentum systems. SLV remains
# available as a tactical options vehicle, while GLD is reserved as macro exposure
# until premium and liquidity filters can safely govern small-account deployment.
# TODO: Add regime-aware ticker routing so profiles can adapt to macro state shifts.
# TODO: Add IV-aware filtering before metals tickers are promoted into options scans.
# TODO: Add options liquidity scoring for ETF-specific chain quality checks.
# TODO: Add max premium filters before GLD or other higher-cost underlyings enter small-account profiles.
# TODO: Add sector risk exposure controls across overlapping watchlist profiles.
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
        "SLV",
    ],
    "metals_macro": [
        "GLD",
        "SLV",
        "GDX",
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
        "SLV",
    ],
    # SLV stays in the tactical small-account options basket because it is liquid,
    # relatively affordable, and usable for directional call structures.
    "small_account_options": [
        "SPY",
        "QQQ",
        "AAPL",
        "AMD",
        "SLV",
    ],
    # GLD stays out of small-account debit spreads until premium filtering logic exists.
    "small_account_debit_spreads": [
        "SPY",
        "QQQ",
        "AAPL",
        "AMD",
        "SLV",
    ],
    "small_account_growth": [
        "PLTR",
        "UBER",
        "SOFI",
        "HOOD",
        "PYPL",
        "RIVN",
        "DKNG",
        "AFRM",
        "HIMS",
        "CLSK",
        "IONQ",
        "TOST",
        "SNAP",
        "SHOP",
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
    "swing-options-debit-spread": "small_account_debit_spreads",
    "metals-macro": "metals_macro",
    "metals-breakout": "metals_macro",
    "metals-mean-reversion": "metals_macro",
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "watchlists.json"
DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[2] / "profiles"


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


def _load_profile_overrides(profile_dir: Path | None = None) -> dict[str, list[str]]:
    effective_profile_dir = profile_dir or DEFAULT_PROFILE_DIR
    if not effective_profile_dir.exists():
        return {}

    overrides: dict[str, list[str]] = {}
    for profile_path in sorted(effective_profile_dir.glob("*.json")):
        with profile_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            tickers = payload
        elif isinstance(payload, dict) and isinstance(payload.get("tickers"), list):
            tickers = payload["tickers"]
        else:
            raise ValueError(
                f"Watchlist profile file must be a JSON list or an object with a 'tickers' list: {profile_path}"
            )

        overrides[profile_path.stem] = _normalize_tickers(tickers)

    return overrides


def _resolved_profiles() -> dict[str, list[str]]:
    profiles = {name: list(tickers) for name, tickers in WATCHLIST_PROFILES.items()}
    profiles.update(_load_config_overrides())
    profiles.update(_load_profile_overrides())
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
