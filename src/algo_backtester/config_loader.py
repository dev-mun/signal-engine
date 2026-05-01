from __future__ import annotations

import json
from pathlib import Path

BUILTIN_RSI_BOLLINGER_V2_PROFILES = {
    "default": {
        "rsi_threshold": 38,
        "stop_loss": 0.04,
        "take_profit": 0.05,
        "trailing_stop": 0.04,
        "max_hold_days": 7,
        "volume_multiplier": 0.6,
        "band_tolerance": 1.02,
        "close_position_min": 0.35,
    },
    "NVDA": {
        "rsi_threshold": 38,
        "stop_loss": 0.04,
        "take_profit": 0.04,
        "trailing_stop": 0.04,
        "max_hold_days": 10,
        "volume_multiplier": 0.8,
        "band_tolerance": 1.02,
        "close_position_min": 0.35,
    },
    "AVGO": {
        "rsi_threshold": 42,
        "stop_loss": 0.04,
        "take_profit": 0.06,
        "trailing_stop": 0.05,
        "max_hold_days": 5,
        "volume_multiplier": 0.5,
        "band_tolerance": 1.01,
        "close_position_min": 0.35,
    },
    "META": {
        "rsi_threshold": 38,
        "stop_loss": 0.05,
        "take_profit": 0.06,
        "trailing_stop": 0.05,
        "max_hold_days": 10,
        "volume_multiplier": 0.6,
        "band_tolerance": 1.03,
        "close_position_min": 0.30,
    },
}

DEFAULT_RSI_BOLLINGER_V2_PROFILES_PATH = Path(__file__).resolve().parents[2] / "config" / "rsi_bollinger_v2_profiles.json"


def _normalize_profile_name(profile_name: str) -> str:
    clean_name = str(profile_name).strip()
    if clean_name.lower() == "default":
        return "default"
    return clean_name.upper()


def _normalize_profile_payload(payload: dict[str, object]) -> dict[str, dict[str, float | int]]:
    normalized: dict[str, dict[str, float | int]] = {}

    for profile_name, profile_values in payload.items():
        if not isinstance(profile_name, str):
            raise ValueError("RSI Bollinger V2 profile names must be strings.")
        if not isinstance(profile_values, dict):
            raise ValueError(f"RSI Bollinger V2 profile '{profile_name}' must map to an object.")
        normalized[_normalize_profile_name(profile_name)] = dict(profile_values)

    return normalized


def load_rsi_bollinger_v2_profiles(config_path: Path | None = None) -> dict[str, dict[str, float | int]]:
    profiles = {
        profile_name: dict(profile_values)
        for profile_name, profile_values in BUILTIN_RSI_BOLLINGER_V2_PROFILES.items()
    }

    effective_config_path = config_path or DEFAULT_RSI_BOLLINGER_V2_PROFILES_PATH
    if not effective_config_path.exists():
        return profiles

    with effective_config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"RSI Bollinger V2 profiles config must be a JSON object: {effective_config_path}")

    profiles.update(_normalize_profile_payload(payload))
    return profiles


def resolve_rsi_bollinger_v2_profile(ticker: str) -> tuple[str, dict[str, float | int]]:
    profiles = load_rsi_bollinger_v2_profiles()
    default_profile = dict(profiles.get("default", BUILTIN_RSI_BOLLINGER_V2_PROFILES["default"]))
    ticker_key = str(ticker).strip().upper()

    if ticker_key and ticker_key in profiles:
        resolved_profile = dict(default_profile)
        resolved_profile.update(profiles[ticker_key])
        return ticker_key, resolved_profile

    return "default", default_profile


def get_rsi_bollinger_v2_profile(ticker: str) -> dict[str, float | int]:
    _, profile = resolve_rsi_bollinger_v2_profile(ticker)
    return profile
