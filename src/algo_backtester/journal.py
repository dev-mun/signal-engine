from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

DAILY_SCAN_LOG_SHEET = "Daily Scan Log"
TRADE_JOURNAL_SHEET = "Trade Journal"

DAILY_SCAN_LOG_COLUMNS = [
    "Scan Timestamp",
    "Signal Date",
    "Planned Execution Date",
    "Ticker",
    "Universe Status",
    "Universe Reason",
    "Signal Type",
    "Setup Status",
    "Distance To Setup",
    "Price",
    "RSI",
    "ATR",
    "Equity",
    "Planned Entry Reference",
    "Stop Loss",
    "Take Profit",
    "Risk/Share",
    "Reward/Share",
    "Options Action",
    "Options Structure",
    "Options Expiration",
    "DTE",
    "Strikes",
    "Debit",
    "Max Loss",
    "Max Profit",
    "Trade Quality",
    "Reason",
    "Options Reason",
    "Avg Dollar Volume",
    "Earnings Date",
]

TRADE_JOURNAL_COLUMNS = [
    "Trade ID",
    "Ticker",
    "Signal Type",
    "Signal Date",
    "Planned Execution Date",
    "Asset Type",
    "Direction",
    "Setup",
    "Planned Entry Reference",
    "Stop Loss",
    "Take Profit",
    "Risk/Share",
    "Reward/Share",
    "Options Structure",
    "DTE",
    "Strikes",
    "Debit",
    "Max Loss",
    "Max Profit",
    "System Notes",
    "Actual Entry",
    "Actual Exit",
    "Exit Date",
    "Realized PnL",
    "Max Favorable Excursion",
    "Max Adverse Excursion",
    "Followed Rules",
    "Trader Notes",
]


def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Excel journal support requires openpyxl. Install project requirements and rerun."
        ) from exc


def _safe_value(value):
    if value in {None, "", "N/A"}:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return value


def _next_business_day(date_str: str) -> str:
    if not date_str:
        return ""

    return str((pd.Timestamp(date_str) + BDay(1)).date())


def _trade_id(ticker: str, signal_type: str, signal_date: str) -> str:
    return f"{signal_date}-{ticker.upper()}-{signal_type}"


def _direction(signal_type: str) -> str:
    if signal_type == "BUY":
        return "LONG"
    if signal_type == "BEARISH_ENTRY":
        return "BEARISH"
    if signal_type == "EXIT_LONG":
        return "EXIT_LONG"
    return signal_type


def _setup_name(signal_type: str) -> str:
    if signal_type == "BUY":
        return "Bullish Pullback"
    if signal_type == "BEARISH_ENTRY":
        return "Bearish Pullback"
    if signal_type == "EXIT_LONG":
        return "Exit Long"
    return signal_type


def _asset_type(result: dict) -> str:
    signal_type = str(result.get("Signal", ""))
    has_options_structure = str(result.get("Structure", "")) not in {"", "No trade"}

    if signal_type == "EXIT_LONG":
        return "Position Exit"
    if has_options_structure:
        return "Options"
    return "Equity"


def _format_strikes(result: dict) -> str:
    long_strike = float(result.get("LongStrike", 0.0) or 0.0)
    short_strike = float(result.get("ShortStrike", 0.0) or 0.0)

    if long_strike <= 0 or short_strike <= 0:
        return ""

    return f"{long_strike:.2f}/{short_strike:.2f}"


def _system_notes(result: dict) -> str:
    notes = [str(result.get("Reason", "")).strip()]

    options_reason = str(result.get("OptionsReason", "")).strip()
    if options_reason and options_reason not in notes:
        notes.append(options_reason)

    return " | ".join([note for note in notes if note])


def _build_daily_scan_rows(results: list[dict]) -> pd.DataFrame:
    rows = []

    for result in results:
        rows.append(
            {
                "Scan Timestamp": str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")),
                "Signal Date": _safe_value(result.get("SignalDate", "")),
                "Planned Execution Date": _safe_value(
                    result.get("PlannedExecutionDate", _next_business_day(str(result.get("SignalDate", ""))))
                ),
                "Ticker": _safe_value(result.get("Ticker", "")),
                "Universe Status": _safe_value(result.get("UniverseStatus", "")),
                "Universe Reason": _safe_value(result.get("UniverseReason", "")),
                "Signal Type": _safe_value(result.get("Signal", "")),
                "Setup Status": _safe_value(result.get("SetupStatus", "")),
                "Distance To Setup": _safe_value(result.get("DistanceToSetup", "")),
                "Price": _safe_value(result.get("Price", "")),
                "RSI": _safe_value(result.get("RSI", "")),
                "ATR": _safe_value(result.get("ATR", "")),
                "Equity": _safe_value(result.get("Equity", "")),
                "Planned Entry Reference": _safe_value(result.get("PlannedEntryReference", "")),
                "Stop Loss": _safe_value(result.get("StopLoss", "")),
                "Take Profit": _safe_value(result.get("TakeProfit", "")),
                "Risk/Share": _safe_value(result.get("RiskPerShare", "")),
                "Reward/Share": _safe_value(result.get("RewardPerShare", "")),
                "Options Action": _safe_value(result.get("OptionsAction", "")),
                "Options Structure": _safe_value(result.get("Structure", "")),
                "Options Expiration": _safe_value(result.get("Expiration", "")),
                "DTE": _safe_value(result.get("DTE", "")),
                "Strikes": _format_strikes(result),
                "Debit": _safe_value(result.get("EstimatedDebit", "")),
                "Max Loss": _safe_value(result.get("MaxLoss", "")),
                "Max Profit": _safe_value(result.get("MaxProfit", "")),
                "Trade Quality": _safe_value(result.get("TradeQuality", "")),
                "Reason": _safe_value(result.get("Reason", "")),
                "Options Reason": _safe_value(result.get("OptionsReason", "")),
                "Avg Dollar Volume": _safe_value(result.get("AvgDollarVolume", "")),
                "Earnings Date": _safe_value(result.get("EarningsDate", "")),
            }
        )

    return pd.DataFrame(rows, columns=DAILY_SCAN_LOG_COLUMNS)


def _build_trade_journal_rows(results: list[dict]) -> pd.DataFrame:
    rows = []

    for result in results:
        signal_type = str(result.get("Signal", ""))

        if signal_type not in {"BUY", "BEARISH_ENTRY", "EXIT_LONG"}:
            continue

        signal_date = str(result.get("SignalDate", ""))
        planned_execution_date = str(result.get("PlannedExecutionDate", _next_business_day(signal_date)))

        rows.append(
            {
                "Trade ID": _trade_id(str(result.get("Ticker", "")), signal_type, signal_date),
                "Ticker": _safe_value(result.get("Ticker", "")),
                "Signal Type": signal_type,
                "Signal Date": signal_date,
                "Planned Execution Date": planned_execution_date,
                "Asset Type": _asset_type(result),
                "Direction": _direction(signal_type),
                "Setup": _setup_name(signal_type),
                "Planned Entry Reference": _safe_value(result.get("PlannedEntryReference", "")),
                "Stop Loss": _safe_value(result.get("StopLoss", "")),
                "Take Profit": _safe_value(result.get("TakeProfit", "")),
                "Risk/Share": _safe_value(result.get("RiskPerShare", "")),
                "Reward/Share": _safe_value(result.get("RewardPerShare", "")),
                "Options Structure": "" if str(result.get("Structure", "")) == "No trade" else _safe_value(result.get("Structure", "")),
                "DTE": _safe_value(result.get("DTE", "")),
                "Strikes": _format_strikes(result),
                "Debit": _safe_value(result.get("EstimatedDebit", "")),
                "Max Loss": _safe_value(result.get("MaxLoss", "")),
                "Max Profit": _safe_value(result.get("MaxProfit", "")),
                "System Notes": _system_notes(result),
                "Actual Entry": "",
                "Actual Exit": "",
                "Exit Date": "",
                "Realized PnL": "",
                "Max Favorable Excursion": "",
                "Max Adverse Excursion": "",
                "Followed Rules": "",
                "Trader Notes": "",
            }
        )

    return pd.DataFrame(rows, columns=TRADE_JOURNAL_COLUMNS)


def _load_sheet(workbook_path: Path, sheet_name: str, columns: list[str]) -> pd.DataFrame:
    if not workbook_path.exists():
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    except ValueError:
        return pd.DataFrame(columns=columns)

    return df.reindex(columns=columns)


def _append_unique(existing_df: pd.DataFrame, new_df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    if existing_df.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    if combined.empty:
        return combined.reindex(columns=existing_df.columns if not existing_df.empty else new_df.columns)

    combined = combined.drop_duplicates(subset=subset, keep="first")
    return combined


def update_paper_trading_journal(results: list[dict], output_dir: str = "reports") -> Path:
    _require_openpyxl()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    workbook_path = output_path / "paper_trading_journal.xlsx"

    daily_scan_df = _append_unique(
        existing_df=_load_sheet(workbook_path, DAILY_SCAN_LOG_SHEET, DAILY_SCAN_LOG_COLUMNS),
        new_df=_build_daily_scan_rows(results),
        subset=["Signal Date", "Ticker", "Signal Type"],
    ).reindex(columns=DAILY_SCAN_LOG_COLUMNS)

    trade_journal_df = _append_unique(
        existing_df=_load_sheet(workbook_path, TRADE_JOURNAL_SHEET, TRADE_JOURNAL_COLUMNS),
        new_df=_build_trade_journal_rows(results),
        subset=["Trade ID"],
    ).reindex(columns=TRADE_JOURNAL_COLUMNS)

    writer_mode = "a" if workbook_path.exists() else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": writer_mode}
    if workbook_path.exists():
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(workbook_path, **writer_kwargs) as writer:
        daily_scan_df.to_excel(writer, sheet_name=DAILY_SCAN_LOG_SHEET, index=False)
        trade_journal_df.to_excel(writer, sheet_name=TRADE_JOURNAL_SHEET, index=False)

    print(f"\nUpdated Paper Trading Journal: {workbook_path}")
    return workbook_path
