# signal-engine

`signal-engine` is a deterministic trading research and daily workflow system for:

- equity signal scanning
- single-ticker deep dives
- options overlay planning
- paper-journal logging
- proxy validation and cadence monitoring

It is built around explainable rules. It does not use live broker execution, machine learning, or live option-chain integrations.

---

## Current System

The project currently runs four active strategy paths:

- `ema-rsi`
- `four-hour-trend`
- `rsi-bollinger-v2`
- `swing-options-debit-spread`

The options layer is an overlay on top of stock signal quality. It is not a standalone prediction engine.

---

## Primary Workflow

The preferred daily command is:

```bash
python scripts/run_daily_workflow.py
```

This runs:

- `ema-rsi`
- `four-hour-trend`
- `rsi-bollinger-v2`
- `swing-options-debit-spread` on `small_account_debit_spreads`
- `swing-options-debit-spread` on `small_account_growth`

It also:

- saves scan CSVs
- updates the paper trading journal
- writes:
  - `reports/daily/YYYY-MM-DD/daily_summary.md`
  - `reports/daily/YYYY-MM-DD/daily_summary.json`

Useful variants:

```bash
python scripts/run_daily_workflow.py --auto-open-summary
python scripts/run_daily_workflow.py --summary-only --date YYYY-MM-DD
python scripts/run_daily_workflow.py --strategies swing-options-debit-spread
python scripts/run_daily_workflow.py --skip-strategy-on-error
```

---

## Strategy Summary

### `ema-rsi`

Daily swing scan for slower pullback setups across the broad-market watchlist.

### `four-hour-trend`

1H data path with 4H structure logic for medium-frequency trend continuation and tactical short setups.

### `rsi-bollinger-v2`

Ticker-profile-driven mean reversion strategy with per-ticker deployment parameters and a focused high-beta universe.

### `swing-options-debit-spread`

Small-account bullish options overlay using bull call debit spreads.

This path reuses the `swing-options` signal engine and adds:

- affordability filters
- spread-structure planning
- reward/risk checks
- small-account eligibility labeling
- proxy validation reports

Live deployment default is the tuned debit-spread path.

---

## Watchlist Profiles

The system uses named watchlists instead of manual ticker entry.

Examples:

- `broad_market`
- `high_beta`
- `small_account_debit_spreads`
- `small_account_growth`

Current small-account debit-spread roles:

- `small_account_debit_spreads`
  - large-cap benchmark/context
  - `SPY`, `QQQ`, `AAPL`, `AMD`

- `small_account_growth`
  - primary small-account deployment universe
  - lower-priced liquid growth names

You can still override with explicit tickers:

```bash
python main.py --scan NVDA,AVGO,META --strategy rsi-bollinger-v2
python main.py --scan --strategy swing-options-debit-spread --profile small_account_growth --no-plot
```

---

## Phase 1 Options Quality Layer

The current options workflow includes a deterministic Phase 1 quality layer focused on:

- market awareness
- false-positive reduction
- stronger no-trade behavior
- better reporting clarity

### Market Regime

Daily regime classification uses:

- `SPY`
- `QQQ`
- `VIX`

Regime states:

- `BULLISH`
- `BEARISH`
- `MIXED`

### No-Trade Filters

The options overlay can force `NO_TRADE` when conditions are poor, including:

- earnings too close
- weak ATR
- weak volume / liquidity
- excessive extension from `EMA20`
- regime conflict
- expected-move exhaustion

### Multi-Timeframe Confirmation

Current confirmation layer uses only:

- Daily
- 4H

### Scoring and Ratings

Options-facing outputs now distinguish:

- `BaseScore`
  - raw underlying score before regime/no-trade overlay
- `SetupScore`
  - regime-aware quality score after assessment
- `FinalScore`
  - current user-facing score used in reports and summaries

Current rating buckets:

- `A_SETUP`
- `B_SETUP`
- `WATCHLIST`
- `NO_TRADE`

### Action State

Operational summaries also normalize results into:

- `ACTIONABLE`
- `WATCHLIST`
- `NO_TRADE`
- `IGNORE`
- `ERROR`

This is a reporting layer only. It does not replace the underlying `Signal` or `Setup` values.

---

## Daily Summary Output

The generated markdown summary is designed for end-of-day review.

Current sections include:

- Executive Decision
- Top Setup
- Market Regime
- Breadth Snapshot
- Market State
- Actionable Signals
- Large-Cap Debit Spread Context
- Small-Account Growth Debit Spread Candidates
- Manual Live Chain Confirmation Required
- Key No-Trade Reasons
- Watchlist Names
- Ignore List
- Workflow Failures
- Paper Execution Checklist
- Tomorrow Plan
- Risk Notes

The summary is deterministic and rule-based. No LLM or API is used in the workflow generator.

---

## Manual Live Chain Confirmation

Planner output is not an executable order.

Before any paper or live trade, manually confirm the spread in Fidelity:

- same ticker
- same expiration range, 30-45 DTE preferred
- same long/short strikes or closest liquid equivalent
- actual debit
- actual max loss
- actual max profit
- bid/ask spread
- volume / open interest
- reward/risk >= 1.5
- total debit within account cap

If live pricing differs materially from the planner estimate:

- use real broker pricing
- skip the trade if the structure no longer fits
- or record it as `planner mismatch`

---

## Deep Dive Commands

Single-ticker analysis:

```bash
python main.py --ticker TICKER --strategy ema-rsi --save-reports --no-plot
python main.py --ticker TICKER --strategy four-hour-trend --interval 1h --save-reports --no-plot
python main.py --ticker TICKER --strategy rsi-bollinger-v2 --save-reports --no-plot
python main.py --ticker TICKER --strategy swing-options-debit-spread --save-reports --no-plot
```

Example:

```bash
python main.py --ticker AMD --strategy swing-options-debit-spread --save-reports --no-plot
```

---

## Validation and Backtests

Examples:

```bash
python scripts/backtest_swing_options_debit_spread.py --mode tuned
python scripts/backtest_small_account_growth.py
python scripts/backtest_small_account_scan.py
python scripts/backtest_swing_options_proxy.py
```

These reports are for proxy validation and cadence monitoring. They are not real option-chain PnL backtests.

---

## Reports

Common report locations:

```text
reports/daily/YYYY-MM-DD/daily_summary.md
reports/daily/YYYY-MM-DD/daily_summary.json
reports/paper_trading_journal.xlsx
reports/ema_rsi/watchlist_scan_YYYY-MM-DD.csv
reports/four_hour/watchlist_scan_YYYY-MM-DD.csv
reports/rsi_bollinger_v2/watchlist_scan_YYYY-MM-DD.csv
reports/swing_options_debit_spread/scan_small_account_debit_spreads_YYYY-MM-DD.csv
reports/swing_options_debit_spread/scan_small_account_growth_YYYY-MM-DD.csv
reports/swing_options_debit_spread/*_options_plan.csv
reports/swing_options_debit_spread/*_latest_signal.csv
```

---

## Installation

Supported Python: `3.9+`

Use one interpreter consistently for installs, tests, and runs.

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

## Testing

```bash
pytest
```

Or:

```bash
python -m pytest
```

---

## Limitations

This is a research, planning, and paper-workflow system.

Current constraints:

- `PROXY VALIDATION ONLY` for options backtest layers
- no live option chain
- no real Greeks
- no IV surface or skew model
- no broker routing
- no auto execution
- no machine learning

Use it as:

- a signal engine
- a filtering engine
- a workflow/reporting layer
- an options planning tool

Do not treat planner output as a broker-ready order ticket.
