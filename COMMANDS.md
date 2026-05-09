# Commands

## Primary Daily Workflow

### 1. Run Full Daily Workflow

Run after market close:

```bash
python scripts/run_daily_workflow.py
```

What it does:

- runs `ema-rsi`
- runs `four-hour-trend`
- runs `rsi-bollinger-v2`
- runs `swing-options-debit-spread` for `small_account_debit_spreads`
- runs `swing-options-debit-spread` for `small_account_growth`
- saves each scan report
- updates the paper journal
- generates:
  - `reports/daily/YYYY-MM-DD/daily_summary.md`
  - `reports/daily/YYYY-MM-DD/daily_summary.json`

The daily summary now includes:

- market regime
- top setup
- final score
- action state
- key no-trade reasons

Optional:

```bash
python scripts/run_daily_workflow.py --auto-open-summary
python scripts/run_daily_workflow.py --summary-only --date YYYY-MM-DD
python scripts/run_daily_workflow.py --strategies swing-options-debit-spread
python scripts/run_daily_workflow.py --skip-strategy-on-error
```

---

## Strategy Scan Commands

### 1. Daily EMA-RSI Scan

Run after market close:

```bash
python main.py --scan --strategy ema-rsi --no-plot --journal
```

Purpose:

- slower swing scan
- broad market profile
- daily journal update
- higher-conviction trend setups

---

### 2. Daily 4H Trend Scan

Run after market close:

```bash
python main.py --scan --strategy four-hour-trend --interval 1h --no-plot --journal
```

Purpose:

- medium-frequency trend scan
- broad market profile
- daily journal update
- primary trend continuation engine
---
 
### 3. Daily RSI-Bollinger V2 Scan

Run after market close:

```bash
python main.py --scan --strategy rsi-bollinger-v2 --no-plot --journal
```

Purpose:

- tactical mean reversion scan
- high-beta profile
- daily journal update
- higher-frequency tactical setup engine

---

### 4. Daily Swing Options Small-Account Scan

Run after market close:

```bash
python main.py --scan --strategy swing-options-debit-spread --no-plot --journal
```

Purpose:

- large-cap benchmark/context profile
- runs the default `small_account_debit_spreads` universe
- daily journal update

### 5. Daily Swing Options Growth Universe Scan

Run after market close:

```bash
python main.py --scan --strategy swing-options-debit-spread --profile small_account_growth --no-plot --journal
```

Purpose:

- primary small-account deployment universe
- lower-priced liquid names more suitable for small-account debit spread planning
- daily journal update

# Swing Options Daily Workflow

## Daily Workflow Runner (Preferred)

Run:

```bash
python scripts/run_daily_workflow.py
```

Operational use:

- this is the preferred daily command
- runs all stock scans plus both debit spread profiles
- writes the daily markdown/json summary
- prints top decision, market regime, and top setup

## Daily Small Account Scan (After Market Close)

Run:

```bash
python main.py --scan --strategy swing-options-debit-spread --profile small_account_growth --no-plot --journal
```

Purpose:

- daily post-close scan
- identify small-account eligible bull call debit spread setups in the primary growth universe
- use tuned mode as the live production default
- only review affordable trades for a $2k to $3k account
- planner output is a candidate generator, not an executable order

Context scan:

```bash
python main.py --scan --strategy swing-options-debit-spread --profile small_account_debit_spreads --no-plot --journal
```

- use this for large-cap benchmark/context
- do not treat it as the primary deployment universe

Review only rows where:

- `Signal = BUY`
- `ActionState = ACTIONABLE`
- `SmallAccountEligible = YES`
- `PremiumStatus = OK` or `ACCEPTABLE`

Ignore:

- `HOLD`
- `NO_TRADE`
- `EXTENDED`
- `AVOID`
- `TOO_EXPENSIVE`
- `BAD_REWARD_RISK`

## If A Valid Candidate Appears (Same Night)

Run ticker deep dive:

```bash
python main.py --ticker TICKER --strategy swing-options-debit-spread --save-reports --no-plot
```

Example:

```bash
python main.py --ticker AMD --strategy swing-options-debit-spread --save-reports --no-plot
```

Purpose:

- inspect full setup
- inspect score
- inspect source alignment
- inspect estimated debit spread structure
- confirm candidate before next session
- prepare for manual live-chain confirmation, not direct execution

## Manual Live Chain Confirmation Required

Before any paper or live trade, manually confirm the spread in Fidelity:

- same ticker
- same expiration range, 30-45 DTE preferred
- same long/short strikes or closest liquid equivalent
- actual debit
- actual max loss
- actual max profit
- bid/ask spread
- volume/open interest
- reward/risk >= 1.5
- total debit within account cap

Rules:

- planner output is not an executable order
- it is only a candidate generator
- if real debit differs materially from estimated debit, use real broker pricing
- skip the trade if the real structure no longer fits the plan
- otherwise record it as `planner mismatch`

## Next Morning Execution Checklist

Only if prior night produced valid candidate.

Before market open:

1. Open Fidelity options chain
2. Find a bull call debit spread
3. Use 30-60 DTE
4. Prefer 30-45 DTE
5. Keep total debit <= $150
6. Keep reward/risk >= 1.5
7. Tight bid/ask spread
8. Skip illiquid chain
9. Skip if live spread no longer fits setup

Execution rules:

- one position at a time
- one spread only
- no averaging down
- no revenge trades
- skip if setup degrades at open

## Weekly Review (Weekend)

Run:

```bash
python scripts/backtest_swing_options_debit_spread.py --mode tuned
```

Purpose:

- validate signal cadence
- review move-quality drift
- ensure debit spread BUY cadence remains healthy
- confirm system behavior has not degraded

Review:

- trades/month
- win rate proxy
- average proxy PnL %
- profit factor proxy
- max drawdown proxy
- % failed
- % suitable+

Do not modify strategy from weekly review alone.
Use weekly review only for monitoring unless paper results confirm execution issues.

Strict mode is research-only.
Small-account long-call workflow is deprecated.
This is now the primary live small-account options workflow.
