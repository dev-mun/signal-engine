# Commands

## Daily Commands

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
python main.py --scan --strategy swing-options --profile small_account_options --no-plot --journal
```

Purpose:

- focused small-account options universe
- labels affordable long-call candidates for a $2k to $3k account
- daily journal update

# Swing Options Daily Workflow

## Daily Small Account Scan (After Market Close)

Run:

```bash
python main.py --scan --strategy swing-options --profile small_account_options --no-plot --journal
```

Purpose:

- daily post-close scan
- identify small-account eligible long-call setups
- only review affordable trades for a $2k to $3k account

Review only rows where:

- `Signal = BUY`
- `SmallAcct = YES`
- `PremiumStatus = OK` or `ACCEPTABLE`

Ignore:

- `HOLD`
- `EXTENDED`
- `AVOID`
- `TOO_EXPENSIVE`

## If A Valid Candidate Appears (Same Night)

Run ticker deep dive:

```bash
python main.py --ticker TICKER --strategy swing-options --save-reports --no-plot
```

Example:

```bash
python main.py --ticker AMD --strategy swing-options --save-reports --no-plot
```

Purpose:

- inspect full setup
- inspect score
- inspect source alignment
- inspect estimated option structure
- confirm candidate before next session

## Next Morning Execution Checklist

Only if prior night produced valid candidate.

Before market open:

1. Open Fidelity options chain
2. Find 30–60 DTE call
3. Prefer ~45 DTE
4. Target delta 0.50–0.65
5. Debit <= $150
6. Tight bid/ask spread
7. Skip illiquid chain
8. Skip if contract no longer fits setup

Execution rules:

- one position at a time
- one contract only
- no averaging down
- no revenge trades
- skip if setup degrades at open

## Weekly Review (Weekend)

Run:

```bash
python scripts/backtest_swing_options_proxy.py
```

Purpose:

- validate signal cadence
- review move-quality drift
- ensure BUY cadence remains healthy
- confirm system behavior has not degraded

Review:

- trades/month
- % reaching 1R within 5D
- % reaching 2R within 10D
- move-quality distribution
- premium-over-budget frequency

Do not modify strategy from weekly review alone.
Use weekly review only for monitoring unless paper results confirm execution issues.

This is now the primary workflow.
