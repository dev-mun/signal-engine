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

## If Scan Shows A Flagged Setup

Run deep dives only for:

- `BUY`
- `ACTIONABLE`
- `NEAR_SETUP`

### EMA-RSI Deep Dive

```bash
python main.py --ticker TICKER --strategy ema-rsi --save-reports --no-plot
```

### 4H Trend Deep Dive

```bash
python main.py --ticker TICKER --strategy four-hour-trend --interval 1h --save-reports --no-plot
```

### RSI-Bollinger V2 Deep Dive

```bash
python main.py --ticker TICKER --strategy rsi-bollinger-v2 --save-reports --no-plot
```

Example:

```bash
python main.py --ticker NVDA --strategy rsi-bollinger-v2 --save-reports --no-plot
```

## Daily Workflow

This process is run once per day after market close. Do not scan intraday unless reviewing an already planned trade.

- Run all 3 scans after market close
- Review only `BUY` / `ACTIONABLE` / `NEAR_SETUP` names
- Run deep dives only on flagged names
- Write next-day trade plan (entry, stop, target, size)
- Execute only pre-planned trades next session
- Journal all completed trades
