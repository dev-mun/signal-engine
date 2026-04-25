# signal-engine

A Python-based trend-following research engine for scanning equities, backtesting pullback entries, and generating executable, risk-aware options trade ideas.

This project is built as a practical trading workflow:

- Scan a watchlist
- Identify actionable setups
- Backtest the signal logic
- Size risk using a defined stop
- Filter option quality using IV environment
- Generate executable defined-risk options structures

---

# Strategy Overview

This system is a **trend-pullback strategy** with defined-risk sizing and options execution logic.

It is designed to:

- avoid chasing extended momentum
- enter strong uptrends on pullbacks
- separate bullish entries, bearish entries, and long exits
- size positions from actual stop distance at fill
- exit using controlled, asymmetric risk rules
- reject poor options environments
- convert valid stock signals into executable options trades

---

# Core Strategy Logic

## Signal Model

The engine now uses three distinct strategy states:

- `BUY`: bullish entry setup
- `EXIT_LONG`: exit an existing long position
- `BEARISH_ENTRY`: separate bearish setup

This avoids treating a long exit as a new bearish trade.

---

## Entry Rules (Buy Signal)

A long signal is generated only when all conditions are true:

- Close > EMA200
- EMA50 > EMA200
- RSI between 40 and 60
- Close crosses above EMA20
- Volume > 20-day average volume

This filters for:

- long-term uptrend
- short-term pullback
- renewed momentum
- above-average participation

---

## Entry Rules (Bearish Entry Signal)

A bearish entry signal is generated only when all conditions are true:

- Close < EMA200
- EMA50 < EMA200
- RSI between 40 and 60
- Close crosses below EMA20
- Volume > 20-day average volume

This mirrors the long pullback logic for bearish setups.

---

## Exit Rules (`EXIT_LONG`)

Positions are closed when one of the following triggers:

- Stop loss: 8%
- Take profit: 20%
- Trailing stop: 8% below highest close since entry
- Max hold: 60 trading days

This creates:

- controlled downside
- asymmetric upside
- winner protection
- forced capital rotation

`EXIT_LONG` means close an existing long position. It is not a bearish entry.

---

## Execution Timing

Signals are generated from the current bar's close, but trades are filled on the next trading day's open.

This reduces look-ahead bias and makes fills more realistic than same-bar close execution.

---

## Position Sizing (Risk Engine)

Position size is based on the actual stop distance at the fill price.

Instead of buying full notional exposure:

- Risk only 1.5% of equity per trade
- Stop is defined as a fixed percentage below the filled entry price
- Position size is based on `entry price - stop price`
- Gap risk can still cause realized loss to differ from planned loss

ATR remains available as a diagnostic metric, but it is no longer the direct position-sizing input.

---

# Options Overlay

The stock engine determines **direction**.

The options engine determines **execution**.

## Options Engine Capabilities

The options engine now performs:

- IV Rank proxy filtering
- options chain selection
- expiration targeting (30–45 DTE)
- strike selection
- spread pricing
- reward/risk calculation
- max loss / max profit calculation
- contract sizing
- trade quality filtering

This converts a signal into an executable options trade.

---

## When signal = BUY

System evaluates a:

- Call Debit Spread
- 30–45 DTE
- ATM / slightly ITM long call
- OTM short call

Then calculates:

- exact expiration
- exact strikes
- estimated debit
- max loss
- max profit
- breakeven
- reward/risk
- IV status
- estimated contracts

---

## When signal = BEARISH_ENTRY

System evaluates a:

- Put Debit Spread
- 30–45 DTE
- ATM / slightly ITM long put
- OTM short put

Then calculates the same risk profile.

---

## When signal = EXIT_LONG

No new bearish options trade is opened.

This signal means close the existing long position on the next open.

---

## When signal = HOLD / HOLD_POSITION

No options trade is printed.

This prevents forcing trades when no setup exists.

---

# Watchlist Scanner

The scanner now runs in two stages:

1. Universe filter
2. Signal scan on eligible names only

Before a ticker is scanned for signals, it must qualify as tradable.

The current eligibility filter checks:

- 20-day average stock dollar volume
- listed options in the configured DTE window
- ATM call/put open interest
- ATM call/put daily option volume
- ATM call/put bid/ask spread quality
- earnings buffer

The scanner then prints:

- Universe status
- Signal
- Setup status
- Price
- RSI
- ATR
- Distance to setup

Example:

```bash
python main.py --scan SPY,QQQ,MSFT,META,NVDA,AAPL --no-plot
```

Example output:

```text
Ticker   Universe       Signal          Setup         Price     RSI    ATR  Distance
AAPL     ELIGIBLE       HOLD            NEAR_SETUP   271.06   62.01   6.21  Needs mild pullback (-2.0 RSI)
XYZ      FILTERED_OUT   INELIGIBLE      FILTERED_OUT  12.40    0.00   0.00  Rejected by universe filter
```

This lets you identify:

- eligible names worth scanning
- filtered-out names not worth trading
- actionable setups
- exit signals
- near setups
- extended names to avoid

---

# Setup Status Definitions

## ACTIONABLE
Signal is live now.

## EXIT
Existing long should be exited on the next open.

## NEAR_SETUP
Close to trigger. Usually needs a mild RSI pullback.

## EXTENDED
Too hot. Avoid chasing.

## WAIT
No signal, not close.

## WEAK
Too weak. Trend not ready.

## FILTERED_OUT
Rejected before signal evaluation because the ticker failed the universe filter.

---

# Fresh Market Data

When using `--ticker` or `--scan`, data is pulled fresh from Yahoo Finance via `yfinance`.

Example:

```bash
python main.py --ticker SPY
```

CSV mode is only used when explicitly passed.

---

# Installation

Supported Python: `3.9+`

Use a virtual environment and keep the same interpreter active for installs, tests, and runs.

## macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Windows

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

# Usage

## Daily Workflow

Run scanner first:

```bash
python main.py --scan SPY,QQQ,MSFT,META,NVDA,AAPL --no-plot
```

This prints:

- eligible names
- filtered-out names
- actionable trades
- exit signals
- near setups
- extended names
- daily watchlist summary

It also saves:

```text
reports/watchlist_scan_YYYY-MM-DD.csv
```

---

## Run Single Ticker Deep Dive

Use when scanner shows a setup:

```bash
python main.py --ticker AAPL --save-reports --no-plot
```

This prints:

- full backtest summary
- latest signal
- executable options trade (only if valid)
- recent trades

And saves:

```text
reports/AAPL_equity.csv
reports/AAPL_trades.csv
reports/AAPL_signals.csv
reports/AAPL_latest_signal.csv
```

---

## Run Single Ticker with Chart

```bash
python main.py --ticker SPY
```

---

## Run Demo Mode

```bash
python main.py --demo
```

---

## Run CSV Mode

CSV must contain:

```text
Date,Open,High,Low,Close,Volume
```

Run:

```bash
python main.py --csv ./data/my_stock_data.csv
```

---

# Reports Generated

Single ticker reports:

```text
reports/SPY_equity.csv
reports/SPY_trades.csv
reports/SPY_signals.csv
reports/SPY_latest_signal.csv
```

Daily scanner reports:

```text
reports/watchlist_scan_YYYY-MM-DD.csv
```

---

# Example Workflow

## Daily

Run scanner:

```bash
python main.py --scan SPY,QQQ,MSFT,META,NVDA,AAPL --no-plot
```

If no actionable setup:
- do nothing

If filtered out:
- skip the ticker entirely

If near setup:
- monitor next session

If `BUY` or `BEARISH_ENTRY`:
- run single ticker deep dive

If `EXIT_LONG`:
- plan to close the long on the next open

```bash
python main.py --ticker MSFT --save-reports --no-plot
```

Then evaluate the executable options structure.

---

# Universe Filter Controls

Scanner defaults:

- `--min-avg-dollar-volume 20000000`
- `--min-atm-open-interest 50`
- `--min-atm-option-volume 5`
- `--max-atm-bid-ask-spread-pct 20`
- `--option-min-dte 30`
- `--option-max-dte 45`
- `--earnings-buffer-days 3`

Example:

```bash
python main.py --scan SPY,QQQ,IWM,DIA,AMZN,GOOGL --min-avg-dollar-volume 30000000 --earnings-buffer-days 7 --no-plot
```

If you want to inspect a ticker without enforcing the universe screen:

```bash
python main.py --scan SPY,QQQ,XYZ --no-universe-filter --no-plot
```

---

# Testing

```bash
pytest
```

Or:

```bash
python -m pytest
```

---

# Risk Notes

This is a **research and trade selection engine**, not a production execution system.

It does not:

- place trades
- model slippage
- model intraday fill quality beyond next-open execution
- model live IV Rank
- route orders to broker

Use this as:

- signal engine
- risk engine
- options trade selection engine

Not as a broker execution engine.

---

# Universe Seed List

Useful starting universe candidates:

- SPY
- QQQ
- IWM
- DIA
- MSFT
- AMZN
- GOOGL
- META
- NVDA
- AAPL

These are not assumed tradable by default. They still must pass the universe filter on the day they are scanned.

---

# Current Best Workflow

1. Provide a broad candidate universe
2. Let the universe filter reject weak instruments
3. Scan eligible names for `BUY`, `BEARISH_ENTRY`, or `EXIT_LONG`
4. Run ticker deep dive on the best eligible setups
5. Review executable options trade only for `BUY` or `BEARISH_ENTRY`
