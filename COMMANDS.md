# Commands

## Daily Commands

### 1. Daily EMA-RSI Scan

Run after market close:

```bash
python main.py --scan SPY,DIA,CRM,PLTR,SHOP,UBER --strategy ema-rsi --no-plot --journal
```

Purpose:

- slower swing scan
- updates daily journal
- only checks high-quality daily setups

### 2. Daily 4H Scan

Run after market close:

```bash
python main.py --scan SPY,QQQ,IWM,DIA,AAPL,MSFT,NVDA,AMZN,CRM,PLTR,SHOP,UBER,META,GOOGL,AVGO,TSM,PANW,CRWD --strategy four-hour-trend --interval 1h --no-plot --journal
```

Purpose:

- primary higher-frequency scan
- updates daily journal
- main monthly opportunity engine

## If Scan Shows A Flagged Setup

### 3. EMA-RSI Deep Dive

Run only for flagged names (`BUY` or `NEAR_SETUP`):

```bash
python main.py --ticker TICKER --strategy ema-rsi --save-reports --no-plot
```

Example:

```bash
python main.py --ticker CRM --strategy ema-rsi --save-reports --no-plot
```

### 4. 4H Deep Dive

Run only for flagged names (`BUY`):

```bash
python main.py --ticker TICKER --strategy four-hour-trend --interval 1h --save-reports --no-plot
```

Example:

```bash
python main.py --ticker PLTR --strategy four-hour-trend --interval 1h --save-reports --no-plot
```
