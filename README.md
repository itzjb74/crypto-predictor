# BTC next-hour direction predictor (experimental)

> **Learning project.** Careful testing found **no reliable, tradable edge** in this
> setup once trading fees are included. Not financial advice.

A small end-to-end machine-learning project: download Bitcoin price data, build
features, train models, evaluate them honestly, then serve live predictions through
a FastAPI service and a Streamlit dashboard. The goal was to learn the full pipeline
and what rigorous evaluation looks like, not to build a money machine.

<!-- Add a screenshot: save it as docs/dashboard.png, or delete this line -->
![Dashboard](docs/dashboard.png)

## What it does

- Streams live BTC/USDT hourly candles from Binance (websocket) and keeps a rolling
  buffer of the last 200 **closed** candles.
- Computes six features from price and volume: 1-hour and 5-hour returns, 20-hour
  volatility, RSI(14), volume ratio, and distance from the 20-hour moving average.
- Predicts the probability that the next hour closes **up**, using Logistic
  Regression and XGBoost.
- Serves the latest prediction over REST (`/predict`) and a websocket
  (`/ws/predictions`), and logs every prediction so it can be scored against what
  the market actually did.

## Results (the honest part)

**Data:** about 5 years of hourly BTC/USDT candles (Nov 2021 to Nov 2026, 43,798 rows).
**Target:** does the next hour close up? About 50% of hours do.
**Method:** walk-forward validation with 8 folds. Each fold trains on all earlier data and
tests on the next 4,000 hours (about 5.5 months), covering March 2023 to November 2026.
Success criteria were written down before the test was run.

An earlier test on only six months of data was inconclusive (both models near 51.5%);
more data was the single biggest improvement.

**Accuracy** (pooled over 32,000 test predictions; coin-flip margin of error is about ±0.55 points):

| Approach | Accuracy | Folds above 50% |
|---|---|---|
| Always predict the majority class | 50.5% | 5 of 8 |
| Opposite of last hour's direction | 52.7% | 8 of 8 |
| Logistic Regression | 52.6% | 8 of 8 |
| XGBoost | 53.2% | 8 of 8 |

There is a small but real directional signal, mostly weak mean reversion after the
previous hour's move. XGBoost beat the one-line rule narrowly.

**Trading after fees** (assumed 0.1% per trade, no slippage; all 8 folds compounded):

| Strategy | Profitable folds | Total return |
|---|---|---|
| Cash (do nothing) | n/a | 0.0% |
| Buy and hold | 5 of 8 | +208.3% (Bitcoin rose over the period) |
| Logistic Regression, trade if P(up) > 0.60 | 2 of 8 | -37.4% |
| XGBoost, trade if P(up) > 0.60 | 0 of 8 | -95.1% |
| Logistic Regression, trade if P(up) > 0.50 | 0 of 8 | -100.0% |
| XGBoost, trade if P(up) > 0.50 | 0 of 8 | -100.0% |

By the success criteria written in advance, the models have **no tradable edge**: the
accuracy criteria passed, but no configuration beat cash after fees.

**Why:** the signal is real but small. XGBoost at the 0.50 threshold would break even at a fee of
roughly 0.01% per trade, against the 0.1% assumed. The edge per trade is concentrated in
high-volatility hours (break-even fees of about 0.03% to 0.05% there), which is closer
but still short of the fee, and trading costs are also higher when volatility is high.

**Live paper-trading results:** _(fill in after the server has run for a couple of
weeks: `python -m training.evaluate_live`)_

| Scored predictions | LogReg right | XGBoost right | Coin-flip margin |
|---|---|---|---|
| _n_ | _%_ | _%_ | _±_ |

## Architecture

```
Binance websocket (1h candles)
        |
        v
  Rolling buffer (200 closed candles) <-- seeded at startup via ccxt
        |
        v
  features.py --> predict.py (LogReg + XGBoost) --> prediction log (CSV)
        |
        v
  FastAPI:  /predict   /candles   /history   /ws/predictions
        |
        v
  Streamlit dashboard (price chart, latest prediction, scoreboard)
```

The same `features.py` is used for training and for live prediction, so the two
cannot drift apart. Features only look backward in time, and the live buffer only
ever contains closed candles.

## Project structure

```
app/
  features.py   feature engineering shared by training and serving
  predict.py    loads saved models and predicts from recent candles
  ingest.py     websocket listener, rolling buffer, gap repair
  log.py        prediction log and scoring
  main.py       FastAPI service
training/
  train.py          trains and saves both models
  evaluate_live.py  scores the live log, with a paper-traded return after fees
  fetch_history.py  downloads several years of hourly candles
dashboard.py        Streamlit dashboard
01_get_data.ipynb ... 03_baselines.ipynb   exploration, baselines, walk-forward tests
```

## Running it

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 1. Download history: run the notebook 01_get_data.ipynb (writes data/btc_1h.csv),
#    or python -m training.fetch_history --years 5 (writes data/btc_1h_long.csv)
# 2. Train and save the models
python -m training.train

# 3. Start the API, then the dashboard (in a second terminal)
uvicorn app.main:app
streamlit run dashboard.py

# 4. After it has run for a while, score the live predictions
python -m training.evaluate_live
```

## Limitations

- One asset, one timeframe, about six months of data; results may not generalize.
- Only price and volume features. News, order flow, and derivatives data are not used.
- Fees are assumed at 0.1% per trade, with no slippage. Real costs would be higher.
- The paper-trading log covers a short period, so its accuracy has a wide margin of
  error until it contains hundreds of predictions.

## What I learned

- Split time-series data by time, never randomly, and judge models with walk-forward
  validation instead of one lucky split.
- Accuracy on training data says almost nothing; compare against simple baselines.
- Trading costs can erase a small statistical edge entirely.
- Sharing one feature function between training and serving prevents a whole class
  of live-system bugs.
- Async Python, websockets, and the FastAPI lifespan pattern (including the
  "coroutine was never awaited" bug I hit and fixed along the way).