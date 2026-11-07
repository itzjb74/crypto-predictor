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

**Data:** about 6 months of hourly candles (May 8 to Nov 4, 2026, 4,319 rows).
**Target:** does the next hour close up? About 50% of hours do.

**Accuracy** (50% is a coin flip; each fold's margin of error is about ±4 points):

| Approach | Single 80/20 split | Walk-forward, 6 folds (average) |
|---|---|---|
| Always predict the majority class | 50.7% | 49.8% |
| Opposite of last hour's direction | 51.4% | 52.7% |
| Logistic Regression | 51.0% | 51.6% |
| XGBoost | 51.0% | 51.5% |

The one-line "opposite of last hour" rule (a faint mean-reversion effect) beat 50% in
all six periods, and neither model improved on it. XGBoost scored **67%** on the data
it was trained on but about 51.5% on unseen data, which is a textbook case of
memorizing noise.

**Trading after fees** (assumed 0.1% per trade, walk-forward over Jun 7 to Nov 4):

| Strategy | Total return |
|---|---|
| Cash (do nothing) | 0.0% |
| Buy and hold | +29.7% (Bitcoin rose over the period) |
| Logistic Regression, trade only if P(up) > 0.60 | -5.2% |
| XGBoost, trade only if P(up) > 0.60 | -35.1% |
| Logistic Regression, trade every hour | -46.6% |
| Opposite of last hour, trade every hour | -83.7% |

Every model-driven strategy lost to doing nothing. A hit rate of 52% sounds like an
edge, but a typical hourly move is about 0.3% and each trade costs 0.1%, so the edge
is several times smaller than the cost of acting on it.

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
dashboard.py        Streamlit dashboard
01_get_data.ipynb ... 03_baselines.ipynb   exploration, baselines, walk-forward tests
```

## Running it

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 1. Download history: run the notebook 01_get_data.ipynb (writes data/btc_1h.csv)
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