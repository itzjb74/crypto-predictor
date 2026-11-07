import asyncio
import json
from collections import deque

import ccxt
import pandas as pd
import websockets

from app.predict import predict_latest

SYMBOL = "btcusdt"
PAIR = "BTC/USDT"
INTERVAL = "1h"   # the model was trained on hourly candles
INTERVAL_MS = {"1m": 60_000, "1h": 3_600_000}[INTERVAL]
BUFFER_SIZE = 200
COLUMNS = ["time", "open", "high", "low", "close", "volume"]
URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_{INTERVAL}"

buffer = deque(maxlen=BUFFER_SIZE)   # closed candles, oldest first


def seed_buffer():
    """Fill the buffer with recent CLOSED candles from the REST API."""
    rows = ccxt.binance().fetch_ohlcv(PAIR, timeframe=INTERVAL, limit=BUFFER_SIZE + 1)
    rows = rows[:-1]   # the newest candle is still in progress, so drop it
    buffer.clear()
    for row in rows:
        buffer.append(dict(zip(COLUMNS, row)))


def add_closed_candle(candle):
    """Add a closed candle to the buffer. Returns True if it was new."""
    if buffer and candle["time"] <= buffer[-1]["time"]:
        return False   # duplicate, ignore
    if buffer and candle["time"] != buffer[-1]["time"] + INTERVAL_MS:
        print("Gap detected, refilling from the REST API...")
        seed_buffer()
    if not buffer or candle["time"] > buffer[-1]["time"]:
        buffer.append(candle)
    return True


def show_prediction():
    df = pd.DataFrame(list(buffer))
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    lr = predict_latest(df, "logreg")
    xg = predict_latest(df, "xgb")
    print(f"candle {lr['time']} | close {lr['price']:.2f} | "
          f"P(up next) logreg {lr['p_up']:.3f} | xgb {xg['p_up']:.3f}")


async def main():
    seed_buffer()
    print(f"Buffer ready: {len(buffer)} closed candles. Waiting for the next close...")
    show_prediction()

    async for ws in websockets.connect(URL):   # reconnects automatically
        try:
            async for message in ws:
                k = json.loads(message)["k"]
                if not k["x"]:
                    continue   # candle still in progress, ignore it
                candle = {
                    "time": k["t"], "open": float(k["o"]), "high": float(k["h"]),
                    "low": float(k["l"]), "close": float(k["c"]), "volume": float(k["v"]),
                }
                if add_closed_candle(candle):
                    show_prediction()
        except websockets.ConnectionClosed:
            print("Connection lost, reconnecting...")
            continue


if __name__ == "__main__":
    asyncio.run(main())