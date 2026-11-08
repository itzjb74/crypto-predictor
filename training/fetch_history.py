"""Download several years of hourly BTC/USDT candles from Binance and save them to CSV.

Run from the project folder:  python -m training.fetch_history --years 5
"""
import argparse
import time
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAIR = "BTC/USDT"
TIMEFRAME = "1h"
COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def fetch_batch(exchange, since, retries=3):
    """One request for up to 1000 candles, retried a few times if the network hiccups."""
    for _ in range(retries):
        try:
            return exchange.fetch_ohlcv(PAIR, TIMEFRAME, since=since, limit=1000)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"  request failed ({e.__class__.__name__}), retrying in 5 seconds...")
            time.sleep(5)
    raise RuntimeError("Binance requests kept failing. Try again in a few minutes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=5, help="how many years back to download")
    parser.add_argument("--out", default="data/btc_1h_long.csv", help="output file, relative to the project folder")
    args = parser.parse_args()

    exchange = ccxt.binance()
    hour_ms = 60 * 60 * 1000
    now = exchange.milliseconds()
    since = now - int(args.years * 365 * 24) * hour_ms

    rows = []
    print(f"Downloading about {args.years} years of {PAIR} {TIMEFRAME} candles...")
    while since < now:
        batch = fetch_batch(exchange, since)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1   # continue just after the last candle we got
        print(f"  {len(rows):>6} candles, up to {pd.to_datetime(batch[-1][0], unit='ms')}")
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df = df.iloc[:-1]   # the newest candle is still in progress, so drop it

    gaps = df["time"].diff().dropna()
    bad = gaps[gaps != pd.Timedelta(hours=1)]

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print()
    print(f"Saved {len(df)} candles to {out}")
    print(f"From {df['time'].min()} to {df['time'].max()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Gaps (steps that are not exactly 1 hour): {len(bad)}")
    if len(bad):
        print(f"Largest gap: {bad.max()} (ending {bad.idxmax() and df['time'].iloc[bad.idxmax()]})")


if __name__ == "__main__":
    main()