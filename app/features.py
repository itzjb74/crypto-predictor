import pandas as pd

FEATURE_COLS = ["return", "ret_5", "vol_20", "rsi_14", "vol_ratio", "dist_ma"]


def make_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Turn raw candles (time, open, high, low, close, volume) into model features.

    Every feature uses only the current and previous rows, so the same
    function works for training and for live data.
    """
    df = candles.copy()

    df["return"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)

    ma_20 = df["close"].rolling(20).mean()
    df["vol_20"] = df["return"].rolling(20).std()
    df["dist_ma"] = df["close"] / ma_20 - 1

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + rs)

    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df[["time", "close"] + FEATURE_COLS].dropna().reset_index(drop=True)


def add_target(features: pd.DataFrame) -> pd.DataFrame:
    """Add the answer key for TRAINING only: did the next hour close up?"""
    out = features.copy()
    out["next_ret"] = out["return"].shift(-1)
    out["next_up"] = (out["next_ret"] > 0).astype(int)
    return out.iloc[:-1]   # the last row has no next hour yet