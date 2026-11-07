from pathlib import Path

import joblib
import pandas as pd

from app.features import FEATURE_COLS, make_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MIN_CANDLES = 21   # make_features needs 21 candles to produce its first row

_models = {}


def load_model(name: str = "logreg"):
    """Load a saved model once and reuse it."""
    if name not in _models:
        _models[name] = joblib.load(MODEL_DIR / f"{name}.joblib")
    return _models[name]


def predict_latest(candles: pd.DataFrame, model_name: str = "logreg") -> dict:
    """Probability that the hour after the newest candle closes up.

    `candles` must contain only CLOSED candles, oldest first, with columns
    time, open, high, low, close, volume.
    """
    if len(candles) < MIN_CANDLES:
        raise ValueError(f"Need at least {MIN_CANDLES} candles, got {len(candles)}")

    feats = make_features(candles)
    row = feats.tail(1)
    p_up = float(load_model(model_name).predict_proba(row[FEATURE_COLS])[0, 1])

    return {
        "time": row["time"].iloc[0].isoformat(),
        "price": float(row["close"].iloc[0]),
        "p_up": round(p_up, 4),
        "model": model_name,
    }