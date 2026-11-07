import csv
import json
from pathlib import Path

import pandas as pd

LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "predictions.csv"
FIELDS = ["candle_time", "price", "predicts_hour_starting", "p_up_logreg", "p_up_xgb"]


def append_prediction(result: dict) -> bool:
    """Append a prediction unless this candle is already logged. Returns True if written."""
    if LOG_FILE.exists():
        with open(LOG_FILE, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and rows[-1]["candle_time"] == result["candle_time"]:
            return False
    LOG_FILE.parent.mkdir(exist_ok=True)
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: result[k] for k in FIELDS})
    return True


def load_scored() -> list[dict]:
    """All logged predictions, each scored once the following candle has closed."""
    if not LOG_FILE.exists():
        return []
    df = pd.read_csv(LOG_FILE, parse_dates=["candle_time"])
    resolved = (df["candle_time"].shift(-1) - df["candle_time"]) == pd.Timedelta(hours=1)
    went_up = df["price"].shift(-1) > df["price"]
    df["went_up"] = went_up.where(resolved)
    for name in ["logreg", "xgb"]:
        df[f"{name}_correct"] = ((df[f"p_up_{name}"] > 0.5) == went_up).where(resolved)
    return json.loads(df.to_json(orient="records", date_format="iso"))