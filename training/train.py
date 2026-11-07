from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import FEATURE_COLS, add_target, make_features

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "btc_1h.csv"
MODEL_DIR = ROOT / "models"


def main():
    raw = pd.read_csv(DATA_FILE, parse_dates=["time"])
    data = add_target(make_features(raw))

    X = data[FEATURE_COLS]
    y = data["next_up"]
    print(f"Training on {len(data)} rows: {data['time'].min()} -> {data['time'].max()}")

    logreg = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42).fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(logreg, MODEL_DIR / "logreg.joblib")
    joblib.dump(xgb, MODEL_DIR / "xgb.joblib")
    print("Saved models to", MODEL_DIR)

    # Sanity check only: accuracy on data a model was trained on is NOT a
    # measure of skill. The walk-forward results are the honest numbers.
    for name, m in [("logreg", logreg), ("xgb", xgb)]:
        acc = (m.predict(X) == y).mean() * 100
        print(f"{name}: accuracy on its own training data = {acc:.1f}%")


if __name__ == "__main__":
    main()