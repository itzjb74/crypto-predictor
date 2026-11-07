"""Score the live prediction log: accuracy, plus a paper-traded return after fees.

Run from the project folder:  python -m training.evaluate_live
"""
import numpy as np
import pandas as pd

from app.log import load_scored

FEE = 0.001   # 0.1% per trade, the same assumption as the Phase 4-5 backtests


def paper_trade(done: pd.DataFrame, col: str, threshold: float, fee: float = FEE) -> dict:
    """Hold BTC for the next hour whenever P(up) > threshold; pay a fee on every switch."""
    pos = (done[col] > threshold).astype(int)
    trades = (pos - pos.shift(1).fillna(0)).abs()
    gross = pos * done["next_ret"]
    net = gross - trades * fee
    return {
        "hours_in_market": int(pos.sum()),
        "trades": int(trades.sum()),
        "gross_%": round(((1 + gross).prod() - 1) * 100, 1),
        "net_%": round(((1 + net).prod() - 1) * 100, 1),
    }


def main():
    rows = load_scored()
    if not rows:
        print("No predictions logged yet. Start the server and let it run.")
        return

    df = pd.DataFrame(rows)
    df["next_ret"] = df["price"].shift(-1) / df["price"] - 1
    done = df[df["went_up"].notna()].copy()
    n = len(done)

    print(f"Logged: {len(df)} | scored: {n} | "
          f"from {df['candle_time'].iloc[0]} to {df['candle_time'].iloc[-1]}")
    if n < 2:
        print("Not enough scored predictions yet. Let the server run a while longer.")
        return

    went_up = done["went_up"].astype(bool)
    print(f"Share of hours that went up: {went_up.mean():.1%}")
    print(f"With {n} predictions, a coin flip alone can land within "
          f"about ±{98 / np.sqrt(n):.1f} points of 50%.")
    print()

    table = []
    for name, col in [("Logistic Regression", "p_up_logreg"), ("XGBoost", "p_up_xgb")]:
        acc = ((done[col] > 0.5) == went_up).mean() * 100
        print(f"{name}: right {acc:.1f}% of the time")
        for thr in [0.50, 0.55, 0.60]:
            table.append({"model": name, "threshold": thr, **paper_trade(done, col, thr)})

    print()
    print(pd.DataFrame(table).to_string(index=False))

    hold = (1 + done["next_ret"]).prod() - 1
    print(f"\nBuy and hold over the same hours: {hold * 100:.1f}%   |   Cash: 0.0%")
    print("\nFor reference: the Phase 5 walk-forward backtest found about 51.5% accuracy "
          "for both models, and every strategy lost to cash after fees.")


if __name__ == "__main__":
    main()