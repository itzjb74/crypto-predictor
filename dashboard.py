import altair as alt
import pandas as pd
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="BTC next-hour predictor")
st.title("BTC next-hour direction (experimental)")
st.warning(
    "Learning project. Walk-forward tests found no reliable edge after fees. "
    "Not financial advice."
)


@st.fragment(run_every=10)
def live_panel():
    try:
        data = requests.get(f"{API}/predict", timeout=5).json()
    except Exception as e:
        st.error(f"Can't reach the API at {API}: {e}")
        return

    if "p_up_logreg" not in data:
        st.info("The API is warming up, one moment...")
        return

    st.metric("Last closed price (USDT)", f"{data['price']:,.2f}")
    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression: P(up)", f"{data['p_up_logreg']:.1%}")
    col2.metric("XGBoost: P(up)", f"{data['p_up_xgb']:.1%}")
    st.caption(
        f"Based on the candle starting {data['candle_time']} UTC. "
        f"Predicts the hour starting {data['predicts_hour_starting']} UTC. "
        "Values near 50% mean the model has no real preference."
    )


live_panel()


@st.fragment(run_every=60)
def chart_panel():
    try:
        rows = requests.get(f"{API}/candles", timeout=5).json()
    except Exception as e:
        st.error(f"Can't reach the API at {API}: {e}")
        return

    if not isinstance(rows, list):
        st.info("Waiting for candles...")
        return

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.set_index("time")
    df["20-hour average"] = df["close"].rolling(20).mean()

    chart_df = df.reset_index().melt(
        "time",
        value_vars=["close", "20-hour average"],
        var_name="series",
        value_name="price",
    )
    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("time:T", title="Time (UTC)", axis=alt.Axis(format="%b %d")),
            y=alt.Y("price:Q", scale=alt.Scale(zero=False), title="Price (USDT)"),
            color=alt.Color("series:N", title=None),
        )
        .properties(height=350)
    )

    st.subheader("Last 200 closed hourly candles")
    st.altair_chart(chart)
    st.caption("Times are UTC. Only closed candles are shown.")


chart_panel()

@st.fragment(run_every=60)
def history_panel():
    try:
        rows = requests.get(f"{API}/history", timeout=5).json()
    except Exception as e:
        st.error(f"Can't reach the API at {API}: {e}")
        return

    st.subheader("Prediction scoreboard")
    if not isinstance(rows, list) or not rows:
        st.info("No predictions logged yet. One is logged every hour.")
        return

    df = pd.DataFrame(rows)
    done = df[df["went_up"].notna()]
    n = len(done)

    if n == 0:
        st.info("Predictions are being logged. The first score arrives once the next candle closes.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Scored predictions", n)
        c2.metric("Logistic Regression right", f"{done['logreg_correct'].astype(bool).mean():.1%}")
        c3.metric("XGBoost right", f"{done['xgb_correct'].astype(bool).mean():.1%}")
        st.caption(
            f"With only {n} scored predictions, pure luck can land anywhere within "
            f"about ±{98 / n ** 0.5:.0f} points of 50%."
        )

    recent = df.tail(24).iloc[::-1]
    table = pd.DataFrame({
        "Candle (UTC)": pd.to_datetime(recent["candle_time"]).dt.strftime("%b %d %H:%M"),
        "Close": recent["price"],
        "LogReg P(up)": recent["p_up_logreg"],
        "XGB P(up)": recent["p_up_xgb"],
        "Next hour": recent["went_up"].map({True: "up", False: "down"}).fillna("pending"),
        "LogReg": recent["logreg_correct"].map({True: "right", False: "wrong"}).fillna("-"),
        "XGB": recent["xgb_correct"].map({True: "right", False: "wrong"}).fillna("-"),
    })
    st.dataframe(table, hide_index=True)


history_panel()