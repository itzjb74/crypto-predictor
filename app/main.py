from fastapi import FastAPI

app = FastAPI(title="BTC next-hour predictor (experimental)")


@app.get("/")
def root():
    return {
        "name": "BTC next-hour direction predictor",
        "disclaimer": (
            "Experimental learning project. Walk-forward tests found no reliable "
            "edge after fees. Not financial advice."
        ),
    }


@app.get("/health")
def health():
    return {"status": "ok"}