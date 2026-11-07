import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import ingest

DISCLAIMER = (
    "Experimental learning project. Walk-forward tests found no reliable "
    "edge after fees. Not financial advice."
)

state = {"latest": None}   # newest prediction, filled in by the background task


def store_prediction(result: dict):
    state["latest"] = result


def log_task_result(task: asyncio.Task):
    if not task.cancelled() and task.exception():
        print("Background feed crashed:", repr(task.exception()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(ingest.run(store_prediction))
    task.add_done_callback(log_task_result)
    yield
    task.cancel()


app = FastAPI(title="BTC next-hour predictor (experimental)", lifespan=lifespan)


@app.get("/")
def root():
    return {"name": "BTC next-hour direction predictor", "disclaimer": DISCLAIMER}


@app.get("/health")
def health():
    return {"status": "ok", "has_prediction": state["latest"] is not None}


@app.get("/predict")
def predict():
    if state["latest"] is None:
        raise HTTPException(status_code=503, detail="Warming up: no prediction yet.")
    return {**state["latest"], "disclaimer": DISCLAIMER}