import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from app.log import append_prediction, load_scored

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app import ingest

DISCLAIMER = (
    "Experimental learning project. Walk-forward tests found no reliable "
    "edge after fees. Not financial advice."
)

state = {"latest": None}            # newest prediction, filled in by the background task
clients: set[WebSocket] = set()     # currently connected websocket clients


def with_disclaimer(result: dict) -> dict:
    return {**result, "disclaimer": DISCLAIMER}


async def store_prediction(result: dict):
    """Called by the live feed for every new prediction."""
    state["latest"] = result
    try:
        append_prediction(result)
    except Exception as e:                      # logging must never stop the live feed
        print("Could not log prediction:", repr(e))

    for ws in list(clients):                    # copy: the set can change while we await
        try:
            await ws.send_json(with_disclaimer(result))
        except Exception:
            clients.discard(ws)                 # that client went away


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
    return {
        "status": "ok",
        "has_prediction": state["latest"] is not None,
        "websocket_clients": len(clients),
    }


@app.get("/predict")
def predict():
    if state["latest"] is None:
        raise HTTPException(status_code=503, detail="Warming up: no prediction yet.")
    return with_disclaimer(state["latest"])

@app.get("/candles")
def candles(limit: int = 200):
    rows = list(ingest.buffer)[-max(1, min(limit, 200)):]
    if not rows:
        raise HTTPException(status_code=503, detail="Warming up: no candles yet.")
    return [
        {**row, "time": datetime.fromtimestamp(row["time"] / 1000, tz=timezone.utc).isoformat()}
        for row in rows
    ]

@app.get("/history")
def history(limit: int = 1000):
    return load_scored()[-max(1, min(limit, 1000)):]

@app.websocket("/ws/predictions")
async def ws_predictions(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        if state["latest"] is not None:         # catch the new client up immediately
            await websocket.send_json(with_disclaimer(state["latest"]))
        while True:
            await websocket.receive_text()      # keeps the connection open
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)