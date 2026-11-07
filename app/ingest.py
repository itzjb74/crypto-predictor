import asyncio
import json
from datetime import datetime, timezone

import websockets

SYMBOL = "btcusdt"
INTERVAL = "1m"   # 1m shows a CLOSED candle every minute, handy for testing; the model will use "1h"
URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_{INTERVAL}"


async def main():
    print("Connecting to", URL)
    async with websockets.connect(URL) as ws:
        async for message in ws:
            k = json.loads(message)["k"]
            start = datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc)
            status = "CLOSED     " if k["x"] else "in progress"
            print(f"{status} | candle start {start:%H:%M} UTC | close {k['c']} | volume {k['v']}")


if __name__ == "__main__":
    asyncio.run(main())