import os
import pandas as pd

from datetime import datetime
from dotenv import load_dotenv
from polygon import RESTClient
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage
from typing import List
load_dotenv('secrets.env')
API_KEY = os.environ["POLYGON_API_KEY"]

client = RESTClient(api_key=API_KEY)

ticker = "AAPL"

start = "2023-01-03"
end = "2023-01-03"

aggs = []
for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_=start, to=end, limit=50000):
    aggs.append(a)

df = pd.DataFrame([agg.__dict__ for agg in aggs])

df["timestamp"] = df["timestamp"].apply(
    lambda ts: datetime.fromtimestamp(ts/1000.0))

df.to_csv(f"AAPL-{start}-{end}.csv")
