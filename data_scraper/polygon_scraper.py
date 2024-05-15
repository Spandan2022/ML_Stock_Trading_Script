import os
import pandas as pd
import time

from datetime import datetime, timedelta
from dotenv import load_dotenv
from polygon import RESTClient
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage
from typing import List

load_dotenv('secrets.env')
API_KEY = os.environ["POLYGON_API_KEY"]


def scrape_aggregate_data(client: RESTClient, ticker: str, start: datetime, end: datetime):
    # If there's no folder with stock data -> create new folder with the ticker name
    if not os.path.exists(ticker):
        os.makedirs(ticker)

    # Reference date for API call range
    current_date = start

    try:
        # while current_date is sooner than the end date
        while current_date >= end:
            # Grab data every 90 days
            past_date = current_date - timedelta(days=90)

            # Check if past_date is beyond the desired end date
            if past_date < end:
                past_date = end

            print(f"Gathering data from {current_date} to {past_date}")

            aggs = []
            for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_=past_date, to=current_date, limit=50000):
                aggs.append(a)

            # Create and save data to csv file
            df = pd.DataFrame([agg.__dict__ for agg in aggs])

            df["timestamp"] = df["timestamp"].apply(
                lambda ts: datetime.fromtimestamp(ts/1000.0))

            start_string = current_date.strftime("%m-%d-%Y")
            end_string = past_date.strftime("%m-%d-%Y")

            df.to_csv(f"./{ticker}/{ticker}-{start_string}-{end_string}.csv")

            # Set current_date to past_date minus 1 day
            current_date = past_date - timedelta(days=1)

            time.sleep(20)  # Rate limited to 5 API calls every minute
    except:
        return


if __name__ == '__main__':
    client = RESTClient(api_key=API_KEY)

    ticker = "AAPL"

    start = datetime(2024, 5, 1)  # Start of May cuz I felt like it
    # Do over the span of two years not including the last month
    end = start - timedelta(days=700)

    scrape_aggregate_data(client, ticker, start, end)
