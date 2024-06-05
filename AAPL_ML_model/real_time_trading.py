import websocket
import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import ssl
import certifi
import torch
import torch.nn as nn
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce 
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.preprocessing import MinMaxScaler
import alpaca_config

API_KEY = alpaca_config.API_KEY
SECRET_KEY = alpaca_config.SECRET_KEY
BASE_URL = alpaca_config.ENDPOINT
DATA_URL = 'wss://stream.data.alpaca.markets/v2/iex'

api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Define model class
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load optimized_model.pth
model = SimpleLSTM(input_size=7, hidden_size=50, output_size=1)
model.load_state_dict(torch.load('optimized_model.pth'))
model.eval()

scaler = MinMaxScaler()

# Storage for incoming AAPL stream
bars = []
time_stamps = []
'''
# Initialize Matplotlib figure and axes
fig, ax = plt.subplots()
line_open, = ax.plot([], [], label='Open Price')
line_close, = ax.plot([], [], label='Close Price')
ax.legend()
'''

# Set SYMBOL to desired stock symbol
# SYMBOL can be set to a list of stock symbols
SYMBOL = "AAPL"

def on_open(ws):
    print('Oppening Connection...')
    
    auth_data = {"action": "auth", 
                 "key": API_KEY, 
                 "secret": SECRET_KEY}
    # Convert auth_data dictionary to JSON-formatted string
    auth_data_str = json.dumps(auth_data)
    # Send websocket authentication
    ws.send(auth_data_str)
    
    # Listen for symbol's trades, quotes, & minute bars
    # For trades use: "trades":[SYMBOL]
    # For quotes use: "quotes":[SYMBOL]
    # For minute bars use: "bars":[SYMBOL]
    listen_message = {"action":"subscribe", "bars":[SYMBOL]}
    listen_message_str = json.dumps(listen_message)
    # Send websocket listen request
    ws.send(listen_message_str)

# Define websocket on_message function to process incoming bar data
def on_message(ws, message):
    global bars, time_stamps
    print("Received a message:")
    parsed_message = json.loads(message)
    formatted_message = json.dumps(parsed_message, indent=4)
    print(formatted_message)

    if 'bars' in parsed_message:
        bar = parsed_message['bars'][0]
        bars.append(bar)
        time_stamps.append(datetime.datetime.now())
        if len(bars) > 60:
            bars.pop(0)
            time_stamps.pop(0)

        if len(bars) == 60:
            df = pd.DataFrame(bars)
            features = df[['o', 'h', 'l', 'c', 'v']].values  # open, high, low, close, volume
            features_scaled = scaler.fit_transform(features)
            X_test = np.array([features_scaled])
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

            # Make the prediction
            with torch.no_grad():
                predicted_price = model(X_test_tensor).item()

            # Get the current close price
            current_close = df.iloc[-1]['c']

            # Check the current position
            try:
                position = api.get_position('AAPL')
                position_qty = int(position.qty)
            except:
                position_qty = 0

            # Trading logic: buy if prediction is higher, sell if prediction is lower
            if predicted_price > current_close and position_qty == 0:
                order = MarketOrderRequest(
                    symbol="AAPL",
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                api.submit_order(order)
                print(f"Bought 1 share of AAPL at {current_close}")
            elif predicted_price < current_close and position_qty > 0:
                order = MarketOrderRequest(
                    symbol="AAPL",
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                api.submit_order(order)
                print(f"Sold 1 share of AAPL at {current_close}")

'''
def update_plot(frame):
    if len(bars) > 0:
        df = pd.DataFrame(bars)
        times = time_stamps
        open_prices = df['o'].values
        close_prices = df['c'].values

        line_open.set_data(times, open_prices)
        line_close.set_data(times, close_prices)
        ax.relim()
        ax.autoscale_view()

    return line_open, line_close
'''

# Define websocket on_close function to close connection
def on_close(ws):
    print("Closed connection")

# Define websocket on_error function to print websocket errors
def on_error(ws, error):
    print("Error:", error)

# Define boolean function to check if current time is within market hours
def is_market_open():
    now = datetime.datetime.now(datetime.timezone.utc)
    EST = datetime.timezone(datetime.timedelta(hours=-5))
    current_time = now.astimezone(EST)

    if current_time.weekday() >= 5:
        return False

    market_open_time = current_time.replace(hour=9, minute=30, second=0)
    market_close_time = current_time.replace(hour=16, minute=0, second=0)

    return market_open_time <= current_time <= market_close_time

# Ensure market is currently open to initialize websocket
if is_market_open():
    print("Market is OPEN!")

    socket = DATA_URL
    ws = websocket.WebSocketApp(
        socket,
        on_open=on_open,
        on_error=on_error,
        on_message=on_message,
        on_close=on_close
    )
    ws.run_forever(sslopt={"ca_certs": certifi.where()})
    
    # Run the WebSocket in a separate thread to keep it responsive
    import threading
    ws_thread = threading.Thread(target=ws.run_forever, kwargs={'sslopt': {"ca_certs": certifi.where()}})
    ws_thread.daemon = True
    ws_thread.start()

    '''
    # Set up Matplotlib animation
    ani = FuncAnimation(fig, update_plot, interval=1000)
    plt.show()
    '''

else:
    print("WARNING: NASDAQ Market is CLOSED\nHours of Operation: 9:30am - 4:00pm EST")
    
    