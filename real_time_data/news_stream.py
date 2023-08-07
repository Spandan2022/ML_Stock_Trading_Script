"""
Overview: This file establishes a connection with websocket to access all
real-time news for a specific stock, list of stock symbols, or all stocks 
and reads the data into the terminal.

Resources: 
    - https://docs.alpaca.markets/docs/streaming-market-data (Streaming Market Data)
    - https://docs.alpaca.markets/docs/real-time-stock-pricing-data (Streaming Stock Data)
    - https://docs.alpaca.markets/docs/streaming-real-time-news (Streaming News Data)
    - https://www.youtube.com/watch?v=Mv6c_9FqNx4 (Streaming)
    - https://www.youtube.com/watch?v=EjQ-3iXEPEs&t=277s (Visualization)
    - https://www.qmr.ai/cryptocurrency-trading-bot-with-alpaca-in-python/
    - https://stackoverflow.com/questions/73022927/alpaca-data-not-streaming

Terminal Market News Stream:
    - $ wscat -c wss://stream.data.alpaca.markets/v1beta1/news
    - Note there will only be a 10 second window to authenticate
    - $ {"action": "auth", "key": "PKA1C5GW4X4UOWGT809D", "secret": "0C0v9NZxGbHYafp4ZD0T925hk6H9bS0MTxqIsAud"}
    - There are varying requests for streaming news data:
        - $ {"action": "subscribe", "news":["*"]}
        - $ {"action":"subscribe","news":["<SYMBOL>"]}
"""

import websocket, json
import datetime
import real_time_data.config as config

# Set SYMBOL to desired stock symbol
# SYMBOL can be set to a list of stock symbols
# SYMBOL can also be set to a "*" to listen to all stock symbols
SYMBOL = "*"

# Define websocket on_open function to initialize connection
def on_open(ws):
    print("Openning Connection...")
    auth_data = {"action": "auth", 
                 "key": config.API_KEY, 
                 "secret": config.SECRET_KEY}
    # Convert auth_data dictionary to JSON-formatted string
    auth_data_str = json.dumps(auth_data)  
    # Send websocket authentication
    ws.send(auth_data_str)
    # Listen for symbol's stream
    listen_message = {"action": "subscribe","news":[SYMBOL]}
    # Convert listen_message dictionary to JSON-formatted string
    listen_message_str = json.dumps(listen_message)
    print(listen_message_str)  
    # Send websocket listen request
    ws.send(listen_message_str)

# Define websocket on_message function to print market news recieved
def on_message(ws, message):
    print("recieved a message:")
    print(message)
    
# Define websocket on_close function to close connection
def on_close(ws):
    print("closed connection")

# Define websocket on_error function to print websocket errors     
def on_error(ws, error):
    print("error:", error)

# Define boolean function to check if current time is within market hours 
def is_market_open():
    # Get current UTC time 
    now = datetime.datetime.now(datetime.timezone.utc)
    # Adjust UTC time to EST timezone
    EST = datetime.timezone(datetime.timedelta(hours=-5))
    current_time = now.astimezone(EST)

    # Check if the current day is a Saturday (5) or Sunday (6)
    if current_time.weekday() >= 5:
        return False
    
    # Check if within market hours (9:30am to 4:00pm EST)
    market_open_time = current_time.replace(hour= 9, minute= 30, second= 0)
    market_close_time = current_time.replace(hour= 16, minute= 0, second= 0)

    return market_open_time <= current_time <= market_close_time

# If statement to ensure market is currently open to initalize websocket
if is_market_open():
    print("Market is OPEN!")

    # Websocket endpoint url
    socket = "wss://stream.data.alpaca.markets/v1beta1/news"

    # Intializing websocket
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_error=on_error, 
                                on_message=on_message, on_close=on_close)
    ws.run_forever()

else:
    print("Market is CLOSED!\nHours of Operation: 9:30am - 4:00pm EST")