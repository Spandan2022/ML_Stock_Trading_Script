# ML_Stock_Trading_Script
This project is a machine learning model that uses PyTorch to analyze and trade stocks in using real-time market data with Alpaca API. The model was trained using historical data scraped from Polygon.io.

The real-time data being streamed in this project are in relation to New York Stock Exchange and NASDAQ, therefore it is important to note that the program will only run while the markets are open (9:30 am to 4:00 pm EST) otherwise, the program will simply give an ERROR: Market is CLOSED!

## Getting Started
After forking this repository, the first thing you should do is pip install all packages in requirements.txt. This can be acomplished by running the following command in the terminal.
```bash
$ pip install -r requirements.txt
```
