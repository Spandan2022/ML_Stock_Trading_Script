# ML_Stock_Trading_Script
## Overview
This project is a machine learning model that uses PyTorch to analyze and trade stocks using real-time market data with Alpaca API. The model was trained using 2 year minute by minute historical data scraped from Polygon.io. This model is specifically optimized for short-term trades (day-trading) on NASDAQ.

### Historical Training Data 
This ML Stock Trading Script includes a `data_scraper` directory which includes the code used to collect and organize 2 year minute by minute historical data for all stocks on NASDAQ. Each stock's historical data is stored in a csv file under it's own directory in `historcial_data`. 

### Data Pre-Processing
Using the historcal data we then calculate the daily ATR (Average True Range) of each stock, which allows us to determine the stocks with the highest volatility. By choosing the stock with highest volatility we ensuring the stock is suitable for short-term trading. 


## Getting Started
After forking this repository, the first thing you should do is pip install all packages in requirements.txt. This can be acomplished by running the following command in the terminal.
```bash
$ pip install -r requirements.txt
```
After all the requirements have been sucessfully installed create an `secrets.env` file within the `data_scraper` directory and include the following code and replace `<API_KEY>` with your API key from Polygon.io.
```python
POLYGON_API_KEY = "<API_KEY>"
```
Now `cd` into the `data_scrapper` directory. When the program, `polygon_scraper.py`, is run it will create a csv file with following naming format: `SYMBL-startDate-endDate.csv`. To run the program run the following command in the terminal.
```bash
$ python3 polygon_scraper.py
```

## File Structure
Note there are multiple directories which each having a dedicated purpose. There are a total of 2 directories within this project.

### data_scraper
The `data_scraper` directory includes the code used to collect and organize 2 year minute by minute historical data for all stocks on NASDAQ.

### real_time_data
The `real_time_data` directory includes three files, `config.py`, `news_stream`, and `stock_stream`. 

#### config.py
The `config.py` file includes link to the Alapaca API's websocket URL as well as the Alpaca API keys used by news_stream`, and `stock_stream`.

#### news_stream
The `news_stream.py` file uses websockets to stream all market related & financial news from Benzinga.com in real-time. Benzinga is a financial news outlet that is brokers' primary source of market news. 

#### stock_stream
The `stock_stream.py` file uses websockets to stream all real-time data for NASDAQ. It is important to note that the program only runs when the markets are open (9:30 am to 4:00 pm EST) otherwise, the program will simply give an warning: `WARNING: NASDAQ Market is CLOSED`.