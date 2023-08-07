# ML_Stock_Trading_Script
This project is a machine learning model that uses PyTorch to analyze and trade stocks in using real-time market data with Alpaca API.

The real-time data being streamed in this project are in relation to New York Stock Exchange and NASDAQ, therefore it is important to note that the program will only run while the markets are open (9:30 am to 4:00 pm EST) otherwise, the program will simply give an ERROR: Market is CLOSED!


Framework:
1. Gather all Market related News
2. Identify using ML if News has positive, negative or no impact on a Stock SYMBOL
3. Check to see if we are holding Stock related to Symbol
4. If News is Positive && we are Holding Stock == hold stock
   If News is Positive && we are Not Holding Stock == buy stock
   If News has no impact == do nothing
   If News is Negative && we are Holding Stock == sell stock
   If News is Negative && we are Not Holding Stock == do nothing
5. If Positive && Not Holding then gather Realtime Trade & Quote Data for that SYMBOL
   Identify buy idicators for that SYMBOL with moving average
   Buy when indicated 
6. If Negative && Holding then gather Realtime Trade & Quote Data for that SYMBOL
   Identify sell idicators for that SYMBOL with moving average
   Sell when indicated
