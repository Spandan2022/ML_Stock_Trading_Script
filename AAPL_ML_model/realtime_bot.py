'''
Using Alapaca API for real-time paper trading: 
https://docs.alpaca.markets/docs/getting-started-with-trading-api

WARNING:
Note any order placed outside of NASDAQ trading hours will be queued 
up for release the next trading day. 
Extended hours trading can be accomplished however are higher risk 
due to less liquidity. Alpaca currently supports these extended hours:
    - Pre-market: 4:00am - 9:30am EST (Mon to Fri)
    - After-hours: 4:00pm
'''
import config
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

# Get account info
account = trading_client.get_account()

# Declare symbol & quantity you want to short
SYMBOL = 'NVDA'
QNTY = 1

# Check restrictions on account
if account.trading_blocked:
    print('Account is currently restricted from trading.')
    
# Check available balance for opening new positions
print('${} is available for opening new posisitons.'.format(account.buying_power))

# Check if particular symbol can be traded
asset = trading_client.get_asset(SYMBOL)

if asset.tradable:
    print(SYMBOL + ' is currently tradable')


# SHORTS:
# Preparing market order
market_order_data = MarketOrderRequest(
                        symbol = SYMBOL,
                        qty = QNTY,
                        side = OrderSide.SELL,
                        time_in_force = TimeInForce.GTC
                    )
# Market order
market_order = trading_client.submit_order(
                    order_data = market_order_data
                )

print(str(QNTY) + ' short of ' + SYMBOL + ' sucessfuly placed')

portfolio = trading_client.get_all_positions()
print('PORTFOLIO SUMMARY:')
for position in portfolio:
    print("{} shares of {}".format(position.qty, position.symbol))
