### Plotly with Gecko cryto data
# Install plotly in the terminal: $ pip install plotly
import requests
import pandas as pd
from datetime import datetime

numberOfCoins = 50
numberOfDays = 400

# List of coins in a dataframe
url= "https://api.coingecko.com/api/v3/"
urlListCoins = url + "coins/list"
print("\nData obtained from: " + url)
page = requests.get(urlListCoins)
j = page.json()
dfCoins = pd.DataFrame(j)
listData = []

## Obtain historical data for each coin
days = str(numberOfDays)
print("\nDays considered: " + days)
print("\nObtaining price for... ")
for c in range(0,numberOfCoins):
    coinName = dfCoins.id[c]
    coinSymbol = dfCoins.symbol[c]
    print("(" + coinSymbol + ") " + coinName)
    coinDataUrl = url + '/coins/' + coinName + '/market_chart?vs_currency=eur&days='+ days

    page = requests.get(coinDataUrl)
    j = page.json()
    coinPriceDf = pd.DataFrame(j['prices'], columns = ['time', coinSymbol])
    coinPriceDf['time'] = pd.to_datetime(coinPriceDf['time'], unit='ms').dt.round('1min')
    coinPriceDf.set_index('time', inplace =True)
    listData.append(coinPriceDf)
priceDf = pd.concat(listData, axis=1)
datetimeX = pd.to_datetime(priceDf.index.values, unit='ms')

import plotly
import plotly.graph_objs as go

dataPanda = []
for k, df in enumerate(listData):
    trace = go.Scatter( x = datetimeX,
            y = df.iloc[:,0],
            mode = 'markers+lines',
            name = df.columns[0]
            )
    dataPanda.append(trace)
fig = dict(data=dataPanda)
plotly.offline.plot(fig)

print('END! Find interactive graphic in the browser.')

