import requests
import pandas as pd
from datetime import datetime

def getGeckoMarket(numberOfCoins, numberOfDays):

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

    return listData, priceDf
