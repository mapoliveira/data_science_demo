import requests
import pandas as pd
from datetime import datetime

def getGeckoMarket(numberOfCoins, numberOfDays):

    # List of coins in a dataframe
    url= "https://api.coingecko.com/api/v3/"
    urlListCoins = url + "coins/list"
    print("\nData obtained from: " + url)
    j = requests.get(urlListCoins).json()
    dfCoins = pd.DataFrame(j)
    
    ## Obtain historical data for each coin
    print("\nDays considered: " + str(numberOfDays))
    print("\nObtaining price for... ")
    listData = []
    days = str(numberOfDays) 
    for c in range(0, numberOfCoins):
        coinName = dfCoins.id[c]
        coinSymbol = dfCoins.symbol[c]
        print("(" + coinSymbol + ") " + coinName)
        coinDataUrl = url + '/coins/' + coinName + '/market_chart?vs_currency=eur&days='+ days

        j = requests.get(coinDataUrl).json()
        coinPriceDf = pd.DataFrame(j['prices'], columns = ['time', coinSymbol])
        coinPriceDf['time'] = pd.to_datetime(coinPriceDf['time'], unit='ms').dt.round('1min')
        coinPriceDf.set_index('time', inplace =True)
        listData.append(coinPriceDf)
    priceDf = pd.concat(listData, axis=1)
    return listData, priceDf

#def getCoinFeatures()    
#    pairs = [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)] 
#    for shift1, shift2 in pairs:
#        col1 = f"A(_bitcoin) - {shift1}"
#        col2 = f"A(_bitcoin) - {shift2}"
#        new_col = f"C(_bitcoin){shift}_to_{shift2}"
#        df[new_col] = df[col1] - df[col2]


