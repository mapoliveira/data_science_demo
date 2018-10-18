### Plotly with Gecko cryto data
# Install plotly in the terminal: $ pip install plotly

import sys
import pandas as pd
from datetime import datetime
sys.path.insert(0, '../src') # Identify src directory
from getGeckoData import *
from plotlyTimeSeries import *

numberOfCoins = 50
numberOfDays = 400

listData, priceDf = getGeckoMarket(numberOfCoins, numberOfDays)
datetimeX = pd.to_datetime(priceDf.index.values, unit='ms')

plotlyTimeSeries(datetimeX, listData)
