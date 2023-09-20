import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 
from ta import add_all_ta_features
from ta.utils import dropna


aapl = pd.read_csv("data/AAPL.csv")
amzn = pd.read_csv("data/AMZN.csv")
atvi = pd.read_csv("data/ATVI.csv")
dbx = pd.read_csv("data/DBX.csv")
ea = pd.read_csv("data/EA.csv")
goog = pd.read_csv("data/GOOG.csv")
meta = pd.read_csv("data/META.csv")
nflx = pd.read_csv("data/NFLX.csv")
ntdoy = pd.read_csv("data/NTDOY.csv")
para = pd.read_csv("data/PARA.csv")
pins = pd.read_csv("data/PINS.csv")
rblx = pd.read_csv("data/RBLX.csv")
sono = pd.read_csv("data/SONO.csv")
sony = pd.read_csv("data/SONY.csv")
spot = pd.read_csv("data/SPOT.csv")
tsco = pd.read_csv("data/TSCO.L.csv")
tsla = pd.read_csv("data/TSLA.csv")
vod = pd.read_csv("data/VOD.L.csv")
zm = pd.read_csv("data/ZM.csv")

stocks = [aapl, amzn, atvi, dbx, ea, goog, meta, nflx, ntdoy, para, pins, rblx, sono, sony, spot, tsco, tsla, vod, zm]
ma_day = [10,20,50]
stock_symbols = [
    "AAPL",
    "AMZN",
    "ATVI",
    "DBX",
    "EA",
    "GOOG",
    "META",
    "NFLX",
    "NTDOY",
    "PARA",
    "PINS",
    "RBLX",
    "SONO",
    "SONY",
    "SPOT",
    "TSCO",
    "TSLA",
    "VOD",
    "ZM"
]
stock_data = []
for symbol in stock_symbols:
    file_path = os.path.join("data", f"{symbol}.csv")
    stock_df = pd.read_csv(file_path)
    stock_data.append(stock_df)

# Create a DataFrame for adjusted closing prices
adj_close_df = pd.concat([df.set_index("Date")["Adj Close"].rename(symbol) for symbol, df in zip(stock_symbols, stock_data)], axis=1)

# Calculate the correlation matrix
corr_matrix = adj_close_df.corr()

# Create a correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Stock Price Correlation Heatmap")
plt.show()

def moving_average():
    for i in range(19):
        stock = stocks[i]
        stock['Date']= pd.to_datetime(stock['Date'])

        stock.set_index('Date', inplace=True)

        for ma in ma_day:
            column_name = "MA for %s days" %(str(ma))
            stock[column_name] = stock['Adj Close'].rolling(window=ma,center=False).mean()
                
        stock.tail()

        stock[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(12,5))
        plt.xlabel("Date")
        plt.ylabel('Price')
        plt.title(f"{stock_symbols[i]} Moving Averages")
        plt.savefig(f"{stock_symbols[i]}_ma.png")