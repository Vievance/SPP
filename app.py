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

ftse = pd.read_csv("data/FTSE.csv")

paths = [
    "data/AAPL.csv",
    "data/AMZN.csv",
    "data/ATVI.csv",
    "data/DBX.csv",
    "data/EA.csv",
    "data/GOOG.csv",
    "data/META.csv",
    "data/NFLX.csv",
    "data/NTDOY.csv",
    "data/PARA.csv",
    "data/PINS.csv",
    "data/RBLX.csv",
    "data/SONO.csv",
    "data/SONY.csv",
    "data/SPOT.csv",
    "data/TSCO.L.csv",
    "data/TSLA.csv",
    "data/VOD.L.csv",
    "data/ZM.csv"
]

stocks = [aapl, amzn, atvi, dbx, ea, goog, meta, nflx, ntdoy, para, pins, rblx, sono, sony, spot, tsco, tsla, vod, zm]

'''def remove_close_volume_columns():
    for i in range(19):

        # Check if 'Close' and 'Volume' columns exist in the DataFrame
        if 'Close' in stocks[i].columns and 'Volume' in stocks[i].columns:
            # Drop the 'Close' and 'Volume' columns
            stocks[i].drop(columns=['Close', 'Volume'], inplace=True)

            # Save the modified DataFrame back to the same CSV file
            stocks[i].to_csv(paths[i], index=False)

# Usage example: Remove 'Close' and 'Volume' columns from CSV files in the 'data' directory
remove_close_volume_columns()'''

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
    "TSCO.L",
    "TSLA",
    "VOD.L",
    "ZM"
]
stock_data = []
for symbol in stock_symbols:
    file_path = os.path.join("data", f"{symbol}.csv")
    stock_df = pd.read_csv(file_path)
    stock_data.append(stock_df)



def annotate_heatmap(data, fmt=".2f", fontsize=10, ax=None, cmap="coolwarm"):
    if ax is None:
        ax = plt.gca()
    # Fill both upper and lower triangles
    mask = np.tri(data.shape[0], k=-1)
    data = np.nan_to_num(data)  # Convert NaNs to 0 for annotation
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text_color = "black" if data[i, j] < 0.5 else "white"  # Adjust text color based on background
            ax.text(
                j + 0.5,  # Adjust text position to center
                i + 0.5,  # Adjust text position to center
                format(data[i, j], fmt),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
                rotation=45,  # Rotate the text by 45 degrees
            )


def heatmap():
    # Create a DataFrame for adjusted closing prices
    adj_close_df = pd.concat([df.set_index("Date")["Adj Close"].rename(symbol) for symbol, df in zip(stock_symbols, stock_data)], axis=1)

    # Calculate the correlation matrix
    corr_matrix = adj_close_df.corr()

    # Create a correlation heatmap with annotated values
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5, square=True)
    annotate_heatmap(corr_matrix.values, ax=ax)
    plt.title("Stock Price Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig("correlation_heatmap.png")
    plt.show()

def moving_average():
    ma_day = [10,20,50]
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




def daily_returns():
    for i in range(19):
        stock_df = stocks[i]  # Access the DataFrame by index
        stock_df['Daily Return'] = stock_df['Adj Close'].pct_change()
        sns.histplot(stock_df['Daily Return'].dropna(), bins=50, color='blue', kde=True)  # You can include the KDE curve
        plt.xlabel("Density")
        plt.ylabel("Daily Return")
        plt.title(f"{stock_symbols[i]} Daily Returns")
        plt.savefig(f"{stock_symbols[i]}_daily_returns.png")
        plt.show()

daily_returns()