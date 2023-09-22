import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 
from ta import add_all_ta_features
from ta.utils import dropna

#Stock Data Frames
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


stocks = [aapl, amzn, atvi, dbx, ea, goog, meta, nflx, ntdoy, para, pins, rblx, sono, sony, spot, tsco, tsla, vod, zm]
stock_symbols = ["AAPL", "AMZN", "ATVI", "DBX", "EA", "GOOG", "META", "NFLX", "NTDOY", "PARA", "PINS", "RBLX", "SONO", "SONY", "SPOT", "TSCO.L", "TSLA", "VOD.L", "ZM"]

stock_data = []
for symbol in stock_symbols:
    file_path = os.path.join("data", f"{symbol}.csv")
    stock_df = pd.read_csv(file_path)
    stock_data.append(stock_df)

def plot_stock_and_ftse(stock_symbols, ftse_data):
    # Create a color palette for blue and green shades
    colors = sns.color_palette("coolwarm", n_colors=len(stock_symbols))
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual stock data
    for i, symbol in enumerate(stock_symbols):
        stock_df = stock_data[i]
        ax.plot(stock_df['Date'], stock_df['Close'], label=symbol, color=colors[i])
    
    # Plot FTSE data in red
    ax.plot(ftse_data['Date'], ftse_data['Close'], label='FTSE', color='red')
    
    # Fill the area between the blue/green lines and the red line
    for i in range(len(stock_symbols)):
        stock_df = stock_data[i]
        ax.fill_between(stock_df['Date'], stock_df['Close'], ftse_data['Close'], where=(stock_df['Close'] > ftse_data['Close']), interpolate=True, color=colors[i], alpha=0.3)
    
    # Set labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock Prices vs. FTSE')
    ax.legend(loc='upper left')
    
    # Show the plot
    plt.show()

# Example usage:
# Plot AAPL, META, and SPOT along with FTSE data
plot_stock_and_ftse(["AAPL", "META", "SPOT"], ftse)


def risk_return_analysis(stock_data, stock_names):
    # Define a list of distinct colors
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'lime', 'teal', 'navy', 'olive', 'maroon', 'sienna', 'slateblue', 'indigo', 'gold']

    # Calculate daily returns for each stock
    returns = pd.DataFrame({name: df['Adj Close'].pct_change().dropna() for name, df in zip(stock_names, stock_data)})

    # Calculate mean returns and standard deviations
    mean_returns = returns.mean()
    std_returns = returns.std()

    # Create a scatter plot of risk vs. expected return
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(stock_names):
        x_offset = 20 if name not in ['SONO', 'EA'] else -20  # Adjust the x-axis offset for specific stocks
        plt.scatter(mean_returns[i], std_returns[i], s=100, c=colors[i])

        # Adjust the x-axis offset for labels to remove overlap
        label_x_offset = -40 if name in ['SONO', 'EA', 'META', 'AAPL'] else 20
        plt.annotate(
            name,
            xy=(mean_returns[i], std_returns[i]),
            xytext=(label_x_offset, 0),  # Adjust this value to control the x-axis offset for labels
            textcoords='offset points', ha='left', va='center', color=colors[i])

    plt.xlabel('Expected Return')
    plt.ylabel('Risk')

    plt.title("Risk vs. Expected Return")
    plt.show()
    plt.savefig('Risk_Plot.png')

# Call the risk_return_analysis function with your stock data and names
risk_return_analysis(stock_data, stock_symbols)


def annotate_heatmap(data, fmt=".2f", fontsize=10, ax=None, cmap="coolwarm"):
    if ax is None:
        ax = plt.gca()
    # Fill square with annotations
    mask = np.tri(data.shape[0], k=-1)
    data = np.nan_to_num(data)  # Convert NaNs to 0 for annotation
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text_color = "black" if data[i, j] < 0.5 else "white"  # Adjust text color based on background brightness
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

def hs_daily_returns():
    for i in range(19):
        stock_df = stocks[i]  # Access the DataFrame by index
        stock_df['Daily Return'] = stock_df['Adj Close'].pct_change()
        sns.histplot(stock_df['Daily Return'].dropna(), bins=50, color='blue', kde=True)  # You can include the KDE curve
        plt.ylabel("Density")
        plt.xlabel("Daily Return")
        plt.title(f"{stock_symbols[i]} Daily Returns")
        plt.savefig(f"{stock_symbols[i]}_daily_returns.png")
        plt.show()

