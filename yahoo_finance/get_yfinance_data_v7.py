import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set stock symbol
# WMT. DLTR, FBLG, INFN, NINE, NVDA, RIG
stock_symbol = 'RIG'  # Replace with your stock symbol

def is_valid_stock(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info and 'shortName' in info:
            print(f"Found valid stock: {info['shortName']}")
            return True
    except Exception as e:
        print(f"Error checking stock symbol: {e}")
    return False

def get_earliest_date(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="max", interval="1d")
        if not data.empty:
            earliest_date = data.index.min()
            print(f"Earliest available date for {symbol}: {earliest_date.date()}")
            return earliest_date
        else:
            print(f"No historical data found for {symbol}.")
    except Exception as e:
        print(f"Error fetching earliest date: {e}")
    return None

def download_stock_data(symbol, start_date, end_date, intervals=['1d', '1wk', '1mo']):
    for interval in intervals:
        print(f"Trying to download data with interval: {interval} from {start_date.date()} to {end_date.date()}")
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval, group_by='ticker')
        if not data.empty:
            return data, interval
        print(f"Data not found with interval: {interval}, trying next option...")
    return None, None

# Step 1: Validate the stock symbol
if is_valid_stock(stock_symbol):
    # Step 2: Get the earliest available date for the stock
    earliest_date = get_earliest_date(stock_symbol)

    if earliest_date is not None:
        # Step 3: Attempt to download stock data from the earliest date to today
        end_date = datetime.now()
        data, used_interval = download_stock_data(stock_symbol, earliest_date, end_date)

        if data is not None and not data.empty:
            # Step 4: Inspect the columns
            print("Columns in data DataFrame:")
            print(data.columns)
            print("Is data.columns a MultiIndex?", isinstance(data.columns, pd.MultiIndex))

            # Flatten columns if they are MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                # Drop the first level (ticker symbol)
                data.columns = data.columns.droplevel(0)
                # Reset the columns' index name
                data.columns.name = None
                print("Dropped first level of MultiIndex columns and reset columns' index name.")
            else:
                # Reset columns index name if it's not None
                data.columns.name = None
                print("Reset columns index name to None.")

            # Verify columns after flattening
            print("Columns after flattening:")
            print(data.columns)

            # Reorder the columns
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

            # Reset index to make 'Date' a column
            data.reset_index(inplace=True)

            # Specify the order of the columns including 'Date'
            desired_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            data = data[desired_order]

            # Save to CSV without the index (since 'Date' is now a column)
            csv_file_path = f'{stock_symbol}_stock_data_full.csv'
            data.to_csv(csv_file_path, index=False)
            print(f"Data saved to {csv_file_path} using interval: {used_interval}")

            # Visualize the data
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['Close'], label='Closing Price')
            plt.title(f'{stock_symbol} Stock Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        else:
            print(f"Failed to download data for {stock_symbol}.")
    else:
        print(f"No available data for {stock_symbol}.")
else:
    print(f"Stock symbol '{stock_symbol}' is invalid or not available on Yahoo Finance.")
