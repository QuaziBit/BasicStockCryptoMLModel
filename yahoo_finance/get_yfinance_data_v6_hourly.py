import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set the stock symbol
stock_symbol = 'INFN'  # Replace with your stock symbol

def is_valid_stock(symbol):
    """
    Validate if the stock symbol exists on Yahoo Finance.
    
    Parameters:
    - symbol: Stock symbol.
    
    Returns:
    - True if valid, False otherwise.
    """
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
    """
    Get the earliest available date for the given stock symbol from Yahoo Finance.
    
    Parameters:
    - symbol: Stock symbol.
    
    Returns:
    - Earliest available date (datetime) if found, None otherwise.
    """
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

def download_hourly_data(symbol, earliest_date):
    """
    Download hourly stock data from Yahoo Finance for the last 2 years.
    
    Parameters:
    - symbol: Stock symbol.
    - earliest_date: The earliest available date for the stock.
    
    Returns:
    - DataFrame containing the hourly stock data.
    """
    # Set the end date as the current time and make it offset-naive
    end_date = datetime.now().replace(tzinfo=None)
    
    # Ensure earliest_date is also offset-naive
    earliest_date = earliest_date.replace(tzinfo=None)

    # Calculate the start_date
    start_date = max(earliest_date, end_date - timedelta(days=730))
    
    print(f"Downloading hourly data from {start_date} to {end_date}...")
    
    data = yf.download(symbol, start=start_date, end=end_date, interval='1h')
    
    if not data.empty:
        print(f"Data downloaded successfully from {start_date} to {end_date}.")
        return data
    else:
        print(f"No hourly data found for {symbol} in the given period.")
        return None

# Step 1: Validate the stock symbol
if is_valid_stock(stock_symbol):
    # Step 2: Get the earliest available date for the stock
    earliest_date = get_earliest_date(stock_symbol)

    if earliest_date is not None:
        # Step 3: Download hourly data
        data = download_hourly_data(stock_symbol, earliest_date)

        if data is not None and not data.empty:
            # Step 4: Save and visualize the data
            csv_file_path = f'{stock_symbol}_stock_data_hourly.csv'
            data.to_csv(csv_file_path)
            print(f"Data saved to {csv_file_path} with hourly intervals.")
            
            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'], label='Hourly Closing Price')
            plt.title(f'{stock_symbol} Stock Price (Hourly)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        else:
            print(f"Failed to download hourly data for {stock_symbol}.")
    else:
        print(f"No available data for {stock_symbol}.")
else:
    print(f"Stock symbol '{stock_symbol}' is invalid or not available on Yahoo Finance.")
    
