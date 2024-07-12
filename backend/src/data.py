import os
from datetime import datetime, timedelta

import pandas as pd
import requests


def _fetch_and_save_stock(symbol, end_date, N_trading_days, api_key):
    # Calculate the start date
    start_date = end_date - timedelta(days=N_trading_days + 100)

    # Format dates for the API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Fetch daily data
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
        f"&outputsize=full"
    )

    response = requests.get(url)
    data = response.json()

    # Extract time series data
    time_series = data.get("Time Series (Daily)", {})

    # Create DataFrame
    df = pd.DataFrame.from_dict(time_series, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()  # Ensure ascending order

    # Filter data for the specified date range
    df = df.loc[start_date_str:end_date_str]

    # Convert columns to float
    df = df.astype(float)

    # Rename columns
    df = df.rename(
        columns={
            "1. open": "open",
            "4. close": "close",
            "2. high": "high",
            "3. low": "low",
            "5. volume": "volume",
        }
    )

    # Save the DataFrame to a CSV file
    today = datetime.now().strftime("%y%m%d")
    filename = f"data_stock_{today}.csv"
    df.to_csv(filename)
    print(f"Data saved to {filename}")

    return df


def fetch_stock_data(symbol, end_date, N_trading_days, api_key):
    end_date_str = end_date.strftime("%y%m%d")
    filename = f"data_stock_{end_date_str}.csv"

    if os.path.exists(filename):
        print(f"Loading data from {filename}")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        print(f"File {filename} not found. Fetching data from API.")
        df = _fetch_and_save_stock(symbol, end_date, N_trading_days, api_key)

    return df


def engineer_features(df, N_trading_days):
    # Calculate RSI (14-day period)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Calculate SMA (10-day period)
    df["SMA_10"] = df["close"].rolling(window=10).mean()

    # Calculate Bollinger Bands (20-day period, 2 standard deviations)
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["Upper_BB"] = df["SMA_20"] + (df["close"].rolling(window=20).std() * 2)
    df["Lower_BB"] = df["SMA_20"] - (df["close"].rolling(window=20).std() * 2)

    # Trim the DataFrame to the last N_trading_days
    df = df.iloc[-N_trading_days:]

    return df


if __name__ == "__main__":
    ALPHA_VANTAGE_API_KEY = "7H8XHMRSGISFBKK7"
    STOCK_SYMBOL = "AAPL"
    N_TRADING_DAYS = 30

    # Fetch raw stock data
    df_raw = fetch_stock_data(
        STOCK_SYMBOL, datetime.now(), N_TRADING_DAYS, ALPHA_VANTAGE_API_KEY
    )
    print("\nRAW FEATURES")
    print(df_raw.head())

    # Engineer features
    df_engineered = engineer_features(df_raw, N_TRADING_DAYS)
    print("\nENGINEER FEATURES")
    print(df_engineered.head())

    # Select only the required columns and trim to the desired date range
    df_result = df_engineered[
        ["open", "close", "RSI", "SMA_10", "SMA_20", "Upper_BB", "Lower_BB"]
    ]
    df_result = df_result.iloc[-N_TRADING_DAYS:]
    print("\nTRIMMED FEATURES")
    print(df_result.head())
