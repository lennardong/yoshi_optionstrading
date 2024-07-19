import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import requests

# TYPE SAFETY


@dataclass
class DataSchema:
    data_open: str = "data_open"
    data_high: str = "data_high"
    data_low: str = "data_low"
    data_close: str = "data_close"
    data_volume: str = "data_volume"
    data_delta: str = "data_delta"
    engineered_RSI: str = "engineered_RSI"
    engineered_SMA_10: str = "engineered_SMA_10"
    engineered_SMA_20: str = "engineered_SMA_20"
    engineered_Upper_BB: str = "engineered_Upper_BB"
    engineered_Lower_BB: str = "engineered_Lower_BB"
    engineered_MACD: str = "engineered_MACD"
    engineered_MACD_Signal: str = "engineered_MACD_Signal"
    engineered_ATR: str = "engineered_ATR"
    pred_delta: str = "pred_delta"
    monitor_rmse: str = "monitor_rmse"
    monitor_mape: str = "monitor_mape"
    monitor_ks_statistic: str = "monitor_ks_statistic"
    monitor_wasserstein_distance: str = "monitor_wasserstein_distance"
    monitor_data_drift = "monitor_data_drift"
    monitor_target_drift = "monitor_target_drift"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        return [getattr(cls, attr) for attr in cls.__annotations__]

    @classmethod
    def get_independent_vars(cls) -> List[str]:
        return [
            cls.data_open,
            cls.data_high,
            cls.data_low,
            cls.data_volume,
            cls.engineered_RSI,
            cls.engineered_SMA_10,
            cls.engineered_SMA_20,
            cls.engineered_Upper_BB,
            cls.engineered_Lower_BB,
            cls.engineered_MACD,
            cls.engineered_MACD_Signal,
            cls.engineered_ATR,
        ]

    @classmethod
    def get_dependant_var(cls) -> str:
        return cls.data_delta

    @classmethod
    def get_pred_var(cls) -> str:
        return cls.pred_delta

    @classmethod
    def get_monitor_vars(cls) -> List[str]:
        return [
            cls.monitor_rmse,
            cls.monitor_mape,
            cls.monitor_ks_statistic,
            cls.monitor_wasserstein_distance,
            cls.monitor_data_drift,
            cls.monitor_target_drift,
        ]


COLS = DataSchema()


def load_df(stock_symbol: str, end_date: str, N_trading_days: int, api_key: str):
    """
    Loads and preprocesses stock data for a given stock symbol, end date, and number of trading days.

    This function fetches the raw stock data for the specified stock symbol and date range, engineers additional features, and selects only the required columns. It also creates lagged features for the 'close' column.

    Args:
        stock_symbol (str): The stock symbol to fetch data for.
        end_date (str): The end date for the data range, in the format 'YYYY-MM-DD'.
        N_trading_days (int): The number of trading days to include in the data.
        api_key (str): The API key to use for fetching the stock data.

    Returns:
        pandas.DataFrame: The preprocessed stock data, with the required columns and lagged features.
    """
    # Fetch raw stock data
    df = _fetch_stock_data(stock_symbol, end_date, N_trading_days, api_key)
    logging.info(f"\n #### RAW DF ### \n{df.head()}")

    # Engineer features
    df = _engineer_features(df)
    logging.info(f"\n #### FEATURE ENGINEERING ### \n{df.head()}")

    # Select only the required columns and trim to the desired date range
    df = df[COLS.get_all_columns()]
    logging.info(f"\n #### CLEANED DF ### \n{df.head()}")

    df = _create_lagged_features(df)
    logging.info(f"\n #### LAGGED FEATURES, HEAD ### \n{df.head()}")
    logging.info(f"\n #### LAGGED FEATURES, TAIL ### \n{df.tail()}")

    return df


# UTILITY FUNCTIONS


def _fetch_stock_data(symbol, end_date, N_trading_days, api_key):
    END_DATE_STR = end_date.strftime("%y%m%d")
    FILENAME = f"data_stock_{END_DATE_STR}.csv"

    # HELPER FUNCTION
    def _fetch_and_save_stock(symbol, end_date, N_trading_days, api_key):
        # Calculate the start date
        start_date = end_date - timedelta(days=N_trading_days + 300)

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
        logging.info(f"DataFrame structure after creation:\n{df.dtypes} \n{df.head()}")

        # Rename columns
        df = df.rename(
            columns={
                "1. open": COLS.data_open,
                "4. close": COLS.data_close,
                "2. high": COLS.data_high,
                "3. low": COLS.data_low,
                "5. volume": COLS.data_volume,
            }
        )
        df[COLS.data_delta] = df[COLS.data_close] - df[COLS.data_open]

        # Iterate through dataclass columns and add them to the DataFrame
        # ensure it doesn't override if column exists (e.g. COLS.data_open)
        for col in DataSchema.get_all_columns():
            if not col in df.columns:
                df[col] = np.nan

        # Save the DataFrame to a CSV file
        df.to_csv(FILENAME)
        print(f"Data saved to {FILENAME}")

        return df

    if os.path.exists(FILENAME):
        print(f"Loading data from {FILENAME}")
        df = pd.read_csv(FILENAME, index_col=0, parse_dates=True)
    else:
        print(f"File {FILENAME} not found. Fetching data from API.")
        df = _fetch_and_save_stock(symbol, end_date, N_trading_days, api_key)

    return df


def _engineer_features(df):
    # Relative Strength Index (RSI)
    # RSI is a momentum indicator that measures the magnitude of recent price changes
    # to evaluate overbought or oversold conditions. Values range from 0 to 100.
    delta = df[COLS.data_close].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df[COLS.engineered_RSI] = 100 - (100 / (1 + rs))

    # Simple Moving Averages (SMA)
    # SMAs help identify trends by smoothing out price data
    df[COLS.engineered_SMA_10] = df[COLS.data_delta].rolling(window=10).mean()
    df[COLS.engineered_SMA_20] = df[COLS.data_delta].rolling(window=20).mean()

    # Bollinger Bands
    # Bollinger Bands consist of a middle band (20-day SMA) and two outer bands
    # They help identify volatility and potential overbought/oversold conditions
    std_dev = df[COLS.data_delta].rolling(window=20).std()
    df[COLS.engineered_Upper_BB] = df[COLS.engineered_SMA_20] + (std_dev * 2)
    df[COLS.engineered_Lower_BB] = df[COLS.engineered_SMA_20] - (std_dev * 2)

    # MACD (Moving Average Convergence Divergence)
    # MACD is a trend-following momentum indicator that shows the relationship
    # between two moving averages of a security's price
    ema_12 = df[COLS.data_delta].ewm(span=12, adjust=False).mean()
    ema_26 = df[COLS.data_delta].ewm(span=26, adjust=False).mean()
    df[COLS.engineered_MACD] = ema_12 - ema_26
    df[COLS.engineered_MACD_Signal] = (
        df[COLS.engineered_MACD].ewm(span=9, adjust=False).mean()
    )

    # Average True Range (ATR)
    # ATR is a market volatility indicator used in technical analysis
    high_low = df[COLS.data_high] - df[COLS.data_low]
    high_close = np.abs(df[COLS.data_high] - df[COLS.data_close].shift())
    low_close = np.abs(df[COLS.data_low] - df[COLS.data_close].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df[COLS.engineered_ATR] = true_range.rolling(14).mean()

    return df


def _create_lagged_features(df, max_lag=10):
    """
    Create lagged features from the target column.

    :param df: DataFrame containing the time series data
    :param target_col: Name of the target column
    :param max_lag: Maximum number of lagged days
    :return: DataFrame with lagged features
    """
    lagged_df = df.copy()
    for lag in range(1, max_lag + 1):
        lagged_df[f"{COLS.data_delta}_lag_{lag}"] = lagged_df[COLS.data_delta].shift(
            lag
        )

    lagged_df = lagged_df.dropna(how="all")
    return lagged_df


if __name__ == "__main__":
    ALPHA_VANTAGE_API_KEY = "7H8XHMRSGISFBKK7"
    STOCK_SYMBOL = "AAPL"
    N_TRADING_DAYS = 30

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    df = load_df(STOCK_SYMBOL, datetime.now(), N_TRADING_DAYS, ALPHA_VANTAGE_API_KEY)
