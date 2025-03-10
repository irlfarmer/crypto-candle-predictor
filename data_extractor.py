import requests
import pandas as pd
import time

def get_binance_full_history(symbol="BTCUSDT", interval="4h", save_csv=True):
    """
    Fetches all historical OHLCV data for a given symbol from Binance API in chunks.
    
    Parameters:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        interval (str): Candle timeframe (e.g., '4h').
        save_csv (bool): Whether to save the data to a CSV file.

    Returns:
        pd.DataFrame: Complete historical OHLCV dataset.
    """

    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000  # Binance allows max 1000 candles per request
    all_data = []
    
    # Fetch first timestamp (Binance listing date for BTC)
    first_timestamp = 1502928000000  # Approx. August 2017 (Unix timestamp in ms)
    
    # Start fetching data
    start_time = first_timestamp
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        if not data:
            print("No more data available.")
            break

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        all_data.append(df)

        print(f"Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        # Update start_time for next request (next candle's timestamp)
        start_time = int(df["timestamp"].iloc[-1].timestamp() * 1000) + 1

        # Sleep to avoid hitting API rate limits
        time.sleep(15)

    # Concatenate all dataframes
    full_df = pd.concat(all_data, ignore_index=True)

    # Save to CSV
    if save_csv:
        full_df.to_csv(f"{symbol}_{interval}_historical_data.csv", index=False)
        print(f"Data saved as {symbol}_{interval}_historical_data.csv")

    return full_df

# Fetch full BTC/USDT 4H data and save it
btc_data = get_binance_full_history()
