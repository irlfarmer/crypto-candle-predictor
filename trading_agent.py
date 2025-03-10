import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
import tensorflow as tf
import pickle
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import threading
import json

def candlestick_loss(y_true, y_pred):
    # Unpack predictions
    high_pred = y_pred[:, 0]
    low_pred = y_pred[:, 1]
    close_pred = y_pred[:, 2]
    
    # Basic MSE loss
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)
    
    # Penalty for high < low
    high_low_penalty = tf.maximum(0.0, low_pred - high_pred)
    
    # Penalty for close outside high-low range
    close_range_penalty = (
        tf.maximum(0.0, close_pred - high_pred) +
        tf.maximum(0.0, low_pred - close_pred)
    )
    
    # Combine losses
    total_loss = mse_loss + 10.0 * high_low_penalty + 10.0 * close_range_penalty
    return total_loss

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'api_key': '', 'api_secret': '', 'symbol': 'BTCUSDT', 'quantity': 0.001}

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
st.title("🤖 Automated Trading Bot")

# Sidebar configuration
st.sidebar.header("Configuration")
config = load_config()

api_key = st.sidebar.text_input("API Key", value=config['api_key'], type="password")
api_secret = st.sidebar.text_input("API Secret", value=config['api_secret'], type="password")
symbol = st.sidebar.text_input("Trading Pair", value=config['symbol'])
quantity = st.sidebar.number_input("Trading Quantity", value=config['quantity'], step=0.001)

if st.sidebar.button("Save Configuration"):
    config = {
        'api_key': api_key,
        'api_secret': api_secret,
        'symbol': symbol,
        'quantity': quantity
    }
    save_config(config)
    st.sidebar.success("Configuration saved!")

# Load the model and scalers
try:
    model = tf.keras.models.load_model('models/candle_predictor_model.keras',
                                     custom_objects={'candlestick_loss': candlestick_loss})
    with open('models/price_scaler.pkl', 'rb') as f:
        price_scaler = pickle.load(f)
    with open('models/volume_scaler.pkl', 'rb') as f:
        volume_scaler = pickle.load(f)
    with open('models/other_scaler.pkl', 'rb') as f:
        other_scaler = pickle.load(f)
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False

def prepare_data(df):
    # Calculate additional features
    df['price_change'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['low'] * 100
    df['volatility'] = df['price_change'].rolling(window=20).std()
    df['trend'] = df['close'].rolling(window=20).mean()
    df = df.dropna()
    
    # Scale features
    price_data = df[['open', 'high', 'low', 'close', 'trend']].values
    volume_data = df[['volume']].values
    other_data = df[['price_change', 'high_low_range', 'volatility']].values
    
    scaled_price = price_scaler.transform(price_data)
    scaled_volume = volume_scaler.transform(volume_data)
    scaled_other = other_scaler.transform(other_data)
    
    # Combine scaled data
    scaled_data = np.hstack((
        scaled_price[:, :4],  # open, high, low, close
        scaled_volume,
        scaled_other,
        scaled_price[:, -1:]  # trend
    ))
    
    return scaled_data

def predict_next_candle(df, model, scalers):
    # Prepare data
    scaled_data = prepare_data(df)
    sequence = scaled_data[-100:].reshape(1, 100, scaled_data.shape[1])
    
    # Get prediction (percentage changes)
    prediction = model.predict(sequence, verbose=0)[0]
    
    # Convert percentage predictions to actual prices
    current_price = df['close'].iloc[-1]
    prediction_descaled = np.array([
        current_price * (1 + prediction[0]/100),  # High
        current_price * (1 + prediction[1]/100),  # Low
        current_price * (1 + prediction[2]/100)   # Close
    ])
    
    # Ensure predictions follow candlestick logic
    prediction_descaled = np.array([
        max(prediction_descaled[0], current_price),  # High should be at least current price
        min(prediction_descaled[1], current_price),  # Low should be at most current price
        max(min(prediction_descaled[2], prediction_descaled[0]), prediction_descaled[1])  # Close between high and low
    ])
    
    return prediction_descaled

def get_historical_klines(client, symbol, interval, limit):
    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=limit
    )
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'quote_asset_volume', 'number_of_trades',
                                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def execute_trade(client, symbol, side, quantity):
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        return order
    except Exception as e:
        st.error(f"Error executing trade: {str(e)}")
        return None

def trading_loop():
    while st.session_state.trading_active:
        try:
            client = Client(api_key, api_secret)
            df = get_historical_klines(client, symbol, Client.KLINE_INTERVAL_4HOUR, 101)
            
            if len(df) >= 100:
                prediction = predict_next_candle(df, model, [price_scaler, volume_scaler, other_scaler])
                current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                
                # Trading logic
                if prediction[2] > current_price:  # If predicted close is higher
                    # Open long position
                    order = execute_trade(client, symbol, SIDE_BUY, quantity)
                    if order:
                        entry_price = float(order['fills'][0]['price'])
                        st.session_state.trade_history.append({
                            'timestamp': datetime.now(),
                            'action': 'BUY',
                            'price': entry_price,
                            'target': prediction[0]  # Target is predicted high
                        })
                        
                        # Wait for target or candle close
                        start_time = time.time()
                        while time.time() - start_time < 4 * 3600:  # 4 hours
                            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                            if current_price >= prediction[0]:
                                # Close position at target
                                order = execute_trade(client, symbol, SIDE_SELL, quantity)
                                if order:
                                    st.session_state.trade_history.append({
                                        'timestamp': datetime.now(),
                                        'action': 'SELL',
                                        'price': float(order['fills'][0]['price']),
                                        'reason': 'Target reached'
                                    })
                                break
                            time.sleep(10)
                        
                        # If target not reached, close at candle end
                        if time.time() - start_time >= 4 * 3600:
                            order = execute_trade(client, symbol, SIDE_SELL, quantity)
                            if order:
                                st.session_state.trade_history.append({
                                    'timestamp': datetime.now(),
                                    'action': 'SELL',
                                    'price': float(order['fills'][0]['price']),
                                    'reason': 'Candle closed'
                                })
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            st.error(f"Error in trading loop: {str(e)}")
            time.sleep(60)

# Main dashboard
if model_loaded:
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.trading_active:
            if st.button("Start Trading"):
                st.session_state.trading_active = True
                thread = threading.Thread(target=trading_loop)
                thread.start()
                st.success("Trading bot started!")
        else:
            if st.button("Stop Trading"):
                st.session_state.trading_active = False
                st.warning("Trading bot stopping...")

    with col2:
        st.write("Bot Status:", "🟢 Active" if st.session_state.trading_active else "🔴 Inactive")

    # Display trade history
    if st.session_state.trade_history:
        st.subheader("Trade History")
        df_trades = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(df_trades)
        
        # Calculate and display statistics
        if len(df_trades) >= 2:
            trades = []
            for i in range(0, len(df_trades) - 1, 2):
                if i + 1 < len(df_trades):
                    entry = df_trades.iloc[i]
                    exit = df_trades.iloc[i + 1]
                    profit = (exit['price'] - entry['price']) / entry['price'] * 100
                    trades.append(profit)
            
            if trades:
                st.subheader("Trading Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trades", len(trades))
                with col2:
                    st.metric("Win Rate", f"{sum(1 for x in trades if x > 0) / len(trades):.2%}")
                with col3:
                    st.metric("Average Profit", f"{sum(trades) / len(trades):.2%}")
                
                # Plot profit history
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=np.cumsum(trades), name='Cumulative Profit %'))
                fig.update_layout(title='Cumulative Profit History', yaxis_title='Profit %')
                st.plotly_chart(fig)

else:
    st.warning("Please train the model first using train_ui.py") 