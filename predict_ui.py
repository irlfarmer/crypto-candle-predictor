import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import pickle
import plotly.graph_objects as go
from binance.client import Client
from datetime import datetime, timedelta
import tensorflow as tf

st.set_page_config(page_title="Crypto Price Predictor", layout="wide")
st.title("🔮 Crypto Price Predictor")

# Load the model and scaler
@st.cache_resource
def load_model_and_scalers():
    try:
        model = keras.models.load_model('models/candle_predictor_model.keras', 
                                      custom_objects={'candlestick_loss': candlestick_loss})
        with open('models/price_scaler.pkl', 'rb') as f:
            price_scaler = pickle.load(f)
        with open('models/volume_scaler.pkl', 'rb') as f:
            volume_scaler = pickle.load(f)
        with open('models/other_scaler.pkl', 'rb') as f:
            other_scaler = pickle.load(f)
        return model, price_scaler, volume_scaler, other_scaler, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, False

def prepare_data(df):
    # Ensure we have enough data
    if len(df) < 100:
        raise ValueError("Need at least 100 candles for prediction")
        
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

# Load model and scalers
model, price_scaler, volume_scaler, other_scaler, model_loaded = load_model_and_scalers()

if not model_loaded:
    st.error("Error: Could not load the model. Please ensure the model is trained and saved in the 'models' directory.")
    st.stop()

# Sidebar for configuration
st.sidebar.header("Configuration")
use_binance = st.sidebar.checkbox("Use Binance Data", value=True)

if use_binance:
    symbol = st.sidebar.text_input("Trading Pair", value="BTCUSDT")
    
    try:
        client = Client("", "")  # Public API access for historical data
        
        # Get historical data
        def get_historical_klines(symbol, interval, limit):
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
        
        df = get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, 150)
        
    except Exception as e:
        st.error(f"Error fetching data from Binance: {str(e)}")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload your candlestick data (CSV)", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        st.warning("Please upload a CSV file or use Binance data")
        st.stop()

# Make prediction
if len(df) >= 100:
    try:
        # Prepare data
        scaled_data = prepare_data(df)
        
        # Ensure we have exactly 100 timesteps for the sequence
        if len(scaled_data) > 100:
            sequence = scaled_data[-100:].reshape(1, 100, scaled_data.shape[1])
        else:
            st.error("Not enough data points after preprocessing")
            st.stop()
            
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
        
        # Display current price and predictions
        current_price = df['close'].iloc[-1]
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")
        with col2:
            st.metric("Predicted High", f"${prediction_descaled[0]:,.2f}", 
                     f"{((prediction_descaled[0] - current_price) / current_price * 100):+.2f}%")
        with col3:
            st.metric("Predicted Low", f"${prediction_descaled[1]:,.2f}", 
                     f"{((prediction_descaled[1] - current_price) / current_price * 100):+.2f}%")
        with col4:
            st.metric("Predicted Close", f"${prediction_descaled[2]:,.2f}", 
                     f"{((prediction_descaled[2] - current_price) / current_price * 100):+.2f}%")
        
        # Plot historical data and prediction
        fig = go.Figure()
        
        # Plot historical candles
        fig.add_trace(go.Candlestick(
            x=df['timestamp'].tail(25),
            open=df['open'].tail(25),
            high=df['high'].tail(25),
            low=df['low'].tail(25),
            close=df['close'].tail(25),
            name="Historical"
        ))
        
        # Add predicted values
        next_timestamp = pd.to_datetime(df['timestamp'].iloc[-1]) + pd.Timedelta(hours=4)
        fig.add_trace(go.Candlestick(
            x=[next_timestamp],
            open=[current_price],
            high=[prediction_descaled[0]],
            low=[prediction_descaled[1]],
            close=[prediction_descaled[2]],
            name="Prediction",
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        fig.update_layout(
            title="Price History and Prediction",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading signals
        st.subheader("Trading Signals")
        
        if prediction_descaled[2] > current_price:
            st.success("🔼 Bullish Signal: Predicted close is higher than current price")
            st.write(f"Potential Profit Target: ${prediction_descaled[0]:,.2f} (+{((prediction_descaled[0] - current_price) / current_price * 100):,.2f}%)")
        else:
            st.error("🔽 Bearish Signal: Predicted close is lower than current price")
            st.write(f"Potential Drop to: ${prediction_descaled[1]:,.2f} ({((prediction_descaled[1] - current_price) / current_price * 100):,.2f}%)")
        
        # Last update time
        st.sidebar.write("Last Update:", df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'))
        
        if use_binance:
            if st.button("Refresh Data"):
                st.rerun()  # Updated from experimental_rerun to rerun
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
else:
    st.error("Not enough historical data. Need at least 100 candles for prediction.") 