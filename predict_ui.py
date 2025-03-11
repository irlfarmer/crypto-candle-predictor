import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import pickle
import plotly.graph_objects as go
from binance.client import Client
from datetime import datetime, timedelta
import tensorflow as tf
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)

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
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate additional features with shorter windows to preserve more data
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['high_low_range'] = ((df['high'] - df['low']) / df['low'] * 100).fillna(0)
    df['volatility'] = df['price_change'].rolling(window=10, min_periods=1).std().fillna(0)
    df['trend'] = df['close'].rolling(window=10, min_periods=1).mean().bfill()
    
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

# Initialize session state for data storage
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'historical_predictions' not in st.session_state:
    st.session_state.historical_predictions = []
if 'last_symbol' not in st.session_state:
    st.session_state.last_symbol = None

# Sidebar for configuration
st.sidebar.header("Configuration")
use_binance = st.sidebar.checkbox("Use Binance Data", value=True)

if use_binance:
    symbol = st.sidebar.text_input("Trading Pair", value="BTCUSDT")
    is_perpetual = st.sidebar.checkbox("Use Perpetual Futures Data", value=False)
    
    # Only fetch new data if symbol changes or data is None
    if symbol != st.session_state.last_symbol or st.session_state.historical_data is None:
        try:
            client = Client("", "")  # Public API access for historical data
            
            # Get historical data - fetch 300 candles to have enough for historical predictions
            if is_perpetual:
                klines = client.futures_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_4HOUR,
                    limit=300
                )
            else:
                klines = client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_4HOUR,
                    limit=300
                )
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'close_time', 'quote_asset_volume', 'number_of_trades',
                                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Check if the last candle is closed
            current_time = pd.Timestamp.now()
            last_candle_time = df['timestamp'].iloc[-1]
            if current_time < last_candle_time + pd.Timedelta(hours=4):
                # Remove the last (unclosed) candle
                df = df.iloc[:-1]
            
            # Store in session state
            st.session_state.historical_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            st.session_state.last_symbol = symbol
            st.session_state.historical_predictions = []  # Clear predictions for new symbol
            
        except Exception as e:
            st.error(f"Error fetching data from Binance: {str(e)}")
            st.stop()
    
    df = st.session_state.historical_data
    
    # Add market type indicator
    st.sidebar.info(f"Using {'Perpetual Futures' if is_perpetual else 'Spot'} market data")

else:
    uploaded_file = st.sidebar.file_uploader("Upload your candlestick data (CSV)", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.session_state.historical_data = df
        st.session_state.historical_predictions = []
    else:
        st.warning("Please upload a CSV file or use Binance data")
        st.stop()

# Add refresh button in sidebar
if st.sidebar.button("Refresh Data"):
    st.session_state.historical_data = None  # This will trigger a new data fetch
    st.rerun()

# Make prediction
if len(df) >= 100:
    try:
        # Make predictions for historical data if predictions list is empty
        if not st.session_state.historical_predictions:
            # Start from the end of the data and work backwards
            for i in range(102):  # We want 102 predictions
                end_idx = len(df) - i - 1  # Start from the last candle and move backwards
                start_idx = end_idx - 100  # Get 100 candles before the end_idx
                
                if start_idx < 0:  # Break if we don't have enough historical data
                    break
                
                # Get the sequence of 100 candles
                historical_df = df.iloc[start_idx:end_idx]
                if len(historical_df) < 100:  # Skip if we don't have enough data
                    continue
                    
                historical_scaled_data = prepare_data(historical_df)
                if len(historical_scaled_data) < 100:  # Skip if we lost too much data after preparation
                    continue
                
                # Make prediction using these 100 candles
                sequence = historical_scaled_data[-100:].reshape(1, 100, historical_scaled_data.shape[1])
                historical_prediction = model.predict(sequence, verbose=0)[0]
                
                # Get the current price (last close in the sequence)
                historical_current_price = historical_df['close'].iloc[-1]
                
                # Convert percentage predictions to actual prices
                historical_prediction_descaled = np.array([
                    historical_current_price * (1 + historical_prediction[0]/100),  # High
                    historical_current_price * (1 + historical_prediction[1]/100),  # Low
                    historical_current_price * (1 + historical_prediction[2]/100)   # Close
                ])
                
                # Ensure predictions follow candlestick logic
                historical_prediction_descaled = np.array([
                    max(historical_prediction_descaled[0], historical_current_price),
                    min(historical_prediction_descaled[1], historical_current_price),
                    max(min(historical_prediction_descaled[2], historical_prediction_descaled[0]), 
                        historical_prediction_descaled[1])
                ])
                
                # Get the actual next candle's data
                pred_timestamp = historical_df['timestamp'].iloc[-1] + pd.Timedelta(hours=4)
                actual_candle = df[df['timestamp'] == pred_timestamp]
                
                # Store prediction and actual values
                prediction_data = {
                    'timestamp': pred_timestamp,
                    'current_price': historical_current_price,
                    'pred_high': historical_prediction_descaled[0],
                    'pred_low': historical_prediction_descaled[1],
                    'pred_close': historical_prediction_descaled[2],
                    'actual_high': actual_candle['high'].iloc[0] if not actual_candle.empty else None,
                    'actual_low': actual_candle['low'].iloc[0] if not actual_candle.empty else None,
                    'actual_close': actual_candle['close'].iloc[0] if not actual_candle.empty else None,
                    'actual_open': actual_candle['open'].iloc[0] if not actual_candle.empty else None
                }
                
                # Insert at the beginning to maintain chronological order
                st.session_state.historical_predictions.insert(0, prediction_data)

        # Make current prediction (for the next candle)
        current_data = df.iloc[-100:]  # Get last 100 candles
        current_scaled_data = prepare_data(current_data)
        if len(current_scaled_data) == 100:  # Only predict if we have exactly 100 points
            sequence = current_scaled_data.reshape(1, 100, current_scaled_data.shape[1])
            prediction = model.predict(sequence, verbose=0)[0]
            
            # Convert percentage predictions to actual prices
            current_price = current_data['close'].iloc[-1]
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
            
            # Add current prediction to historical predictions
            next_timestamp = df['timestamp'].iloc[-1] + pd.Timedelta(hours=4)
            st.session_state.historical_predictions.append({
                'timestamp': next_timestamp,
                'current_price': current_price,
                'pred_high': prediction_descaled[0],
                'pred_low': prediction_descaled[1],
                'pred_close': prediction_descaled[2],
                'actual_high': None,
                'actual_low': None,
                'actual_close': None,
                'actual_open': None
            })
        else:
            st.error("Not enough data points after preprocessing for current prediction")
            st.stop()
        
        # Update historical predictions with actual values
        for pred in st.session_state.historical_predictions:
            if pred['actual_close'] is None:  # Only update if not already updated
                actual_candle = df[df['timestamp'] == pred['timestamp']]
                if not actual_candle.empty:
                    pred['actual_high'] = actual_candle['high'].iloc[0]
                    pred['actual_low'] = actual_candle['low'].iloc[0]
                    pred['actual_close'] = actual_candle['close'].iloc[0]
                    pred['actual_open'] = actual_candle['open'].iloc[0]
        
        # Keep only last 102 predictions
        st.session_state.historical_predictions = st.session_state.historical_predictions[-102:]
        
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
        
        # Plot current prediction chart
        st.subheader("Current Prediction")
        fig = go.Figure()
        
        # Plot historical candles (last 102 candles)
        fig.add_trace(go.Candlestick(
            x=df['timestamp'].tail(102),
            open=df['open'].tail(102),
            high=df['high'].tail(102),
            low=df['low'].tail(102),
            close=df['close'].tail(102),
            name="Historical"
        ))
        
        # Add predicted values
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
            title="Price History and Current Prediction (Last 102 Candles)",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot historical predictions chart
        st.subheader("Historical Predictions")
        if st.session_state.historical_predictions:
            hist_pred_df = pd.DataFrame(st.session_state.historical_predictions)
            
            # Add previous close columns for both actual and predicted
            hist_pred_df['prev_pred_close'] = hist_pred_df['pred_close'].shift(1)
            hist_pred_df['prev_actual_close'] = hist_pred_df['actual_close'].shift(1)
            
            fig2 = go.Figure()
            
            # Calculate colors based on comparing each close with previous close
            colors = []
            for i in range(len(hist_pred_df)):
                if i == 0:
                    colors.append('#26A69A')  # Default to green for first candle
                else:
                    prev_close = hist_pred_df['pred_close'].iloc[i-1]
                    colors.append('#26A69A' if hist_pred_df['pred_close'].iloc[i] > prev_close else '#EF5350')
            
            # Plot predicted candles with explicit colors and using previous close as open
            for i in range(len(hist_pred_df)):
                open_price = hist_pred_df['prev_pred_close'].iloc[i] if i > 0 else hist_pred_df['current_price'].iloc[i]
                fig2.add_trace(go.Candlestick(
                    x=[hist_pred_df['timestamp'].iloc[i]],
                    open=[open_price],
                    high=[hist_pred_df['pred_high'].iloc[i]],
                    low=[hist_pred_df['pred_low'].iloc[i]],
                    close=[hist_pred_df['pred_close'].iloc[i]],
                    name="Predicted",
                    increasing=dict(line=dict(color=colors[i]), fillcolor=colors[i]),
                    decreasing=dict(line=dict(color=colors[i]), fillcolor=colors[i]),
                    showlegend=False
                ))
            
            # Add legend items for trend directions
            fig2.add_trace(go.Scatter(
                x=[hist_pred_df['timestamp'].iloc[0]],
                y=[hist_pred_df['pred_high'].max()],
                mode='markers',
                marker=dict(color='#26A69A'),
                name='Close > Previous Close',
                showlegend=True
            ))
            fig2.add_trace(go.Scatter(
                x=[hist_pred_df['timestamp'].iloc[0]],
                y=[hist_pred_df['pred_high'].max()],
                mode='markers',
                marker=dict(color='#EF5350'),
                name='Close < Previous Close',
                showlegend=True
            ))
            
            fig2.update_layout(
                title="Historical Predictions (Last 102 Predictions)",
                xaxis_title="Time",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=600
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display prediction accuracy metrics
            actual_data = hist_pred_df[hist_pred_df['actual_close'].notna()]
            if not actual_data.empty:
                st.subheader("Prediction Accuracy Metrics")
                
                # Calculate trend direction accuracy (comparing with previous close)
                # We need to shift by 1 to compare each prediction with its corresponding actual
                actual_data = actual_data.copy()  # Create a copy to avoid SettingWithCopyWarning
                actual_data['prev_actual_close'] = df['close'].shift(1).loc[actual_data.index]
                actual_trend = actual_data['actual_close'] > actual_data['prev_actual_close']
                predicted_trend = actual_data['pred_close'] > actual_data['prev_actual_close']  # Compare with same previous close
                
                # Calculate accuracy
                trend_matches = actual_trend == predicted_trend
                trend_accuracy = np.sum(trend_matches) / len(trend_matches) if len(trend_matches) > 0 else 0
                
                # Calculate accuracy metrics
                high_mae = np.mean(np.abs(actual_data['actual_high'] - actual_data['pred_high']))
                low_mae = np.mean(np.abs(actual_data['actual_low'] - actual_data['pred_low']))
                close_mae = np.mean(np.abs(actual_data['actual_close'] - actual_data['pred_close']))
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("High Price MAE", f"${high_mae:,.2f}")
                with col2:
                    st.metric("Low Price MAE", f"${low_mae:,.2f}")
                with col3:
                    st.metric("Close Price MAE", f"${close_mae:,.2f}")
                with col4:
                    st.metric("Trend Accuracy", f"{trend_accuracy:.1%}")
                
                # Add a chart showing prediction errors over time
                error_df = pd.DataFrame({
                    'timestamp': actual_data['timestamp'],
                    'High Error': actual_data['actual_high'] - actual_data['pred_high'],
                    'Low Error': actual_data['actual_low'] - actual_data['pred_low'],
                    'Close Error': actual_data['actual_close'] - actual_data['pred_close']
                })
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=error_df['timestamp'], y=error_df['High Error'], 
                                        name='High Error', line=dict(color='green')))
                fig3.add_trace(go.Scatter(x=error_df['timestamp'], y=error_df['Low Error'], 
                                        name='Low Error', line=dict(color='red')))
                fig3.add_trace(go.Scatter(x=error_df['timestamp'], y=error_df['Close Error'], 
                                        name='Close Error', line=dict(color='blue')))
                
                fig3.update_layout(
                    title="Prediction Errors Over Time",
                    xaxis_title="Time",
                    yaxis_title="Error ($)",
                    height=400
                )
                
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No historical predictions available yet. Predictions will appear here as they are made.")
        
        # Trading signals
        st.subheader("Trading Signals")
        
        # Compare with previous close for trading signals
        last_pred = st.session_state.historical_predictions[-1]
        prev_close = st.session_state.historical_predictions[-2]['pred_close'] if len(st.session_state.historical_predictions) > 1 else current_price
        
        if last_pred['pred_close'] > prev_close:
            st.success("🔼 Bullish Signal: Predicted close is higher than previous close")
            st.write(f"Potential Profit Target: ${last_pred['pred_high']:,.2f} (+{((last_pred['pred_high'] - current_price) / current_price * 100):,.2f}%)")
        else:
            st.error("🔽 Bearish Signal: Predicted close is lower than previous close")
            st.write(f"Potential Drop to: ${last_pred['pred_low']:,.2f} ({((last_pred['pred_low'] - current_price) / current_price * 100):,.2f}%)")
        
        # Last update time
        st.sidebar.write("Last Update:", df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
else:
    st.error("Not enough historical data. Need at least 100 candles for prediction.") 