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
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Filter out unclosed candles
            current_time = pd.Timestamp.now()
            df = df[df['close_time'] < current_time].copy()
            
            # Ensure we have enough closed candles
            if len(df) < 100:
                st.error("Not enough closed candles available. Need at least 100 closed candles.")
                st.stop()
            
            # Store in session state
            st.session_state.historical_data = df[['timestamp', 'close_time', 'open', 'high', 'low', 'close', 'volume']]
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
    # Make predictions for historical data if predictions list is empty
    if not st.session_state.historical_predictions:
        # Start from the end of the data and work backwards
        for i in range(102):  # We want 102 predictions
                # Calculate which candle we're trying to predict
                target_idx = len(df) - i - 1
                if target_idx < 0:  # Break if we've gone too far back
                    break
                
                # Get the timestamp of the candle we're trying to predict
                target_candle_time = df['timestamp'].iloc[target_idx]
                
                # Get the 100 candles BEFORE this candle (these would have been available at prediction time)
                start_idx = target_idx - 100
                end_idx = target_idx - 1  # Exclude the target candle
                
                if start_idx < 0:  # Break if we don't have enough historical data
                    break
                
                # Get the sequence of 100 candles that were available at prediction time
                historical_df = df.iloc[start_idx:end_idx + 1]  # +1 because end is exclusive
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
                
                # Get the actual candle data
                actual_candle = df.iloc[target_idx] if target_idx < len(df) else pd.Series()
                
                # Store prediction and actual values
                prediction_data = {
                    'timestamp': target_candle_time,
                    'current_price': historical_current_price,
                    'pred_high': historical_prediction_descaled[0],
                    'pred_low': historical_prediction_descaled[1],
                    'pred_close': historical_prediction_descaled[2],
                    'actual_high': actual_candle['high'] if not actual_candle.empty else None,
                    'actual_low': actual_candle['low'] if not actual_candle.empty else None,
                    'actual_close': actual_candle['close'] if not actual_candle.empty else None,
                    'actual_open': actual_candle['open'] if not actual_candle.empty else None
                }
                
                # Insert at the beginning to maintain chronological order
                st.session_state.historical_predictions.insert(0, prediction_data)

        # Make current prediction
        current_time = pd.Timestamp.now()
        
        # Get exactly the last 101 closed candles
        closed_candles = df[df['close_time'] < current_time].tail(101)
        if len(closed_candles) < 101:
            st.error("Not enough closed candles for prediction")
            st.stop()
        
        # Make prediction for the next candle
        current_data = closed_candles.copy()
        current_scaled_data = prepare_data(current_data)
        
        if len(current_scaled_data) == 101:  # Verify we have exactly 101 points
            sequence = current_scaled_data.reshape(1, 101, current_scaled_data.shape[1])
            prediction = model.predict(sequence, verbose=0)[0]
            
            # Convert percentage predictions to actual prices
            current_price = current_data['close'].iloc[-1]
            prediction_descaled = np.array([
                current_price * (1 + prediction[0]/100),  # High
                current_price * (1 + prediction[1]/100),  # Low
                current_price * (1 + prediction[2]/100)   # Close
            ])
            
            # The next candle's timestamp (4 hours after last closed candle)
            next_timestamp = closed_candles['timestamp'].iloc[-1] + pd.Timedelta(hours=4)
            
            # Add current prediction to historical predictions
            current_prediction = {
                'timestamp': next_timestamp,
                'current_price': current_price,
                'pred_high': prediction_descaled[0],
                'pred_low': prediction_descaled[1],
                'pred_close': prediction_descaled[2],
                'actual_high': None,
                'actual_low': None,
                'actual_close': None,
                'actual_open': None
            }
            
            # Update historical predictions list
            if len(st.session_state.historical_predictions) >= 102:
                # Remove oldest prediction if we already have 102
                st.session_state.historical_predictions.pop(0)
            st.session_state.historical_predictions.append(current_prediction)
            
            # Display current price and predictions
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

            # Display latest closed candle information
            st.subheader("Latest Closed Candle Information")
            latest_candle = closed_candles.iloc[-1]
            latest_candle_time = latest_candle['timestamp']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Last Candle Time", latest_candle_time.strftime('%Y-%m-%d %H:%M'))
            with col2:
                st.metric("Open", f"${latest_candle['open']:,.2f}")
            with col3:
                st.metric("High", f"${latest_candle['high']:,.2f}", 
                         f"{((latest_candle['high'] - latest_candle['open']) / latest_candle['open'] * 100):+.2f}%")
            with col4:
                st.metric("Low", f"${latest_candle['low']:,.2f}", 
                         f"{((latest_candle['low'] - latest_candle['open']) / latest_candle['open'] * 100):+.2f}%")
            with col5:
                st.metric("Close", f"${latest_candle['close']:,.2f}", 
                         f"{((latest_candle['close'] - latest_candle['open']) / latest_candle['open'] * 100):+.2f}%")
            
            # Plot current prediction chart
            st.subheader("Current Prediction")
            fig = go.Figure()
            
            # Plot historical candles (last 101 closed candles)
            fig.add_trace(go.Candlestick(
                x=closed_candles['timestamp'],
                open=closed_candles['open'],
                high=closed_candles['high'],
                low=closed_candles['low'],
                close=closed_candles['close'],
                name="Historical"
            ))
            
            # Add predicted values with cyan color
            fig.add_trace(go.Candlestick(
                x=[next_timestamp],
                open=[current_price],  # Use last actual close as open
                high=[prediction_descaled[0]],
                low=[prediction_descaled[1]],
                close=[prediction_descaled[2]],
                name="Next Prediction",
                increasing_line_color='cyan',
                decreasing_line_color='cyan',
                increasing_fillcolor='cyan',
                decreasing_fillcolor='cyan'
            ))
            
            # Set exact range for 102 candles
            fig.update_layout(
                title="Price History and Current Prediction (Last 102 Candles)",
                xaxis_title="Time",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=600,
                xaxis_range=[closed_candles['timestamp'].iloc[0], next_timestamp]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot historical predictions chart
            st.subheader("Historical Predictions")
            if st.session_state.historical_predictions:
                hist_pred_df = pd.DataFrame(st.session_state.historical_predictions)
                
                fig2 = go.Figure()
                
                # Plot all predictions
                for i in range(len(hist_pred_df)):
                    # Get the open price (always use previous candle's predicted close)
                    if i == 0:
                        open_price = current_price  # Use current price only for the very first candle
                    else:
                        open_price = hist_pred_df['pred_close'].iloc[i-1]  # Use previous candle's predicted close
                    
                    # Use cyan color for the latest prediction, regular colors for others
                    if i == len(hist_pred_df) - 1:
                        fig2.add_trace(go.Candlestick(
                            x=[hist_pred_df['timestamp'].iloc[i]],
                            open=[open_price],
                            high=[hist_pred_df['pred_high'].iloc[i]],
                            low=[hist_pred_df['pred_low'].iloc[i]],
                            close=[hist_pred_df['pred_close'].iloc[i]],
                            name="Next Prediction",
                            increasing_line_color='cyan',
                            decreasing_line_color='cyan',
                            increasing_fillcolor='cyan',
                            decreasing_fillcolor='cyan'
                        ))
                    else:
                        # Color based on predicted trend (comparing predicted close with open)
                        color = '#26A69A' if hist_pred_df['pred_close'].iloc[i] > open_price else '#EF5350'
                        fig2.add_trace(go.Candlestick(
                            x=[hist_pred_df['timestamp'].iloc[i]],
                            open=[open_price],
                            high=[hist_pred_df['pred_high'].iloc[i]],
                            low=[hist_pred_df['pred_low'].iloc[i]],
                            close=[hist_pred_df['pred_close'].iloc[i]],
                            name="Historical Predictions",
                            increasing=dict(line=dict(color=color), fillcolor=color),
                            decreasing=dict(line=dict(color=color), fillcolor=color),
                            showlegend=False
                        ))
                
                # Use same time range as current prediction chart
                fig2.update_layout(
                    title="Historical Predictions (Last 102 Predictions)",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=600,
                    xaxis_range=[closed_candles['timestamp'].iloc[0], next_timestamp]
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Display prediction accuracy metrics
                actual_data = hist_pred_df[hist_pred_df['actual_close'].notna()]
                if not actual_data.empty:
                    st.subheader("Prediction Accuracy Metrics")
                    
                    # Calculate trend direction accuracy (comparing predicted direction vs actual direction)
                    actual_data = actual_data.copy()  # Create a copy to avoid SettingWithCopyWarning
                    
                    # For each prediction, get its open (which is previous prediction's close)
                    actual_data['pred_open'] = actual_data['pred_close'].shift(1)
                    # For the first prediction, use the current price as open
                    actual_data.loc[actual_data.index[0], 'pred_open'] = actual_data['current_price'].iloc[0]
                    
                    # Calculate predicted and actual trends
                    predicted_trend = actual_data['pred_close'] > actual_data['pred_open']
                    actual_trend = actual_data['actual_close'] > actual_data['actual_open']
                    
                    # Calculate accuracy
                    trend_matches = predicted_trend == actual_trend
                    trend_accuracy = np.sum(trend_matches) / len(trend_matches) if len(trend_matches) > 0 else 0
                    
                    # Add debug information
                    st.write("Trend Direction Debug:")
                    debug_df = pd.DataFrame({
                        'Timestamp': actual_data['timestamp'],
                        'Predicted Open': actual_data['pred_open'],
                        'Predicted Close': actual_data['pred_close'],
                        'Predicted Trend': predicted_trend,
                        'Actual Open': actual_data['actual_open'],
                        'Actual Close': actual_data['actual_close'],
                        'Actual Trend': actual_trend,
                        'Correct?': trend_matches
                    })
                    
                    # Calculate and display percentage of correct predictions
                    correct_predictions = debug_df['Correct?'].sum()
                    total_predictions = len(debug_df)
                    accuracy_percentage = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                    st.write(f"**Overall Prediction Accuracy**: {accuracy_percentage:.2f}% ({correct_predictions} correct out of {total_predictions} predictions)")
                    
                    st.dataframe(debug_df)
                    
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
                        st.metric("Trend Direction Accuracy", f"{trend_accuracy:.1%}")

                    # Add timestamp verification
                    st.write("### Timestamp Verification")
                    st.write("Checking for potential timing mismatches between predictions and actual data...")
                    
                    # Get a few sample rows to check timestamps
                    sample_data = actual_data.head()
                    st.write("Sample prediction timestamps vs actual candle timestamps:")
                    timestamp_df = pd.DataFrame({
                        'Prediction Time': sample_data['timestamp'],
                        'Actual Candle Time': [df['timestamp'].iloc[i] for i in sample_data.index],
                        'Time Difference': [pred - actual for pred, actual in 
                                         zip(sample_data['timestamp'], 
                                             [df['timestamp'].iloc[i] for i in sample_data.index])]
                    })
                    st.dataframe(timestamp_df)
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
            
            # Display last update time in sidebar
            try:
                st.sidebar.write("Last Update:", df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'))
            except Exception as e:
                st.error(f"Error displaying last update time: {str(e)}")
    else:
        st.error("Not enough historical data. Need at least 100 candles for prediction.")