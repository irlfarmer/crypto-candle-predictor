import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from binance.client import Client
import json

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'api_key': '',
            'api_secret': '',
            'symbol': 'BTCUSDT',
            'is_perpetual': False
        }

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)

# Load the improved model and scalers
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'models/improved_model_fold_1.keras',
        custom_objects={
            'enhanced_candlestick_loss': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_pred - y_true))
        }
    )
    with open('models/improved_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    return model, scalers

def get_historical_klines(client, symbol, interval, limit, is_perpetual=False):
    """Get historical klines from Binance, excluding unclosed candles."""
    try:
        # Request one extra candle to account for potential unclosed candle
        if is_perpetual:
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit + 1
            )
        else:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit + 1
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
        
        # Verify we have enough closed candles
        if len(df) < limit:
            st.error(f"Not enough closed candles available. Need at least {limit} closed candles.")
            return pd.DataFrame()
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error fetching klines: {str(e)}")
        return pd.DataFrame()

def add_features(df):
    """Add technical indicators and features."""
    # Basic price changes and ranges
    df['price_change'] = df['close'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    df['high_low_range'] = np.where(
        df['low'] != 0,
        (df['high'] - df['low']) / df['low'] * 100,
        0
    )
    df['body_size'] = np.where(
        df['open'] != 0,
        (df['close'] - df['open']) / df['open'] * 100,
        0
    )
    
    # Candlestick shadows
    max_prices = df[['open', 'close']].max(axis=1)
    min_prices = df[['open', 'close']].min(axis=1)
    df['upper_shadow'] = np.where(
        max_prices != 0,
        (df['high'] - max_prices) / max_prices * 100,
        0
    )
    df['lower_shadow'] = np.where(
        min_prices != 0,
        (min_prices - df['low']) / min_prices * 100,
        0
    )
    
    # Multiple timeframe volatility
    for window in [10, 20, 30]:
        df[f'volatility_{window}'] = df['price_change'].rolling(window=window).std().fillna(0)
    
    # Multiple timeframe trends
    for window in [10, 20, 30]:
        df[f'trend_{window}'] = df['close'].rolling(window=window).mean().bfill()
        df[f'trend_strength_{window}'] = np.where(
            df[f'trend_{window}'] != 0,
            ((df['close'] - df[f'trend_{window}']) / df[f'trend_{window}'] * 100),
            0
        )
    
    # Volume analysis
    df['volume_change'] = df['volume'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean().bfill()
    df['volume_ma_ratio'] = np.where(
        df['volume_ma_10'] != 0,
        df['volume'] / df['volume_ma_10'],
        1
    )
    
    # Price momentum
    for window in [10, 20]:
        df[f'momentum_{window}'] = df['close'].pct_change(periods=window).fillna(0).replace([np.inf, -np.inf], 0)
    
    # Clip extreme values
    for col in df.columns:
        if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].clip(-1000, 1000)
    
    return df

def prepare_data_for_scaling(df):
    """Prepare data groups for scaling."""
    price_features = ['open', 'high', 'low', 'close'] + [f'trend_{w}' for w in [10, 20, 30]]
    volume_features = ['volume', 'volume_ma_10']
    other_features = [
        'price_change', 'high_low_range', 'body_size',
        'upper_shadow', 'lower_shadow',
        'volume_change', 'volume_ma_ratio'
    ] + [
        f'volatility_{w}' for w in [10, 20, 30]
    ] + [
        f'trend_strength_{w}' for w in [10, 20, 30]
    ] + [
        f'momentum_{w}' for w in [10, 20]
    ]
    
    return {
        'price': df[price_features].values,
        'volume': df[volume_features].values,
        'other': df[other_features].values
    }

def create_sequence(data, seq_length=100):
    """Create a sequence for prediction with safety checks."""
    if len(data) < seq_length:
        raise ValueError(f"Not enough data points. Need at least {seq_length} points.")
    return np.clip(data[-seq_length:], -10, 10).reshape(1, seq_length, -1)

def safe_percentage_calculation(current_price, prediction_pct):
    """Safely calculate price predictions with percentage changes."""
    try:
        # Convert inputs to float64 for better precision
        current_price = np.float64(current_price)
        prediction_pct = np.float64(prediction_pct)
        
        # Clip prediction percentages to reasonable ranges (-20% to +20%)
        clipped_pct = np.clip(prediction_pct, -20.0, 20.0)
        
        # Calculate the price change more safely
        price_change = (current_price * clipped_pct) / 100.0
        
        # Calculate new price and ensure it's positive
        new_price = current_price + price_change
        if new_price <= 0:
            return current_price
            
        return float(new_price)  # Convert back to float32
    except (ValueError, OverflowError, TypeError) as e:
        print(f"Error in price calculation: {str(e)}")
        return float(current_price)  # Return current price if calculation fails

def plot_predictions(df, predictions, current_price):
    """Create an interactive plot with predictions."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=('Price Action', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Historical'
        ),
        row=1, col=1
    )
    
    # Volume bars with safety check for colors
    colors = ['red' if c < o else 'green' for c, o in zip(df['close'].fillna(0), df['open'].fillna(0))]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Add prediction markers with safety checks
    last_timestamp = df['timestamp'].iloc[-1]
    next_timestamp = last_timestamp + pd.Timedelta(hours=4)
    
    # Predicted range with safety checks
    if not np.isnan(predictions['high']) and not np.isnan(predictions['low']):
        fig.add_trace(
            go.Scatter(
                x=[next_timestamp, next_timestamp],
                y=[predictions['low'], predictions['high']],
                mode='lines',
                line=dict(color='rgba(255,165,0,0.5)', width=20),
                name='Predicted Range'
            ),
            row=1, col=1
        )
    
    # Predicted close with safety check
    if not np.isnan(predictions['close']):
        fig.add_trace(
            go.Scatter(
                x=[next_timestamp],
                y=[predictions['close']],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=15,
                    color='orange',
                    line=dict(color='black', width=2)
                ),
                name='Predicted Close'
            ),
            row=1, col=1
        )
    
    # Current price line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Current: ${current_price:.2f}",
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Price Prediction with Technical Analysis',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Time',
        yaxis_title='Price (USDT)',
        xaxis2_title='Time',
        yaxis2_title='Volume',
        showlegend=True,
        height=800
    )
    
    # Remove rangeslider
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

class PredictionAccuracyTracker:
    def __init__(self):
        self.prediction_history = []
        self.accuracy_metrics = {
            'direction': {'correct': 0, 'total': 0},
            'price': {
                'high': {'mape': [], 'rmse': []},
                'low': {'mape': [], 'rmse': []},
                'close': {'mape': [], 'rmse': []}
            }
        }

    def calculate_accuracy(self, predictions, actual_data):
        """Calculate accuracy metrics for a single prediction."""
        metrics = {
            'timestamp': datetime.now(),
            'direction': {},
            'price': {}
        }

        # Direction accuracy - comparing predicted and actual moves relative to previous close
        pred_move = predictions['close'] - actual_data['previous']  # Predicted move from previous
        actual_move = actual_data['close'] - actual_data['previous']  # Actual move from previous
        
        # Direction is correct if both moves are in the same direction (both positive or both negative)
        direction_correct = (pred_move * actual_move) > 0
        
        self.accuracy_metrics['direction']['total'] += 1
        if direction_correct:
            self.accuracy_metrics['direction']['correct'] += 1

        metrics['direction'] = {
            'correct': direction_correct,
            'score': 100 if direction_correct else 0,
            'predicted_move': pred_move,
            'actual_move': actual_move
        }

        # Price accuracy metrics
        for price_type in ['high', 'low', 'close']:
            pred_price = predictions[price_type]
            actual_price = actual_data[price_type]
            
            # Calculate MAPE
            mape = abs((pred_price - actual_price) / actual_price) * 100
            # Calculate RMSE
            rmse = np.sqrt(np.square(pred_price - actual_price))
            
            self.accuracy_metrics['price'][price_type]['mape'].append(mape)
            self.accuracy_metrics['price'][price_type]['rmse'].append(rmse)
            
            metrics['price'][price_type] = {
                'mape': mape,
                'rmse': rmse,
                'accuracy': max(0, 100 - mape)  # Convert error to accuracy percentage
            }

        self.prediction_history.append(metrics)
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
            
        return metrics

    def get_summary(self):
        """Get summary of accuracy metrics."""
        if not self.prediction_history:
            return None

        summary = {
            'direction': {
                'accuracy': (self.accuracy_metrics['direction']['correct'] / 
                           max(1, self.accuracy_metrics['direction']['total']) * 100)
            },
            'price': {}
        }

        for price_type in ['high', 'low', 'close']:
            mapes = self.accuracy_metrics['price'][price_type]['mape'][-100:]
            rmses = self.accuracy_metrics['price'][price_type]['rmse'][-100:]
            
            summary['price'][price_type] = {
                'mape': np.mean(mapes) if mapes else 0,
                'rmse': np.mean(rmses) if rmses else 0,
                'accuracy': max(0, 100 - (np.mean(mapes) if mapes else 0))
            }

        return summary

def get_historical_data(client, symbol, is_perpetual=False):
    """Get sufficient historical data for accuracy calculation."""
    try:
        # Get 150 4-hour candles (600 hours of data)
        klines = get_historical_klines(
            client=client,
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_4HOUR,
            limit=150,
            is_perpetual=is_perpetual
        )
        
        if len(klines) < 100:
            raise ValueError("Insufficient historical data")
            
        return klines
    except Exception as e:
        raise Exception(f"Error fetching historical data: {str(e)}")

def main():
    st.set_page_config(page_title="Improved Crypto Price Predictor", layout="wide")
    
    # Initialize prediction accuracy tracker
    if 'accuracy_tracker' not in st.session_state:
        st.session_state.accuracy_tracker = PredictionAccuracyTracker()
    
    st.title("🚀 Enhanced Cryptocurrency Price Predictor")
    st.markdown("""
    This improved version uses advanced technical indicators and real-time market data
    to predict cryptocurrency price movements.
    """)
    
    # Load configuration
    config = load_config()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("API Key", value=config['api_key'], type="password")
    api_secret = st.sidebar.text_input("API Secret", value=config['api_secret'], type="password")
    symbol = st.sidebar.text_input("Trading Pair", value=config['symbol'])
    is_perpetual = st.sidebar.checkbox("Use Perpetual Futures", value=config.get('is_perpetual', False))
    
    if st.sidebar.button("Save Configuration"):
        config = {
            'api_key': api_key,
            'api_secret': api_secret,
            'symbol': symbol,
            'is_perpetual': is_perpetual
        }
        save_config(config)
        st.sidebar.success("Configuration saved!")
    
    try:
        # Load model and scalers
        model, scalers = load_model()
        st.success("✅ Model and scalers loaded successfully!")
        
        # Initialize Binance client
        client = Client(api_key, api_secret)
        st.success("✅ Connected to Binance API")
        
        # Get historical data once
        df = get_historical_data(client, symbol, is_perpetual)
        
        if len(df) < 100:
            st.error("❌ Not enough historical data")
            return
            
        # Process all historical data for accuracy calculation
        all_predictions = []
        for i in range(len(df) - 100):
            window = df[i:i+100].copy()
            window_processed = add_features(window)
            data_groups = prepare_data_for_scaling(window_processed)
            
            # Scale data
            scaled_data = np.hstack([
                scalers['price'].transform(data_groups['price']),
                scalers['volume'].transform(data_groups['volume']),
                scalers['other'].transform(data_groups['other'])
            ])
            
            # Predict
            sequence = create_sequence(scaled_data, 100)
            prediction = model.predict(sequence, verbose=0)[0]
            
            current_price = float(window_processed['close'].iloc[-1])
            predictions = {
                'high': safe_percentage_calculation(current_price, prediction[0]),
                'low': safe_percentage_calculation(current_price, prediction[1]),
                'close': safe_percentage_calculation(current_price, prediction[2])
            }
            
            # Get actual next candle data
            if i + 101 < len(df):
                next_candle = df.iloc[i + 100]
                actual_data = {
                    'high': float(next_candle['high']),
                    'low': float(next_candle['low']),
                    'close': float(next_candle['close']),
                    'previous': current_price
                }
                
                # Update accuracy metrics
                st.session_state.accuracy_tracker.calculate_accuracy(predictions, actual_data)
        
        # Display accuracy metrics
        accuracy_summary = st.session_state.accuracy_tracker.get_summary()
        if accuracy_summary:
            st.subheader("🎯 Prediction Accuracy (Last 100 Predictions)")
            
            # Direction accuracy
            st.metric(
                "Direction Accuracy",
                f"{accuracy_summary['direction']['accuracy']:.1f}%",
                help="Percentage of correct price movement direction predictions"
            )
            
            # Price accuracy metrics
            cols = st.columns(3)
            for idx, (price_type, metrics) in enumerate(accuracy_summary['price'].items()):
                with cols[idx]:
                    st.metric(
                        f"{price_type.capitalize()} Price Accuracy",
                        f"{metrics['accuracy']:.1f}%",
                        help=f"MAPE: {metrics['mape']:.2f}%, RMSE: {metrics['rmse']:.2f}"
                    )

            # Debug Table for Last 10 Predictions
            st.subheader("🔍 Debug: Last 10 Predictions")
            if st.session_state.accuracy_tracker.prediction_history:
                debug_data = []
                for pred in st.session_state.accuracy_tracker.prediction_history[-10:]:
                    debug_data.append({
                        'Timestamp': pred['timestamp'],
                        'Direction': '✅' if pred['direction']['correct'] else '❌',
                        'Pred Move': f"${pred['direction']['predicted_move']:.2f}",
                        'Actual Move': f"${pred['direction']['actual_move']:.2f}",
                        'Close Error': f"{pred['price']['close']['mape']:.2f}%",
                        'High Error': f"{pred['price']['high']['mape']:.2f}%",
                        'Low Error': f"{pred['price']['low']['mape']:.2f}%"
                    })
                st.dataframe(pd.DataFrame(debug_data))

            # Error Charts
            st.subheader("📊 Prediction Error Analysis")
            
            # Prepare error data
            error_data = {
                'close': {'mape': [], 'timestamp': []},
                'high': {'mape': [], 'timestamp': []},
                'low': {'mape': [], 'timestamp': []}
            }
            
            for pred in st.session_state.accuracy_tracker.prediction_history[-100:]:
                for price_type in ['close', 'high', 'low']:
                    error_data[price_type]['mape'].append(pred['price'][price_type]['mape'])
                    error_data[price_type]['timestamp'].append(pred['timestamp'])

            # Create separate error charts
            for price_type in ['close', 'high', 'low']:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=error_data[price_type]['timestamp'],
                    y=error_data[price_type]['mape'],
                    mode='lines+markers',
                    name=f'{price_type.capitalize()} MAPE',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f'{price_type.capitalize()} Price Prediction Error Over Time',
                    xaxis_title='Time',
                    yaxis_title='Mean Absolute Percentage Error (%)',
                    height=400,
                    showlegend=True
                )
                
                # Add moving average
                ma_window = 10
                ma = pd.Series(error_data[price_type]['mape']).rolling(window=ma_window).mean()
                fig.add_trace(go.Scatter(
                    x=error_data[price_type]['timestamp'],
                    y=ma,
                    mode='lines',
                    name=f'{ma_window}-Period MA',
                    line=dict(dash='dash', width=2)
                ))
                
                st.plotly_chart(fig, use_container_width=True)

            # Direction Accuracy Analysis
            st.subheader("🎯 Direction Prediction Analysis")
            direction_data = []
            for pred in st.session_state.accuracy_tracker.prediction_history[-100:]:
                direction_data.append({
                    'Timestamp': pred['timestamp'],
                    'Predicted Move': pred['direction']['predicted_move'],
                    'Actual Move': pred['direction']['actual_move'],
                    'Correct': pred['direction']['correct']
                })
            
            direction_df = pd.DataFrame(direction_data)
            
            # Create direction accuracy chart
            fig = go.Figure()
            
            # Correct predictions
            correct_preds = direction_df[direction_df['Correct']]
            incorrect_preds = direction_df[~direction_df['Correct']]
            
            fig.add_trace(go.Scatter(
                x=correct_preds['Timestamp'],
                y=correct_preds['Predicted Move'],
                mode='markers',
                name='Correct Predictions',
                marker=dict(color='green', size=8, symbol='circle')
            ))
            
            fig.add_trace(go.Scatter(
                x=incorrect_preds['Timestamp'],
                y=incorrect_preds['Predicted Move'],
                mode='markers',
                name='Incorrect Predictions',
                marker=dict(color='red', size=8, symbol='x')
            ))
            
            fig.update_layout(
                title='Direction Prediction Analysis',
                xaxis_title='Time',
                yaxis_title='Predicted Price Move ($)',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # Get and process data
        df = get_historical_klines(
            client=client,
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_4HOUR,
            limit=101,
            is_perpetual=is_perpetual
        )
        
        if len(df) < 100:
            st.error("❌ Not enough historical data (need at least 100 candles)")
            return
        
        # Get current price with safety check
        try:
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            st.error(f"❌ Error getting current price: {str(e)}")
            return
        
        # Process features
        df_processed = add_features(df.copy())
        data_groups = prepare_data_for_scaling(df_processed)
        
        # Scale data with safety checks
        try:
            scaled_data = np.hstack([
                scalers['price'].transform(data_groups['price']),
                scalers['volume'].transform(data_groups['volume']),
                scalers['other'].transform(data_groups['other'])
            ])
            
            # Create sequence and predict
            sequence = create_sequence(scaled_data, 100)
            prediction = model.predict(sequence, verbose=0)[0]
            
            # Calculate predicted prices with safety checks
            predictions = {
                'high': safe_percentage_calculation(current_price, prediction[0]),
                'low': safe_percentage_calculation(current_price, prediction[1]),
                'close': safe_percentage_calculation(current_price, prediction[2])
            }
            
            # Display predictions with safety checks
            col1, col2, col3 = st.columns(3)
            
            def safe_percentage_change(new_value, old_value):
                try:
                    if old_value == 0 or np.isnan(old_value) or np.isnan(new_value):
                        return "0.00%"
                    pct_change = ((float(new_value) - float(old_value)) / float(old_value)) * 100
                    return f"{np.clip(pct_change, -100, 100):.2f}%"
                except (ValueError, OverflowError, TypeError, ZeroDivisionError):
                    return "0.00%"
            
            with col1:
                st.metric(
                    "Predicted High",
                    f"${predictions['high']:.2f}" if not np.isnan(predictions['high']) else "N/A",
                    safe_percentage_change(predictions['high'], current_price)
                )
            
            with col2:
                st.metric(
                    "Predicted Low",
                    f"${predictions['low']:.2f}" if not np.isnan(predictions['low']) else "N/A",
                    safe_percentage_change(predictions['low'], current_price)
                )
            
            with col3:
                st.metric(
                    "Predicted Close",
                    f"${predictions['close']:.2f}" if not np.isnan(predictions['close']) else "N/A",
                    safe_percentage_change(predictions['close'], current_price)
                )
            
            # Plot
            fig = plot_predictions(df.tail(100), predictions, current_price)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics with safety checks
            st.subheader("📊 Additional Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    predicted_range = float(predictions['high'] - predictions['low'])
                    range_percentage = (predicted_range / current_price * 100) if current_price > 0 else 0
                    range_percentage = np.clip(range_percentage, 0, 100)
                    st.metric(
                        "Predicted Range",
                        f"${predicted_range:.2f}" if not np.isnan(predicted_range) else "N/A",
                        f"{range_percentage:.2f}%" if not np.isnan(range_percentage) else "N/A"
                    )
                except (ValueError, OverflowError, TypeError):
                    st.metric("Predicted Range", "N/A", "0.00%")
            
            with col2:
                try:
                    price_diff = float(predictions['close'] - current_price)
                    predicted_direction = "🔼 Up" if price_diff > 0 else "🔽 Down"
                    st.metric(
                        "Predicted Direction",
                        predicted_direction,
                        f"{abs(price_diff):.2f} USDT" if not np.isnan(price_diff) else "N/A"
                    )
                except (ValueError, OverflowError, TypeError):
                    st.metric("Predicted Direction", "N/A", "0.00 USDT")
            
            # Technical indicators
            st.subheader("📈 Current Technical Indicators")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Volatility (10-period)", f"{df_processed['volatility_10'].iloc[-1]:.4f}")
                st.metric("Trend Strength (10-period)", f"{df_processed['trend_strength_10'].iloc[-1]:.2f}%")
            
            with col2:
                st.metric("Volume Change", f"{df_processed['volume_change'].iloc[-1]:.2f}%")
                st.metric("Volume/MA Ratio", f"{df_processed['volume_ma_ratio'].iloc[-1]:.2f}")
            
            with col3:
                st.metric("Price Momentum (10-period)", f"{df_processed['momentum_10'].iloc[-1]:.2f}%")
                st.metric("Body Size", f"{df_processed['body_size'].iloc[-1]:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)
            
    except Exception as e:
        st.error(f"❌ Error during setup: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main() 