import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import (
    SIDE_BUY,
    SIDE_SELL,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_LIMIT
)
import tensorflow as tf
import pickle
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import threading
import json

# Define futures-specific order types
ORDER_TYPE_TAKE_PROFIT_MARKET = 'TAKE_PROFIT_MARKET'

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
        return {
            'api_key': '',
            'api_secret': '',
            'symbol': 'BTCUSDT',
            'market_quantity': 0.001,
            'limit_quantity': 0.002,
            'is_perpetual': False,
            'leverage': 10,
            'margin_type': 'ISOLATED'
        }

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'active_orders' not in st.session_state:
    st.session_state.active_orders = []

st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
st.title("🤖 Automated Trading Bot")

# Sidebar configuration
st.sidebar.header("Configuration")
config = load_config()

api_key = st.sidebar.text_input("API Key", value=config['api_key'], type="password")
api_secret = st.sidebar.text_input("API Secret", value=config['api_secret'], type="password")
symbol = st.sidebar.text_input("Trading Pair", value=config['symbol'])
is_perpetual = st.sidebar.checkbox("Use Perpetual Futures", value=config.get('is_perpetual', False))

if is_perpetual:
    leverage = st.sidebar.select_slider("Leverage", options=[1, 2, 3, 5, 10, 20, 50, 75, 100, 125], value=config.get('leverage', 10))
    margin_type = st.sidebar.selectbox("Margin Type", ['ISOLATED', 'CROSSED'], index=0 if config.get('margin_type') == 'ISOLATED' else 1)
    st.sidebar.info("Using Perpetual Futures with {}x leverage and {} margin".format(leverage, margin_type))

market_quantity = st.sidebar.number_input("Market Order Quantity", value=config['market_quantity'], step=0.001)
limit_quantity = st.sidebar.number_input("Limit Order Quantity", value=config['limit_quantity'], step=0.001)

if st.sidebar.button("Save Configuration"):
    config = {
        'api_key': api_key,
        'api_secret': api_secret,
        'symbol': symbol,
        'market_quantity': market_quantity,
        'limit_quantity': limit_quantity,
        'is_perpetual': is_perpetual,
        'leverage': leverage if is_perpetual else 1,
        'margin_type': margin_type if is_perpetual else 'ISOLATED'
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

def setup_perpetual_trading(client, symbol, leverage, margin_type):
    """Setup perpetual futures trading with specified leverage and margin type."""
    try:
        # Change margin type
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
    except:
        pass  # Ignore if margin type is already set
    
    try:
        # Set leverage
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        st.error(f"Error setting leverage: {str(e)}")

def get_historical_klines(client, symbol, interval, limit):
    try:
        # Request one extra candle to account for potential unclosed candle
        if config['is_perpetual']:
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

def get_next_candle_time():
    """Get the timestamp of the next 4h candle."""
    now = datetime.now()
    hours_since_midnight = now.hour + now.minute/60
    current_4h_block = int(hours_since_midnight / 4)
    next_4h = (current_4h_block + 1) * 4
    next_candle = now.replace(hour=int(next_4h), minute=0, second=0, microsecond=0)
    if next_4h >= 24:
        next_candle = next_candle + timedelta(days=1)
        next_candle = next_candle.replace(hour=next_4h - 24)
    return next_candle

def cancel_all_orders(client, symbol):
    """Cancel all open orders for the symbol."""
    try:
        if config['is_perpetual']:
            client.futures_cancel_all_open_orders(symbol=symbol)
        else:
            client.cancel_all_orders(symbol=symbol)
        st.session_state.active_orders = []
    except Exception as e:
        st.error(f"Error canceling orders: {str(e)}")

def close_all_positions(client, symbol):
    """Close all open positions for the symbol."""
    try:
        if config['is_perpetual']:
            # Get current position
            position = float(client.futures_position_information(symbol=symbol)[0]['positionAmt'])
            if position != 0:
                # Close position with market order
                client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL if position > 0 else SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=abs(position)
                )
        else:
            # Spot trading position closure
            position = float(client.get_asset_balance(asset=symbol[:-4])['free'])
            if position > 0:
                client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=position
                )
    except Exception as e:
        st.error(f"Error closing positions: {str(e)}")

def execute_trade_strategy(client, symbol, prediction, current_price, market_qty, limit_qty):
    """Execute the dual-order trading strategy."""
    try:
        # Cancel existing orders and close positions
        cancel_all_orders(client, symbol)
        close_all_positions(client, symbol)
        
        # Setup perpetual trading if enabled
        if config['is_perpetual']:
            setup_perpetual_trading(client, symbol, config['leverage'], config['margin_type'])
            order_function = client.futures_create_order
        else:
            order_function = client.create_order
        
        if prediction[2] > current_price:  # Bullish prediction
            # Market buy order with take profit at predicted high
            market_order = order_function(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=market_qty
            )
            
            # Set take profit for market order
            tp_market = order_function(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT if not config['is_perpetual'] else ORDER_TYPE_TAKE_PROFIT_MARKET,
                timeInForce='GTC',
                quantity=market_qty,
                price=prediction[0],  # Predicted high
                stopPrice=prediction[0] if config['is_perpetual'] else None
            )
            
            # Limit buy order at predicted low with take profit
            limit_order = order_function(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_LIMIT,
                timeInForce='GTC',
                quantity=limit_qty,
                price=prediction[1]  # Predicted low
            )
            
            # Set take profit for limit order
            tp_limit = order_function(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT if not config['is_perpetual'] else ORDER_TYPE_TAKE_PROFIT_MARKET,
                timeInForce='GTC',
                quantity=limit_qty,
                price=prediction[0],  # Predicted high
                stopPrice=prediction[0] if config['is_perpetual'] else None
            )
            
        else:  # Bearish prediction
            # Market sell order with take profit at predicted low
            market_order = order_function(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=market_qty
            )
            
            # Set take profit for market order
            tp_market = order_function(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_LIMIT if not config['is_perpetual'] else ORDER_TYPE_TAKE_PROFIT_MARKET,
                timeInForce='GTC',
                quantity=market_qty,
                price=prediction[1],  # Predicted low
                stopPrice=prediction[1] if config['is_perpetual'] else None
            )
            
            # Limit sell order at predicted high with take profit
            limit_order = order_function(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce='GTC',
                quantity=limit_qty,
                price=prediction[0]  # Predicted high
            )
            
            # Set take profit for limit order
            tp_limit = order_function(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_LIMIT if not config['is_perpetual'] else ORDER_TYPE_TAKE_PROFIT_MARKET,
                timeInForce='GTC',
                quantity=limit_qty,
                price=prediction[1],  # Predicted low
                stopPrice=prediction[1] if config['is_perpetual'] else None
            )
        
        # Store orders
        st.session_state.active_orders.extend([
            market_order['orderId'],
            tp_market['orderId'],
            limit_order['orderId'],
            tp_limit['orderId']
        ])
        
        return True
    except Exception as e:
        st.error(f"Error executing trade strategy: {str(e)}")
        return False

def trading_loop():
    while st.session_state.trading_active:
        try:
            client = Client(api_key, api_secret)
            
            # Wait for next 4h candle
            next_candle = get_next_candle_time()
            while datetime.now() < next_candle and st.session_state.trading_active:
                time.sleep(10)
            
            if not st.session_state.trading_active:
                break
            
            # Get historical data
            df = get_historical_klines(client, symbol, Client.KLINE_INTERVAL_4HOUR, 101)
            
            if len(df) >= 100:
                # Make prediction
                prediction = predict_next_candle(df, model, [price_scaler, volume_scaler, other_scaler])
                current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                
                # Execute trading strategy
                success = execute_trade_strategy(
                    client, symbol, prediction, current_price,
                    market_quantity, limit_quantity
                )
                
                if success:
                    st.session_state.trade_history.append({
                        'timestamp': datetime.now(),
                        'action': 'STRATEGY_EXECUTED',
                        'prediction': 'BULLISH' if prediction[2] > current_price else 'BEARISH',
                        'current_price': current_price,
                        'pred_high': prediction[0],
                        'pred_low': prediction[1],
                        'pred_close': prediction[2]
                    })
            
            # Wait for next candle
            time.sleep(60)
            
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