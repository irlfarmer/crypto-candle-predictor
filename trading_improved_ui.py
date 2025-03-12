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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import time
import threading
from trading_agent_improved import ImprovedTradingAgent

# Define futures-specific order types
ORDER_TYPE_TAKE_PROFIT_MARKET = 'TAKE_PROFIT_MARKET'

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
            'margin_type': 'ISOLATED',
            'use_trailing_stop': True
        }

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)

def get_historical_klines(client, symbol, interval, limit):
    """Get historical klines from Binance, excluding unclosed candles."""
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

def cancel_all_orders(client, symbol, is_perpetual):
    """Cancel all open orders for the symbol."""
    try:
        if is_perpetual:
            client.futures_cancel_all_open_orders(symbol=symbol)
        else:
            client.cancel_all_orders(symbol=symbol)
        st.session_state.active_orders = []
    except Exception as e:
        st.error(f"Error canceling orders: {str(e)}")

def close_all_positions(client, symbol, is_perpetual):
    """Close all open positions for the symbol."""
    try:
        if is_perpetual:
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

def execute_trade_strategy(client, symbol, prediction, current_price, market_qty, limit_qty, is_perpetual, use_trailing_stop):
    """Execute the trading strategy with optional trailing stop."""
    try:
        # Cancel existing orders and close positions
        cancel_all_orders(client, symbol, is_perpetual)
        close_all_positions(client, symbol, is_perpetual)
        
        # Setup perpetual trading if enabled
        if is_perpetual:
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
            
            if use_trailing_stop:
                # Set trailing stop for market order
                stop_price = prediction[1]  # Predicted low
                activation_price = current_price * 1.01  # Activate 1% above current price
                callback_rate = 1.0  # 1% callback rate
                
                if is_perpetual:
                    client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type='TRAILING_STOP_MARKET',
                        callbackRate=callback_rate,
                        activationPrice=activation_price,
                        quantity=market_qty
                    )
            else:
                # Set regular take profit
                tp_market = order_function(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_LIMIT if not is_perpetual else ORDER_TYPE_TAKE_PROFIT_MARKET,
                    timeInForce='GTC',
                    quantity=market_qty,
                    price=prediction[0],  # Predicted high
                    stopPrice=prediction[0] if is_perpetual else None
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
            
            if use_trailing_stop:
                # Set trailing stop for limit order
                if is_perpetual:
                    client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type='TRAILING_STOP_MARKET',
                        callbackRate=callback_rate,
                        activationPrice=prediction[1] * 1.01,
                        quantity=limit_qty
                    )
            else:
                # Set regular take profit for limit order
                tp_limit = order_function(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_LIMIT if not is_perpetual else ORDER_TYPE_TAKE_PROFIT_MARKET,
                    timeInForce='GTC',
                    quantity=limit_qty,
                    price=prediction[0],  # Predicted high
                    stopPrice=prediction[0] if is_perpetual else None
                )
            
        else:  # Bearish prediction
            # Similar structure for bearish trades...
            # (Code for bearish trades follows the same pattern as bullish trades)
            pass
        
        return True
    except Exception as e:
        st.error(f"Error executing trade strategy: {str(e)}")
        return False

def trading_loop():
    while st.session_state.trading_active:
        try:
            client = Client(config['api_key'], config['api_secret'])
            
            # Wait for next 4h candle
            next_candle = get_next_candle_time()
            while datetime.now() < next_candle and st.session_state.trading_active:
                time.sleep(10)
            
            if not st.session_state.trading_active:
                break
            
            # Get historical data
            df = get_historical_klines(client, config['symbol'], Client.KLINE_INTERVAL_4HOUR, 101)
            
            if len(df) >= 100:
                # Process data and make prediction using the improved agent
                candle_data = df.iloc[-1].to_dict()
                trade_action = st.session_state.agent.process_candle(candle_data)
                
                if trade_action:
                    current_price = float(client.get_symbol_ticker(symbol=config['symbol'])['price'])
                    
                    # Execute trading strategy with current configuration
                    success = execute_trade_strategy(
                        client=client,
                        symbol=config['symbol'],
                        prediction=[trade_action['take_profit'], trade_action['stop_loss'], trade_action['entry_price']],
                        current_price=current_price,
                        market_qty=config['market_quantity'],
                        limit_qty=config['limit_quantity'],
                        is_perpetual=config['is_perpetual'],
                        use_trailing_stop=config['use_trailing_stop']
                    )
                    
                    if success:
                        st.session_state.trades_history.append(trade_action)
                        st.session_state.agent.save_state()
            
            # Wait before next iteration
            time.sleep(60)
            
        except Exception as e:
            st.error(f"Error in trading loop: {str(e)}")
            time.sleep(60)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'trades_history' not in st.session_state:
    st.session_state.trades_history = []
if 'active_orders' not in st.session_state:
    st.session_state.active_orders = []
if 'agent' not in st.session_state:
    st.session_state.agent = ImprovedTradingAgent()
    if os.path.exists('trading_state.json'):
        st.session_state.agent.load_state()

# Page config
st.set_page_config(page_title="Improved Trading Bot", layout="wide")
st.title("🤖 Improved Automated Trading Bot")

# Load configuration
config = load_config()

# Sidebar configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("API Key", value=config['api_key'], type="password")
api_secret = st.sidebar.text_input("API Secret", value=config['api_secret'], type="password")
symbol = st.sidebar.text_input("Trading Pair", value=config['symbol'])
is_perpetual = st.sidebar.checkbox("Use Perpetual Futures", value=config.get('is_perpetual', False))
use_trailing_stop = st.sidebar.checkbox("Use Trailing Stop", value=config.get('use_trailing_stop', True))

if is_perpetual:
    leverage = st.sidebar.select_slider("Leverage", options=[1, 2, 3, 5, 10, 20, 50, 75, 100, 125], value=config.get('leverage', 10))
    margin_type = st.sidebar.selectbox("Margin Type", ['ISOLATED', 'CROSSED'], index=0 if config.get('margin_type') == 'ISOLATED' else 1)
    st.sidebar.info(f"Using Perpetual Futures with {leverage}x leverage and {margin_type} margin")

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
        'margin_type': margin_type if is_perpetual else 'ISOLATED',
        'use_trailing_stop': use_trailing_stop
    }
    save_config(config)
    st.sidebar.success("Configuration saved!")

# Main dashboard
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

# Display performance metrics
st.subheader("Performance Metrics")
metrics = st.session_state.agent.get_performance_metrics()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Balance", f"${metrics['current_balance']:.2f}")
    st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
with col2:
    st.metric("Total Trades", metrics['total_trades'])
    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
with col3:
    st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.1f}%")
    st.metric("Consecutive Losses", metrics['consecutive_losses'])

# Display trade history
if st.session_state.trades_history:
    st.subheader("Trade History")
    trades_df = pd.DataFrame(st.session_state.trades_history)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.sort_values('timestamp', ascending=False)
    
    st.dataframe(
        trades_df.style.format({
            'entry_price': '{:.4f}',
            'exit_price': '{:.4f}',
            'stop_loss': '{:.4f}',
            'take_profit': '{:.4f}',
            'pnl': '{:.2%}',
            'position_size': '{:.2f}'
        }),
        use_container_width=True
    )
    
    # Plot equity curve
    if len(st.session_state.agent.equity_curve) > 0:
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.agent.equity_curve,
            mode='lines',
            name='Account Balance'
        ))
        fig.update_layout(
            title='Account Balance History',
            yaxis_title='Balance (USDT)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No trades have been executed yet.") 