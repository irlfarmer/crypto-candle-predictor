import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
import json
from trading_agent_improved import ImprovedTradingAgent

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

def get_historical_klines(client, symbol, interval, limit, is_perpetual=False):
    """Get historical klines from Binance, excluding unclosed candles."""
    try:
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
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error fetching klines: {str(e)}")
        return pd.DataFrame()

def plot_trades(df, trades):
    """Create an interactive plot with trade markers."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=('Price Action & Trades', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['red' if c < o else 'green' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Add trade markers
    for trade in trades:
        if trade['type'] == 'entry':
            marker_color = 'green' if trade['direction'] == 'long' else 'red'
            marker_symbol = 'triangle-up' if trade['direction'] == 'long' else 'triangle-down'
            
            # Entry point
            fig.add_trace(
                go.Scatter(
                    x=[trade['timestamp']],
                    y=[trade['entry_price']],
                    mode='markers',
                    name=f"{trade['direction'].capitalize()} Entry",
                    marker=dict(
                        symbol=marker_symbol,
                        size=15,
                        color=marker_color
                    )
                ),
                row=1, col=1
            )
            
            # Stop loss and take profit lines
            fig.add_trace(
                go.Scatter(
                    x=[trade['timestamp']],
                    y=[trade['stop_loss']],
                    mode='markers',
                    name='Stop Loss',
                    marker=dict(
                        symbol='line-ns',
                        size=15,
                        color='red'
                    )
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[trade['timestamp']],
                    y=[trade['take_profit']],
                    mode='markers',
                    name='Take Profit',
                    marker=dict(
                        symbol='line-ns',
                        size=15,
                        color='green'
                    )
                ),
                row=1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title='Trading Activity',
        xaxis_title='Time',
        yaxis_title='Price (USDT)',
        xaxis2_title='Time',
        yaxis2_title='Volume',
        height=800
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def main():
    st.set_page_config(page_title="Improved Trading Agent", layout="wide")
    
    # Initialize trading agent in session state if not exists
    if 'trading_agent' not in st.session_state:
        st.session_state.trading_agent = ImprovedTradingAgent()
        st.session_state.trading_agent.load_state()  # Load previous state if exists
    
    st.title("🤖 Improved Trading Agent")
    st.markdown("""
    This trading agent uses advanced technical indicators and machine learning
    to make trading decisions in real-time.
    """)
    
    # Load configuration
    config = load_config()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("API Key", value=config['api_key'], type="password")
    api_secret = st.sidebar.text_input("API Secret", value=config['api_secret'], type="password")
    symbol = st.sidebar.text_input("Trading Pair", value=config['symbol'])
    is_perpetual = st.sidebar.checkbox("Use Perpetual Futures", value=config.get('is_perpetual', False))
    
    # Trading controls
    st.sidebar.header("Trading Controls")
    if st.sidebar.button("Reset Agent"):
        st.session_state.trading_agent = ImprovedTradingAgent()
        st.success("Trading agent reset successfully!")
    
    if st.sidebar.button("Save State"):
        st.session_state.trading_agent.save_state()
        st.success("Trading state saved successfully!")
    
    # Save configuration if changed
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
        # Initialize Binance client
        client = Client(api_key, api_secret)
        st.success("✅ Connected to Binance API")
        
        # Get historical data
        df = get_historical_klines(
            client=client,
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_4HOUR,
            limit=100,
            is_perpetual=is_perpetual
        )
        
        if df.empty:
            st.error("❌ No data available")
            return
        
        # Process window of candles
        if len(df) >= 100:
            # Create a window of the last 100 candles
            window = df.copy()
            # Convert to dictionary format
            candle_data = {
                'timestamp': window['timestamp'].iloc[-1],
                'open': float(window['open'].iloc[-1]),
                'high': float(window['high'].iloc[-1]),
                'low': float(window['low'].iloc[-1]),
                'close': float(window['close'].iloc[-1]),
                'volume': float(window['volume'].iloc[-1]),
                'window': window  # Pass the entire window for feature calculation
            }
            trade_action = st.session_state.trading_agent.process_candle(candle_data)
        else:
            st.warning("Need at least 100 candles for processing")
            return
        
        # Display current position and performance metrics
        col1, col2, col3 = st.columns(3)
        
        metrics = st.session_state.trading_agent.get_performance_metrics()
        with col1:
            st.metric("Current Balance", f"${metrics['current_balance']:.2f}")
            st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
        
        with col2:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.1f}%")
        
        with col3:
            st.metric("Total Trades", str(metrics['total_trades']))
            st.metric("Consecutive Losses", str(metrics['consecutive_losses']))
        
        # Display trading accuracy metrics
        st.subheader("Trading Accuracy")
        accuracy_metrics = st.session_state.trading_agent.get_trading_accuracy_metrics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Prediction Accuracy",
                f"{accuracy_metrics['prediction_accuracy']['profitable']:.1f}%",
                help="Percentage of predictions that resulted in profitable trades"
            )
            st.metric(
                "Direction Accuracy",
                f"{accuracy_metrics['prediction_accuracy']['direction']:.1f}%",
                help="Percentage of correct price movement predictions"
            )
        
        with col2:
            st.metric(
                "Avg Profit on Correct",
                f"${accuracy_metrics['average_returns']['profit_on_correct']:.2f}",
                help="Average profit when predictions are correct"
            )
            st.metric(
                "Avg Loss on Incorrect",
                f"${abs(accuracy_metrics['average_returns']['loss_on_incorrect']):.2f}",
                help="Average loss when predictions are incorrect"
            )
        
        # Display trade history
        st.subheader("Recent Trades")
        if st.session_state.trading_agent.trades:
            trades_df = pd.DataFrame(st.session_state.trading_agent.trades)
            st.dataframe(trades_df)
        
        # Plot trading activity
        fig = plot_trades(df, st.session_state.trading_agent.trades)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current position if any
        if st.session_state.trading_agent.position is not None:
            st.info(f"""
            Current Position: {st.session_state.trading_agent.position.upper()}
            Entry Price: ${st.session_state.trading_agent.entry_price:.2f}
            Stop Loss: ${st.session_state.trading_agent.stop_loss:.2f}
            Take Profit: ${st.session_state.trading_agent.take_profit:.2f}
            Trailing Stop: ${st.session_state.trading_agent.trailing_stop:.2f}
            """)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main() 