import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import datetime, timedelta
import os
import json
from collections import deque
import streamlit as st

class ImprovedTradingAgent:
    def __init__(self, model_path='models/improved_model_fold_1.keras', scalers_path='models/improved_scalers.pkl'):
        """Initialize the improved trading agent."""
        # Load model and scalers
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'enhanced_candlestick_loss': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_pred - y_true))
            }
        )
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Trading parameters
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.trailing_stop = None
        self.trailing_distance = 0.01  # 1% trailing stop distance
        
        # Performance tracking
        self.trades = []
        self.current_balance = 1000.0  # Starting with $1000
        self.equity_curve = []
        self.win_rate = 0
        self.profit_factor = 0
        
        # Prediction history tracking
        self.prediction_history = []
        self.prediction_accuracy = {
            'direction': 0.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0
        }
        
        # Risk management
        self.max_trades_per_day = 3
        self.daily_trades = deque(maxlen=self.max_trades_per_day)
        self.max_drawdown = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        
        # Technical indicators for confirmation
        self.min_volatility = 0.001
        self.min_volume_ratio = 0.8
        self.trend_confirmation_window = 10
        
        # Trading accuracy tracking
        self.trading_accuracy = {
            'trades': [],  # List of trade results with predictions
            'metrics': {
                'profitable_predictions': 0,
                'total_predictions': 0,
                'direction_correct': 0,
                'avg_profit_on_correct': 0.0,
                'avg_loss_on_incorrect': 0.0,
                'price_accuracy': {
                    'high': {'mape': [], 'rmse': []},
                    'low': {'mape': [], 'rmse': []},
                    'close': {'mape': [], 'rmse': []}
                }
            }
        }
    
    def add_features(self, df):
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
    
    def prepare_data_for_scaling(self, df):
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
    
    def create_sequence(self, data, seq_length=100):
        """Create a sequence for prediction."""
        return data[-seq_length:].reshape(1, seq_length, -1)
    
    def check_trading_conditions(self, df):
        """Check if trading conditions are met."""
        # Check number of daily trades
        today = pd.Timestamp.now().date()
        self.daily_trades = deque([t for t in self.daily_trades if t.date() == today], maxlen=self.max_trades_per_day)
        if len(self.daily_trades) >= self.max_trades_per_day:
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        
        # Check volatility
        if df['volatility_10'].iloc[-1] < self.min_volatility:
            return False
        
        # Check volume
        if df['volume_ma_ratio'].iloc[-1] < self.min_volume_ratio:
            return False
        
        return True
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk."""
        if stop_loss == 0 or entry_price == 0:
            return 0
        
        risk_amount = self.current_balance * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        position_size = risk_amount / stop_distance
        
        return position_size
    
    def update_trailing_stop(self, current_price):
        """Update trailing stop if needed."""
        if self.position is None or self.trailing_stop is None:
            return
        
        if self.position == 'long':
            new_stop = current_price * (1 - self.trailing_distance)
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        else:  # short position
            new_stop = current_price * (1 + self.trailing_distance)
            if new_stop < self.trailing_stop:
                self.trailing_stop = new_stop
    
    def update_prediction_accuracy(self, predicted_prices, actual_prices):
        """Update prediction accuracy metrics with improved calculations."""
        try:
            # Store prediction and actual prices
            prediction_record = {
                'timestamp': datetime.now(),
                'predicted': predicted_prices,
                'actual': actual_prices,
                'accuracy': {},
                'errors': {}
            }
            
            # Calculate direction accuracy with confidence level
            pred_change = ((predicted_prices['close'] - actual_prices['previous']) / actual_prices['previous']) * 100
            actual_change = ((actual_prices['close'] - actual_prices['previous']) / actual_prices['previous']) * 100
            
            # Direction accuracy with magnitude consideration
            direction_correct = (pred_change * actual_change) > 0  # Same sign means correct direction
            magnitude_error = abs(abs(pred_change) - abs(actual_change))
            direction_score = 100 if direction_correct else 0
            
            # Store direction metrics
            prediction_record['accuracy']['direction'] = direction_score
            prediction_record['errors']['direction'] = magnitude_error
            
            # Calculate price accuracies with detailed metrics
            for price_type in ['high', 'low', 'close']:
                if price_type in predicted_prices and price_type in actual_prices:
                    pred_price = float(predicted_prices[price_type])
                    actual_price = float(actual_prices[price_type])
                    
                    if actual_price != 0:
                        # Percentage error
                        error_pct = abs((pred_price - actual_price) / actual_price) * 100
                        
                        # Root Mean Squared Error (RMSE)
                        rmse = np.sqrt(np.square(pred_price - actual_price))
                        
                        # Mean Absolute Percentage Error (MAPE)
                        mape = error_pct
                        
                        # Weighted accuracy score (0-100)
                        base_accuracy = max(0, 100 - error_pct)
                        
                        # Adjust accuracy based on price volatility
                        volatility = abs(actual_prices['high'] - actual_prices['low']) / actual_prices['previous'] * 100
                        volatility_factor = min(1.5, max(0.5, 1 + (volatility - 2) / 10))
                        adjusted_accuracy = min(100, base_accuracy * volatility_factor)
                        
                        prediction_record['accuracy'][price_type] = adjusted_accuracy
                        prediction_record['errors'][price_type] = {
                            'pct_error': error_pct,
                            'rmse': rmse,
                            'mape': mape,
                            'volatility': volatility
                        }
                    else:
                        prediction_record['accuracy'][price_type] = 0
                        prediction_record['errors'][price_type] = {
                            'pct_error': 100,
                            'rmse': float('inf'),
                            'mape': float('inf'),
                            'volatility': 0
                        }
            
            # Store the prediction record
            self.prediction_history.append(prediction_record)
            
            # Update overall accuracy metrics using recent history
            if len(self.prediction_history) > 0:
                recent_history = self.prediction_history[-100:]  # Last 100 predictions
                
                self.prediction_accuracy = {
                    'direction': {
                        'score': np.mean([p['accuracy']['direction'] for p in recent_history]),
                        'error': np.mean([p['errors']['direction'] for p in recent_history])
                    }
                }
                
                for price_type in ['high', 'low', 'close']:
                    self.prediction_accuracy[price_type] = {
                        'accuracy': np.mean([p['accuracy'][price_type] for p in recent_history]),
                        'mape': np.mean([p['errors'][price_type]['mape'] for p in recent_history]),
                        'rmse': np.mean([p['errors'][price_type]['rmse'] for p in recent_history]),
                        'volatility': np.mean([p['errors'][price_type]['volatility'] for p in recent_history])
                    }
                
        except Exception as e:
            print(f"Error updating prediction accuracy: {str(e)}")
            
    def get_prediction_metrics(self):
        """Get current prediction accuracy metrics with detailed statistics."""
        metrics = {
            'accuracy': self.prediction_accuracy,
            'history': self.prediction_history[-100:] if self.prediction_history else [],
            'summary': {}
        }
        
        if self.prediction_history:
            recent_history = self.prediction_history[-100:]
            
            # Calculate summary statistics
            metrics['summary'] = {
                'total_predictions': len(recent_history),
                'direction_accuracy': {
                    'overall': self.prediction_accuracy['direction']['score'],
                    'recent': np.mean([p['accuracy']['direction'] for p in recent_history[-10:]])  # Last 10 predictions
                },
                'price_accuracy': {
                    price_type: {
                        'overall': self.prediction_accuracy[price_type]['accuracy'],
                        'recent': np.mean([p['accuracy'][price_type] for p in recent_history[-10:]]),
                        'mape': self.prediction_accuracy[price_type]['mape'],
                        'rmse': self.prediction_accuracy[price_type]['rmse']
                    }
                    for price_type in ['high', 'low', 'close']
                }
            }
        
    def update_trading_accuracy(self, trade_result):
        """Update accuracy metrics based on trade results."""
        if not trade_result:
            return
            
        self.trading_accuracy['trades'].append(trade_result)
        
        # Keep only last 100 trades
        if len(self.trading_accuracy['trades']) > 100:
            self.trading_accuracy['trades'] = self.trading_accuracy['trades'][-100:]
        
        # Update metrics
        metrics = self.trading_accuracy['metrics']
        metrics['total_predictions'] += 1
        
        # Check if prediction was profitable
        if trade_result['pnl'] > 0:
            metrics['profitable_predictions'] += 1
            metrics['avg_profit_on_correct'] = (
                (metrics['avg_profit_on_correct'] * (metrics['profitable_predictions'] - 1) +
                 trade_result['pnl']) / metrics['profitable_predictions']
            )
        else:
            metrics['avg_loss_on_incorrect'] = (
                (metrics['avg_loss_on_incorrect'] * (metrics['total_predictions'] - metrics['profitable_predictions']) +
                 trade_result['pnl']) / (metrics['total_predictions'] - metrics['profitable_predictions'] + 1)
            )
        
        # Update direction accuracy
        if ('predicted_direction' in trade_result and 'actual_direction' in trade_result and
            trade_result['predicted_direction'] == trade_result['actual_direction']):
            metrics['direction_correct'] += 1
        
        # Update price accuracy metrics
        for price_type in ['high', 'low', 'close']:
            if f'predicted_{price_type}' in trade_result and f'actual_{price_type}' in trade_result:
                pred_price = trade_result[f'predicted_{price_type}']
                actual_price = trade_result[f'actual_{price_type}']
                
                if actual_price != 0:
                    mape = abs((pred_price - actual_price) / actual_price) * 100
                    rmse = np.sqrt(np.square(pred_price - actual_price))
                    
                    metrics['price_accuracy'][price_type]['mape'].append(mape)
                    metrics['price_accuracy'][price_type]['rmse'].append(rmse)
                    
                    # Keep only last 100 values
                    metrics['price_accuracy'][price_type]['mape'] = metrics['price_accuracy'][price_type]['mape'][-100:]
                    metrics['price_accuracy'][price_type]['rmse'] = metrics['price_accuracy'][price_type]['rmse'][-100:]
    
    def get_trading_accuracy_metrics(self):
        """Get current trading accuracy metrics."""
        metrics = self.trading_accuracy['metrics']
        total_preds = max(1, metrics['total_predictions'])
        
        return {
            'prediction_accuracy': {
                'profitable': (metrics['profitable_predictions'] / total_preds) * 100,
                'direction': (metrics['direction_correct'] / total_preds) * 100
            },
            'average_returns': {
                'profit_on_correct': metrics['avg_profit_on_correct'],
                'loss_on_incorrect': metrics['avg_loss_on_incorrect']
            },
            'price_accuracy': {
                price_type: {
                    'mape': np.mean(data['mape']) if data['mape'] else 0,
                    'rmse': np.mean(data['rmse']) if data['rmse'] else 0
                }
                for price_type, data in metrics['price_accuracy'].items()
            },
            'recent_trades': self.trading_accuracy['trades'][-10:]  # Last 10 trades
        }
    
    def process_candle(self, candle_data):
        """Process a new candle and make trading decisions."""
        # Update data
        if 'window' not in candle_data:
            st.error("No window data provided")
            return None
            
        df = candle_data['window']
        df = self.add_features(df)
        data_groups = self.prepare_data_for_scaling(df)
        
        # Scale data
        scaled_data = np.hstack([
            self.scalers['price'].transform(data_groups['price']),
            self.scalers['volume'].transform(data_groups['volume']),
            self.scalers['other'].transform(data_groups['other'])
        ])
        
        # Create sequence and predict
        sequence = self.create_sequence(scaled_data)
        prediction = self.model.predict(sequence, verbose=0)[0]
        
        current_price = float(candle_data['close'])
        previous_price = float(candle_data['open'])
        
        predicted_prices = {
            'high': self.safe_price_calculation(current_price, prediction[0]),
            'low': self.safe_price_calculation(current_price, prediction[1]),
            'close': self.safe_price_calculation(current_price, prediction[2])
        }
        
        # Update prediction accuracy if we have actual prices
        if len(self.prediction_history) > 0:
            last_prediction = self.prediction_history[-1]
            if 'predicted' in last_prediction:
                actual_prices = {
                    'high': float(candle_data['high']),
                    'low': float(candle_data['low']),
                    'close': current_price,
                    'previous': previous_price
                }
                self.update_prediction_accuracy(last_prediction['predicted'], actual_prices)
        
        # Store current prediction
        self.prediction_history.append({
            'timestamp': candle_data['timestamp'],
            'predicted': predicted_prices,
            'actual': None  # Will be updated in next candle
        })
        
        # Update trailing stop if in position
        if self.position is not None:
            self.update_trailing_stop(current_price)
            
            # Check if trailing stop is hit
            if self.position == 'long' and current_price < self.trailing_stop:
                return self.close_position(current_price, 'trailing_stop')
            elif self.position == 'short' and current_price > self.trailing_stop:
                return self.close_position(current_price, 'trailing_stop')
        
        # Check if we should enter a new position
        if self.position is None and self.check_trading_conditions(df):
            # Calculate predicted movement
            predicted_move = (predicted_prices['close'] - current_price) / current_price
            
            if abs(predicted_move) > 0.005:  # Minimum 0.5% predicted move
                if predicted_move > 0:  # Bullish prediction
                    stop_loss = min(predicted_prices['low'], current_price * 0.99)  # Max 1% initial risk
                    take_profit = predicted_prices['high']
                    return self.open_position('long', current_price, stop_loss, take_profit)
                else:  # Bearish prediction
                    stop_loss = max(predicted_prices['high'], current_price * 1.01)  # Max 1% initial risk
                    take_profit = predicted_prices['low']
                    return self.open_position('short', current_price, stop_loss, take_profit)
        
        # Store prediction details for accuracy tracking
        prediction_details = {
            'timestamp': candle_data['timestamp'],
            'predicted_direction': 1 if predicted_prices['close'] > current_price else -1,
            'predicted_high': predicted_prices['high'],
            'predicted_low': predicted_prices['low'],
            'predicted_close': predicted_prices['close'],
            'actual_direction': None,  # Will be updated when position is closed
            'actual_high': None,
            'actual_low': None,
            'actual_close': None,
            'pnl': None
        }
        
        # Update accuracy metrics if we have a completed trade
        if self.position is not None:
            last_trade = self.trades[-1] if self.trades else None
            if last_trade and 'exit_price' in last_trade:
                prediction_details.update({
                    'actual_direction': 1 if last_trade['exit_price'] > last_trade['entry_price'] else -1,
                    'actual_high': float(candle_data['high']),
                    'actual_low': float(candle_data['low']),
                    'actual_close': float(candle_data['close']),
                    'pnl': last_trade['pnl']
                })
                self.update_trading_accuracy(prediction_details)
        
        return None
    
    def open_position(self, direction, entry_price, stop_loss, take_profit):
        """Open a new position."""
        if self.position is not None:
            return None
        
        self.position = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = stop_loss
        
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        trade = {
            'type': 'entry',
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'timestamp': datetime.now()
        }
        
        self.daily_trades.append(datetime.now())
        return trade
    
    def close_position(self, exit_price, reason='normal'):
        """Close the current position."""
        if self.position is None:
            return None
        
        pnl = 0
        if self.position == 'long':
            pnl = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            pnl = (self.entry_price - exit_price) / self.entry_price
        
        trade = {
            'type': 'exit',
            'direction': self.position,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            'timestamp': datetime.now()
        }
        
        # Update performance metrics
        self.trades.append(trade)
        self.current_balance *= (1 + pnl)
        self.equity_curve.append(self.current_balance)
        
        # Update win rate and consecutive losses
        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Calculate win rate
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        self.win_rate = wins / len(self.trades) if self.trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate max drawdown
        peak = max(self.equity_curve)
        drawdown = (peak - self.current_balance) / peak
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Reset position data
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        
        return trade
    
    def get_performance_metrics(self):
        """Get current performance metrics."""
        return {
            'current_balance': self.current_balance,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'total_trades': len(self.trades),
            'consecutive_losses': self.consecutive_losses
        }
    
    def save_state(self, filepath='trading_state.json'):
        """Save the current state of the trading agent."""
        state = {
            'current_balance': self.current_balance,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'prediction_accuracy': self.prediction_accuracy,
            'prediction_history': self.prediction_history[-100:] if self.prediction_history else [],
            'trading_accuracy': self.trading_accuracy
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, default=str)  # Use default=str to handle datetime objects
    
    def load_state(self, filepath='trading_state.json'):
        """Load a previously saved state."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_balance = state['current_balance']
            self.trades = state['trades']
            self.equity_curve = state['equity_curve']
            self.win_rate = state['win_rate']
            self.profit_factor = state['profit_factor']
            self.max_drawdown = state['max_drawdown']
            self.consecutive_losses = state['consecutive_losses']
            
            # Load prediction metrics
            if 'prediction_accuracy' in state:
                self.prediction_accuracy = state['prediction_accuracy']
            if 'prediction_history' in state:
                self.prediction_history = state['prediction_history']
            
            # Load trading accuracy metrics
            if 'trading_accuracy' in state:
                self.trading_accuracy = state['trading_accuracy']
    
    def safe_price_calculation(self, current_price, prediction_pct):
        """Safely calculate price predictions with percentage changes."""
        try:
            current_price = np.float64(current_price)
            prediction_pct = np.float64(prediction_pct)
            
            clipped_pct = np.clip(prediction_pct, -20.0, 20.0)
            price_change = (current_price * clipped_pct) / 100.0
            new_price = current_price + price_change
            
            if new_price <= 0:
                return float(current_price)
                
            return float(new_price)
        except (ValueError, OverflowError, TypeError) as e:
            print(f"Error in price calculation: {str(e)}")
            return float(current_price) 