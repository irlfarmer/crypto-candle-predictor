import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
    
    # Clip extreme values to reasonable ranges
    for col in df.columns:
        if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].clip(-1000, 1000)  # Clip to ±1000%
    
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

def create_sequences(data, price_data, seq_length):
    """Create sequences for training with enhanced target calculation."""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        # Input sequence
        seq = data[i:i + seq_length]
        
        # Get next candle's actual values for target
        next_candle = price_data[i + seq_length]
        current_close = price_data[i + seq_length - 1][3]  # Current close price
        
        # Calculate target values as percentages from current close
        target = np.array([
            (next_candle[1] - current_close) / current_close * 100,  # High as % from current close
            (next_candle[2] - current_close) / current_close * 100,  # Low as % from current close
            (next_candle[3] - current_close) / current_close * 100   # Close as % from current close
        ])
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def enhanced_candlestick_loss(y_true, y_pred):
    """Enhanced loss function with emphasis on trend direction."""
    # Unpack predictions
    high_pred = y_pred[:, 0]
    low_pred = y_pred[:, 1]
    close_pred = y_pred[:, 2]
    
    # Unpack true values
    high_true = y_true[:, 0]
    low_true = y_true[:, 1]
    close_true = y_true[:, 2]
    
    # Basic MSE loss
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)
    
    # Trend direction loss (penalize wrong direction more heavily)
    direction_loss = tf.abs(tf.sign(close_true) - tf.sign(close_pred))
    
    # Candlestick structure penalties
    high_low_penalty = tf.maximum(0.0, low_pred - high_pred)
    close_range_penalty = (
        tf.maximum(0.0, close_pred - high_pred) +
        tf.maximum(0.0, low_pred - close_pred)
    )
    
    # Combine losses with higher weight on direction
    total_loss = (
        mse_loss + 
        15.0 * direction_loss +  # Increased weight on direction
        10.0 * high_low_penalty + 
        10.0 * close_range_penalty
    )
    return total_loss

def build_improved_model(sequence_length, n_features):
    """Build improved model architecture."""
    inputs = tf.keras.Input(shape=(sequence_length, n_features))
    
    # First LSTM layer with larger units
    x = LSTM(256, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third LSTM layer
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Fourth LSTM layer
    x = LSTM(32)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense layers with skip connections
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Output layer
    outputs = Dense(3, activation='tanh')(x)  # Still predicting percentages
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Configure optimizer with float learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=enhanced_candlestick_loss,
        metrics=['mae']
    )
    
    return model

def time_series_cv(X, y, n_splits=5):
    """Create time series cross-validation splits."""
    splits = []
    split_size = len(X) // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = len(X) - (n_splits - i) * split_size
        test_end = train_end + split_size
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        
        splits.append((train_idx, test_idx))
    
    return splits

# Main execution
if __name__ == "__main__":
    # Enable memory growth for GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Set mixed precision policy
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    print("Loading data...")
    df = pd.read_csv('BTCUSDT_4h_historical_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    print(f"Loaded {len(df)} candles")
    
    print("\nAdding features...")
    df = add_features(df)
    print(f"Features added, shape: {df.shape}")
    
    print("\nPreparing data for scaling...")
    data_groups = prepare_data_for_scaling(df)
    for group, data in data_groups.items():
        print(f"{group} features shape: {data.shape}")
    
    # Initialize scalers
    scalers = {
        'price': MinMaxScaler(),
        'volume': MinMaxScaler(),
        'other': MinMaxScaler()
    }
    
    # Scale each group
    scaled_data = np.hstack([
        scalers['price'].fit_transform(data_groups['price']),
        scalers['volume'].fit_transform(data_groups['volume']),
        scalers['other'].fit_transform(data_groups['other'])
    ])
    print(f"Combined scaled data shape: {scaled_data.shape}")
    
    # Parameters
    sequence_length = 100
    n_features = scaled_data.shape[1]
    batch_size = 32  # Reduced batch size
    
    print("\nCreating sequences...")
    X, y = create_sequences(scaled_data, df[['open', 'high', 'low', 'close']].values, sequence_length)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Clear memory
    del scaled_data
    del df
    
    # Create directory for model checkpoints if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Perform time series cross-validation
    cv_splits = time_series_cv(X, y, n_splits=5)
    successful_folds = 0
    best_val_loss = float('inf')
    best_fold = None
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        print(f"\nTraining fold {fold}/5...")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
        
        # Build and compile model
        model = build_improved_model(100, X.shape[2])
        print("\nModel architecture:")
        model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                f'models/model_fold_{fold}_checkpoint.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("\nStarting training...")
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Track best performing fold
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_fold = fold
                # Save the best model
                model.save(f'models/improved_model_fold_{fold}.keras')
            
            successful_folds += 1
            
        except Exception as e:
            print(f"Error during training fold {fold}: {str(e)}")
            continue
    
    print("\nCross-validation complete!")
    if successful_folds > 0:
        print(f"Best performing fold: {best_fold} with validation loss: {best_val_loss:.5f}")
        print(f"Model saved as: models/improved_model_fold_{best_fold}.keras")
        
        # Save scalers
        with open('models/improved_scalers.pkl', 'wb') as f:
            pickle.dump(scalers, f)
        print("Scalers saved as: models/improved_scalers.pkl")
    else:
        print("No successful training folds completed.")
    
    if best_fold is not None:
        # Make a sample prediction
        print("\nMaking sample prediction...")
        sample_sequence = X[-1].reshape(1, sequence_length, -1)
        prediction = model.predict(sample_sequence, verbose=0)[0]
        
        # Load the last candle data for prediction comparison
        df_last = pd.read_csv('BTCUSDT_4h_historical_data.csv').iloc[-2:]
        current_price = df_last['close'].iloc[0]
        predicted_prices = np.array([
            current_price * (1 + prediction[0]/100),  # High
            current_price * (1 + prediction[1]/100),  # Low
            current_price * (1 + prediction[2]/100)   # Close
        ])
        
        actual_prices = df_last[['high', 'low', 'close']].iloc[1].values
        
        print("\nSample prediction (High, Low, Close):")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted: ${predicted_prices[0]:.2f}, ${predicted_prices[1]:.2f}, ${predicted_prices[2]:.2f}")
        print(f"Actual: ${actual_prices[0]:.2f}, ${actual_prices[1]:.2f}, ${actual_prices[2]:.2f}")
        
        # Calculate percentage accuracy
        accuracy = 100 - (np.abs(predicted_prices - actual_prices) / actual_prices * 100)
        print("\nPrediction accuracy:")
        print(f"High: {accuracy[0]:.2f}%")
        print(f"Low: {accuracy[1]:.2f}%")
        print(f"Close: {accuracy[2]:.2f}%") 