import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os

# Load the data
print("Loading data...")
df = pd.read_csv('BTCUSDT_4h_historical_data.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Add price change percentage features
df['price_change'] = df['close'].pct_change()
df['high_low_range'] = (df['high'] - df['low']) / df['low'] * 100
df['volatility'] = df['price_change'].rolling(window=20).std()
df['trend'] = df['close'].rolling(window=20).mean()
df = df.dropna()

# Select features for scaling
features = ['open', 'high', 'low', 'close', 'volume', 'price_change', 'high_low_range', 'volatility', 'trend']
target_features = ['high', 'low', 'close']

# Scale the features
price_scaler = MinMaxScaler()
volume_scaler = MinMaxScaler()
other_scaler = MinMaxScaler()

# Scale price-related features together to maintain relationships
price_features = ['open', 'high', 'low', 'close', 'trend']
price_data = df[price_features].values
scaled_price = price_scaler.fit_transform(price_data)

# Scale volume separately
volume_data = df[['volume']].values
scaled_volume = volume_scaler.fit_transform(volume_data)

# Scale other features
other_features = ['price_change', 'high_low_range', 'volatility']
other_data = df[other_features].values
scaled_other = other_scaler.fit_transform(other_data)

# Combine all scaled data
scaled_data = np.hstack((
    scaled_price[:, :4],  # open, high, low, close
    scaled_volume,
    scaled_other,
    scaled_price[:, -1:]  # trend
))

# Create sequences with proper target normalization
def create_sequences(data, price_data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
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

# Parameters
sequence_length = 100
print("Creating sequences...")
X, y = create_sequences(scaled_data, df[['open', 'high', 'low', 'close']].values, sequence_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Custom loss function to enforce candlestick constraints
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

# Create and compile the model
print("Building model...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, scaled_data.shape[1])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='tanh')  # Output percentage changes (-1 to 1 range)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=candlestick_loss,
    metrics=['mae']
)

# Print model summary
print("\nModel Architecture:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('models/best_model.keras', save_best_only=True)
]

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss[0]:.6f}")
print(f"Test MAE: {test_loss[1]:.6f}")

# Save the model and scalers
print("\nSaving model and scalers...")
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/candle_predictor_model.keras')
with open('models/price_scaler.pkl', 'wb') as f:
    pickle.dump(price_scaler, f)
with open('models/volume_scaler.pkl', 'wb') as f:
    pickle.dump(volume_scaler, f)
with open('models/other_scaler.pkl', 'wb') as f:
    pickle.dump(other_scaler, f)

print("Training complete! Model and scalers have been saved.")

# Example prediction
print("\nMaking sample prediction...")
sample_sequence = X_test[0].reshape(1, sequence_length, -1)
prediction = model.predict(sample_sequence, verbose=0)[0]

# Get the current price for the sample
current_price = df['close'].iloc[train_size + sequence_length - 1]

# Convert percentage predictions back to actual prices
predicted_prices = np.array([
    current_price * (1 + prediction[0]/100),  # High
    current_price * (1 + prediction[1]/100),  # Low
    current_price * (1 + prediction[2]/100)   # Close
])

# Get actual values
actual_prices = df[['high', 'low', 'close']].iloc[train_size + sequence_length].values

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