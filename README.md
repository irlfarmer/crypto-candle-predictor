# Crypto Candle Predictor

A machine learning-based cryptocurrency trading system that predicts future candlestick patterns and automates trading decisions.

## Features

- **Candlestick Pattern Prediction**: Uses LSTM neural networks to predict high, low, and close prices
- **Real-time Trading**: Automated trading bot with Binance integration
- **Interactive UI**: Streamlit-based interface for predictions and trading
- **Technical Indicators**: Incorporates price change, volatility, and trend analysis
- **Custom Loss Function**: Ensures predictions follow valid candlestick patterns

## Project Structure

```
crypto-candle-predictor/
├── train_candle_predictor.py  # Model training script
├── predict_ui.py              # Prediction interface
├── trading_agent.py           # Automated trading bot
├── models/                    # Saved models and scalers
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/irlfarmer/crypto-candle-predictor.git
cd crypto-candle-predictor
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python train_candle_predictor.py
```

2. Run the prediction UI:
```bash
streamlit run predict_ui.py
```

3. Run the trading bot:
```bash
streamlit run trading_agent.py
```

## Configuration

- For the trading bot, create a `config.json` file with your Binance API credentials:
```json
{
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "symbol": "BTCUSDT",
    "quantity": 0.001
}
```

## Model Architecture

- LSTM-based neural network
- Custom loss function enforcing candlestick pattern rules
- Feature engineering including technical indicators
- Percentage-based price predictions

## License

MIT License

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries significant risks. Use at your own risk. 