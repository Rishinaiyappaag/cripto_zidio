# cripto_zidio
# Cryptocurrency Price Prediction

This project demonstrates time series forecasting of cryptocurrency prices using two distinct approaches:
- **ARIMA model** for classical statistical forecasting
- **LSTM (Long Short-Term Memory)** neural network with Conv1D and BiLSTM layers for deep learning-based prediction

It fetches historical BTC/USDT data from the Binance exchange using the `ccxt` library.

---

## 📁 Project Structure

├── arima_model.py # ARIMA model for time series forecasting
├── using_LSTM.py # LSTM-based deep learning model
├── crypto_pred.py # Data fetching utility using CCXT
├── crypto_data.csv # CSV input for LSTM model (must be provided)

markdown
Copy
Edit

---

## 🧠 Models Overview

### 1. ARIMA (AutoRegressive Integrated Moving Average)

- Implemented using `statsmodels` and `pmdarima`
- Automatically determines optimal ARIMA parameters
- Visualizes historical and forecasted prices with confidence intervals

**File**: `arima_model.py`  
**Dependencies**:
- `statsmodels`
- `pmdarima`
- `matplotlib`
- `ccxt` (indirectly through `crypto_pred.py`)

### 2. LSTM with Conv1D + BiLSTM

- Adds multiple technical indicators (e.g., RSI, MACD, Bollinger Bands)
- Scales and sequences data
- Deep learning architecture includes Conv1D, MaxPooling, BiLSTM, and Dropout
- Evaluates performance (RMSE, MAE, MAPE, R², Direction Accuracy)
- Plots actual vs. predicted prices and error distribution

**File**: `using_LSTM.py`  
**Dependencies**:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `tensorflow` or `keras`

---

## 📦 Requirements

Install all necessary libraries with:

```bash
pip install -r requirements.txt
requirements.txt (suggested contents):
nginx
Copy
Edit
pandas
numpy
matplotlib
seaborn
scikit-learn
keras
tensorflow
statsmodels
pmdarima
ccxt
📊 How to Use
ARIMA:
bash
Copy
Edit
python arima_model.py
Make sure you are connected to the internet for fetching real-time crypto data.

LSTM:
Place your dataset (with columns like open, high, low, close, volume) as crypto_data.csv

Run:

bash
Copy
Edit
python using_LSTM.py
📈 Forecast Output
ARIMA generates a future forecast with confidence intervals.

LSTM provides:

Prediction accuracy metrics

Visualization of actual vs predicted

Forecast for future steps

Error distribution histogram

