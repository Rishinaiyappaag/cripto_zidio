import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load your dataset (make sure it has columns like open, high, low, close)
data = pd.read_csv('crypto_data.csv')  # replace with actual file path

# Add technical indicators
def add_technical_indicators(data):
    data['price_change'] = data['close'].pct_change()

    for window in [5, 10, 20, 50, 100]:
        data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
        data[f'ema_{window}'] = data['close'].ewm(span=window, adjust=False).mean()

    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    data['macd'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    sma_20 = data['close'].rolling(window=20).mean()
    std_20 = data['close'].rolling(window=20).std()
    data['upper_band'] = sma_20 + (2 * std_20)
    data['lower_band'] = sma_20 - (2 * std_20)

    if 'volume' in data.columns:
        data['volume_change'] = data['volume'].pct_change()
        data['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()

    return data

data = add_technical_indicators(data)
data.dropna(inplace=True)

# Normalize features
scaler = MinMaxScaler()
features = data.drop(['close'], axis=1)
features = features.select_dtypes(include=[np.number])  # keep only numeric columns
scaled_data = scaler.fit_transform(features)
target = data['close'].values.reshape(-1, 1)
target_scaler = MinMaxScaler()
scaled_target = target_scaler.fit_transform(target)

# Sequence creation
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, scaled_target, sequence_length)

# Train/val/test split
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Model architecture: Conv1D + BiLSTM
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(units=50, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(units=50)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

# Training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

# Predictions and inverse scaling
y_pred_scaled = model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_true = target_scaler.inverse_transform(y_test)

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)
direction_accuracy = np.mean(np.sign(np.diff(y_true.flatten())) == np.sign(np.diff(y_pred.flatten())))

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R² Score: {r2:.2f}')
print(f'Direction Accuracy: {direction_accuracy * 100:.2f}%')

# Plot predictions
plt.figure(figsize=(14, 6))
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs. Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()

# Error histogram
errors = y_true.flatten() - y_pred.flatten()
plt.figure(figsize=(10, 4))
sns.histplot(errors, bins=50, kde=True)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')
plt.grid(True)
plt.show()

# Predict future prices
def predict_future(model, recent_data, steps, scaler_X, scaler_y):
    future_predictions = []
    input_seq = recent_data.copy()

    for _ in range(steps):
        pred = model.predict(input_seq[np.newaxis, :, :])[0, 0]
        future_predictions.append(pred)

        new_input = np.roll(input_seq, -1, axis=0)
        new_input[-1, :] = np.append(input_seq[-1, :-1], pred)[:input_seq.shape[1]]
        input_seq = new_input

    future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()

last_seq = X_test[-1]
future_steps = 10
future_prices = predict_future(model, last_seq, future_steps, scaler, target_scaler)

plt.figure(figsize=(10, 5))
plt.plot(range(len(y_true)), y_true, label='Actual')
plt.plot(range(len(y_true), len(y_true) + future_steps), future_prices, label='Future Prediction')
plt.title('Future Price Forecast')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
