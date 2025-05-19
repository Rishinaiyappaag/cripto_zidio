import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('crypto_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

prophet_df = df.rename(columns={'timestamp': 'ds', 'close': 'y'})

model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=100)

forecast = model.predict(future)

fig1 = model.plot(forecast)
plt.title("Crypto Price Forecast using Prophet")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()
plt.show()
