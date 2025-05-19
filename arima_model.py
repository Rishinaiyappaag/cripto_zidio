from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pmdarima import auto_arima
import pandas as pd

from crypto_pred import get_crypto_data
df =get_crypto_data()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.asfreq('D')
df['close'].interpolate(inplace=True)

result = adfuller(df['close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

auto_model = auto_arima(df['close'], start_p=1, start_q=1,
                        max_p=5, max_q=5, d=1,
                        seasonal=False, trace=True,
                        stepwise=True, suppress_warnings=True)

print(auto_model.summary())
best_order = auto_model.order

model = ARIMA(df['close'], order=best_order)
model_fit = model.fit()
print(model_fit.summary())

n_steps = 250
forecast = model_fit.get_forecast(steps=n_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Historical')
plt.plot(future_dates, forecast_mean, label='Forecast', color='red')
plt.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)

#
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

plt.title("ARIMA Forecast of Crypto Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
