import ccxt
import pandas as pd
import time

binance = ccxt.binance()

def get_crypto_data():
    symbol = 'BTC/USDT'
    timeframe = '1d'
    limit = 1000

    timeframe_to_ms = {
        '1m': 60_000,
        '5m': 300_000,
        '15m': 900_000,
        '30m': 1_800_000,
        '1h': 3_600_000,
        '4h': 14_400_000,
        '1d': 86_400_000,
        '1w': 604_800_000
    }
    since = binance.parse8601('2017-01-01T00:00:00Z')
    all_ohlcv = []

    while True:
        print("Fetching candles since", binance.iso8601(since))
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)

        if len(ohlcv) < limit:
            break

        since = ohlcv[-1][0] + timeframe_to_ms[timeframe]

        time.sleep(binance.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(df.head())
    print(df.to_string())
    print(f"\nTotal rows fetched: {len(df)}")
    print(df.shape)
    return df

