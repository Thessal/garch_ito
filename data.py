import pandas as pd

def get_data(src = "./data/"):
    df = pd.read_pickle("tmp.pkl")
    return df

    # Binance BTCUSDT 1s spot data

    names=["timestamp","open","high","low","close","volume_BTC","close_timestamp","volume_USDT","num_trade","sell_volume_in_BTC","buy_volume_in_USDT", "_"]
    usecols=["timestamp","close"]
    df = pd.concat([
        pd.read_csv(src+"BTCUSDT-1s-"+date+".zip", header=None, names=names, usecols=usecols)
        for date in pd.date_range(start = "2022-11", end = "2025-11").to_period('M').astype(str).unique()
    ])

    for f in ["timestamp", "close_timestamp"]:
        if f in df.columns:
            df[f] = pd.to_datetime(df[f].apply(lambda x: (x*1000000) if x<1500000000000000 else (x*1000)))

    df = df.set_index("timestamp", drop=True).sort_index()
    
    return df