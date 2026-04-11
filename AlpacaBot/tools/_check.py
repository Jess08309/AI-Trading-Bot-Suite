import pandas as pd
for s in ["SPY","QQQ","AAPL","MSFT","NVDA"]:
    df = pd.read_csv(f"data/historical/{s}_5min.csv")
    days = len(set(str(t)[:10] for t in df["timestamp"]))
    print(f"{s}: {len(df)} bars, {days} days, {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")
