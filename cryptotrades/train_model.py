"""
Train a real trading model with proper feature engineering.
Matches the bot's 11 features exactly.
"""

import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

print("=" * 60)
print("CRYPTO TRADING MODEL TRAINER")
print("=" * 60)

# Fetch 2 years of historical data for more samples
print("\n[1/5] Fetching historical data...")
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    "vs_currency": "usd",
    "days": "730",  # 2 years
    "interval": "daily"
}

resp = requests.get(url, params=params, timeout=10)
resp.raise_for_status()
data = resp.json()

prices = [float(p[1]) for p in data['prices']]
print(f"✅ Got {len(prices)} days of price data")

# Create DataFrame
df = pd.DataFrame({'price': prices})

# Calculate all 11 features to match bot's FEATURE_COLUMNS
print("\n[2/5] Engineering features...")

# 1. price (current price)
df['price'] = df['price']

# 2. return_1 (1-period return)
df['return_1'] = df['price'].pct_change()

# 3. momentum_5 (5-period momentum)
df['momentum_5'] = df['price'].pct_change(5)

# 4. trend_strength (EMA5 - EMA20 / price)
df['ema_fast'] = df['price'].ewm(span=5, adjust=False).mean()
df['ema_slow'] = df['price'].ewm(span=20, adjust=False).mean()
df['trend_strength'] = (df['ema_fast'] - df['ema_slow']) / df['price']

# 5. volatility_10 (10-period volatility)
df['volatility_10'] = df['return_1'].rolling(10).std()

# 6. macd (MACD line)
ema12 = df['price'].ewm(span=12, adjust=False).mean()
ema26 = df['price'].ewm(span=26, adjust=False).mean()
df['macd'] = ema12 - ema26

# 7. macd_signal (Signal line)
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# 8. macd_hist (MACD histogram)
df['macd_hist'] = df['macd'] - df['macd_signal']

# 9. stoch_k (Stochastic K)
low_14 = df['price'].rolling(14).min()
high_14 = df['price'].rolling(14).max()
df['stoch_k'] = ((df['price'] - low_14) / (high_14 - low_14)) * 100
df['stoch_k'] = df['stoch_k'].fillna(50)  # Default to 50 if NaN

# 10. stoch_d (Stochastic D - smoothed K)
df['stoch_d'] = df['stoch_k'].rolling(3).mean()
df['stoch_d'] = df['stoch_d'].fillna(50)

# 11. rsi_14 (RSI)
delta = df['price'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['rsi_14'] = 100 - (100 / (1 + rs))
df['rsi_14'] = df['rsi_14'].fillna(50)  # Default to 50 if NaN

print("✅ Created all 11 features")

# Create target: 1 if price goes up 2% in next 5 days, 0 otherwise
print("\n[3/5] Creating labels...")
df['target'] = (df['price'].shift(-5) > df['price'] * 1.02).astype(int)

# Drop NaN rows
df = df.dropna()

# Select features matching FEATURE_COLUMNS exactly
feature_cols = [
    "price",
    "return_1",
    "momentum_5",
    "trend_strength",
    "volatility_10",
    "macd",
    "macd_signal",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    "rsi_14",
]

X = df[feature_cols].values
y = df['target'].values

print(f"✅ Training on {len(X)} samples")
print(f"   Buy signals (target=1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"   Sell signals (target=0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")

# Train/Test split
print("\n[4/5] Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"   Train set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    min_samples_split=10,  # Prevent overfitting
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,             # Use all CPU cores
)

model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"✅ Model trained")
print(f"   Train accuracy: {train_score:.2%}")
print(f"   Test accuracy: {test_score:.2%}")

# Feature importance
print("\n[5/5] Feature importance:")
importances = model.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"   {feat:20s}: {imp:.2%}")

# Save
joblib.dump(model, 'trade_model.joblib')
print(f"\n✅ Saved trade_model.joblib")

# Statistics
print("\n" + "=" * 60)
print("MODEL STATISTICS")
print("=" * 60)
print(f"Features:          {len(feature_cols)}")
print(f"Training samples:  {len(X_train)}")
print(f"Test samples:      {len(X_test)}")
print(f"Train accuracy:    {train_score:.2%}")
print(f"Test accuracy:     {test_score:.2%}")
print(f"Overfit gap:       {(train_score - test_score):.2%}")
print(f"Data range:        {df['price'].min():.0f} - {df['price'].max():.0f} USD")
print("=" * 60)

if test_score < 0.55:
    print("⚠️  WARNING: Test accuracy < 55%")
    print("   Model might not be profitable")
elif test_score < 0.60:
    print("⚠️  WARNING: Test accuracy < 60%")
    print("   Model is marginal, use with caution")
else:
    print("✅ Model looks reasonable!")
    print("   Start paper trading to test")
