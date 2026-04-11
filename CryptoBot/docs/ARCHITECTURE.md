# System Architecture - Crypto Trading Bot

> **Deep technical dive into the trading bot's design, components, and decision-making processes**

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Decision Engine](#decision-engine)
5. [Trading Logic](#trading-logic)
6. [Risk Management](#risk-management)
7. [Learning Systems](#learning-systems)
8. [Technical Decisions](#technical-decisions)

---

## System Overview

### High-Level Design

The trading bot is a **monolithic Python application** with a single-threaded event loop that runs continuously. It operates on a 60-second cycle, checking prices, analyzing market conditions, and executing trades across both spot and futures markets.

**Design Principles:**
- **Simplicity:** Single main loop, no microservices complexity
- **Safety:** Paper trading mode, comprehensive stop-losses
- **Observability:** Extensive logging, trade history tracking
- **Adaptability:** RL agent learns from experience
- **Robustness:** Error handling, graceful degradation

### Core Components

```
┌──────────────────────────────────────────────────────────────────┐
│                        MAIN TRADING LOOP                         │
│                     (60-second cycle time)                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA COLLECTION                                             │
│     ├─ Fetch prices (Coinbase spot, Kraken futures)            │
│     ├─ Fetch news (RSS feeds)                                  │
│     └─ Update price snapshots (100 per symbol)                 │
│                                                                  │
│  2. ANALYSIS                                                     │
│     ├─ ML Model (Random Forest)                                │
│     ├─ RL Agent (Q-Learning)                                   │
│     ├─ Sentiment Analysis (keyword-based)                      │
│     └─ Volatility Calculator                                   │
│                                                                  │
│  3. SCORING & DECISION                                          │
│     ├─ Ensemble scoring (0-15 points)                          │
│     ├─ Buy signal detection (≥4/15)                            │
│     └─ Futures direction (LONG vs SHORT)                       │
│                                                                  │
│  4. EXECUTION                                                    │
│     ├─ Spot market trades (BUY/SELL)                           │
│     └─ Futures market trades (LONG/SHORT/CLOSE)                │
│                                                                  │
│  5. RISK MANAGEMENT                                             │
│     ├─ Stop-loss monitoring (-2% spot, -3% futures)            │
│     ├─ Take-profit monitoring (+2% spot, +5% futures)          │
│     └─ Position limits (15 spot, 10 futures)                   │
│                                                                  │
│  6. LEARNING                                                     │
│     ├─ Update RL Q-values                                      │
│     ├─ Track coin performance                                  │
│     └─ Save state to disk                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Machine Learning Model (`trade_model.joblib`)

**Type:** Random Forest Classifier  
**Purpose:** Predict buy signal probability

**Training:**
- **Data Source:** Historical trades from `trade_history.csv`
- **Sample Size:** 162 trades (limited but sufficient for initial model)
- **Features:**
  - `change_5m`: Price change over 5 snapshots
  - `change_10m`: Price change over 10 snapshots
  - `volatility`: Standard deviation of recent prices
  - `ml_pred`: Recursive ML prediction (bootstrapping)
  - `rl_pred`: RL agent recommendation
  - `sentiment`: News sentiment score
  - `buy_score`: Ensemble score from previous cycle

- **Target Variable:** `profitable` (1 if trade made money, 0 otherwise)

**Model Configuration:**
```python
RandomForestClassifier(
    n_estimators=100,      # 100 trees
    max_depth=10,          # Limit tree depth to prevent overfitting
    random_state=42        # Reproducibility
)
```

**Performance:**
- Training Accuracy: ~65%
- Test Accuracy: ~45%
- **Interpretation:** Conservative predictions (high precision, lower recall)

**Why Random Forest?**
- Handles non-linear relationships (crypto prices are chaotic)
- Resistant to overfitting (ensemble of trees)
- Feature importance tracking (which signals matter most)
- Fast inference (milliseconds per prediction)

---

### 2. Reinforcement Learning Agent (`rl_agent.json`)

**Algorithm:** Q-Learning (Temporal Difference)  
**Purpose:** Learn optimal actions from trading experience

**State Space:**
```python
state = (price_bin, has_position)
# price_bin: 0-9 (discretized price level)
# has_position: 0/1 (True/False)
```

**Action Space:**
```python
actions = ["BUY", "SELL", "HOLD"]
```

**Q-Value Update:**
```python
Q(s, a) = Q(s, a) + α * [reward + γ * max(Q(s', a')) - Q(s, a)]

Where:
α (alpha) = 0.1      # Learning rate
γ (gamma) = 0.95     # Discount factor
reward = pnl_percent # Trade profit/loss %
```

**Exploration vs Exploitation:**
```python
epsilon = 0.2  # 20% random exploration
if random() < epsilon:
    action = random_choice(actions)  # Explore
else:
    action = argmax(Q[state])        # Exploit
```

**Why Q-Learning?**
- Simple and proven algorithm
- Works well with discrete state/action spaces
- Fast updates (no neural network overhead)
- Interpretable Q-values

**Limitations:**
- Small state space (10 bins × 2 positions = 20 states)
- Doesn't capture complex market patterns
- No memory of past sequences

**Future Enhancement:** Consider Deep Q-Network (DQN) for larger state space

---

### 3. Sentiment Analyzer (`news_sentiment.py`)

**Data Sources (RSS Feeds):**
```python
feeds = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://cryptobriefing.com/feed/",
    # ... more feeds
]
```

**Analysis Method:**
```python
def analyze_sentiment(text, symbol):
    score = 0
    
    # Positive keywords
    if any(word in text for word in ["rally", "surge", "pump", "bullish"]):
        score += 0.3
    
    # Negative keywords
    if any(word in text for word in ["crash", "dump", "bearish", "plunge"]):
        score -= 0.3
    
    # Symbol-specific boost
    if symbol.replace("-USD", "").lower() in text.lower():
        score *= 1.5
    
    return clip(score, -1.0, 1.0)
```

**Output Range:** -1.0 (bearish) to +1.0 (bullish)

**Why Keyword-Based?**
- Fast (no API calls or heavy NLP)
- Reliable for obvious sentiment
- No external dependencies
- Good enough for ensemble scoring

**Limitations:**
- Misses sarcasm/nuance
- Limited to predefined keywords
- No context understanding

**Future Enhancement:** Integrate GPT-based sentiment or FinBERT

---

### 4. Volatility Calculator

**Method:** Rolling standard deviation

```python
def calculate_volatility(prices):
    if len(prices) < 10:
        return 0.0
    
    recent_prices = prices[-10:]  # Last 10 snapshots (10 minutes)
    returns = [
        (prices[i] - prices[i-1]) / prices[i-1] 
        for i in range(1, len(prices))
    ]
    
    return std_dev(returns) * 100  # Convert to percentage
```

**Purpose:**
- Measure market turbulence
- Avoid trading during extreme volatility
- Adjust position sizing (future feature)

**Threshold:** Volatility > 5% triggers caution flag

---

## Data Flow

### Complete Cycle Diagram

```
┌─────────────┐
│   START     │
│  (60s wait) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  1. FETCH PRICES                        │
│     - Coinbase API (spot)               │
│     - Kraken API (futures)              │
│     - Update snapshots (circular buffer)│
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  2. FETCH NEWS                          │
│     - Parse RSS feeds                   │
│     - Extract headlines                 │
│     - Analyze sentiment                 │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  3. CALCULATE FEATURES                  │
│     - 5-min price change                │
│     - 10-min price change               │
│     - Volatility (std dev)              │
│     - ML prediction                     │
│     - RL recommendation                 │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  4. ENSEMBLE SCORING (per symbol)       │
│     ┌─────────────────────────────┐     │
│     │ Base: 3 points              │     │
│     │ + ML pred × 4 (0-4 pts)     │     │
│     │ + RL action (0-2 pts)       │     │
│     │ + Sentiment (0-2 pts)       │     │
│     │ + Price trend (0-2 pts)     │     │
│     │ + Low volatility (0-2 pts)  │     │
│     │ ─────────────────────────   │     │
│     │ Total: 0-15 points          │     │
│     └─────────────────────────────┘     │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  5. TRADING DECISIONS                   │
│                                         │
│  IF score ≥ 4/15 AND within limits:    │
│  ├─ SPOT: BUY $100 worth               │
│  └─ FUTURES: LONG or SHORT $200        │
│                                         │
│  IF position exists AND hit target:    │
│  ├─ SPOT: SELL at +2% or -2%           │
│  └─ FUTURES: CLOSE at +5% or -3%       │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  6. EXECUTE TRADES (Paper Trading)      │
│     - Update positions.json             │
│     - Log to trade_history.csv          │
│     - Adjust paper_balances.json        │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  7. UPDATE LEARNING                     │
│     - RL agent Q-values                 │
│     - Coin performance stats            │
│     - Meta learner patterns             │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  SLEEP 60s  │
│  (repeat)   │
└─────────────┘
```

---

## Decision Engine

### Ensemble Scoring Algorithm

**Objective:** Combine multiple signals into a single buy confidence score (0-15 points)

**Components & Weights:**

```python
def calculate_buy_score(symbol, features):
    score = 3  # Base score (neutral starting point)
    
    # 1. ML Model Prediction (0-4 points, 26.7% weight)
    ml_confidence = ml_model.predict_proba([features])[0][1]  # Prob of class 1
    score += ml_confidence * 4
    
    # 2. RL Agent Recommendation (0-2 points, 13.3% weight)
    rl_action = rl_agent.get_action(state)
    if rl_action == "BUY":
        score += 2
    elif rl_action == "HOLD":
        score += 1
    # SELL adds 0
    
    # 3. Sentiment Analysis (0-2 points, 13.3% weight)
    sentiment = get_sentiment(symbol)
    score += (sentiment + 1)  # Convert -1 to +1 → 0 to 2
    
    # 4. Price Trend (0-2 points, 13.3% weight)
    change_5m = (prices[-1] - prices[-5]) / prices[-5]
    if change_5m > 0.01:  # 1% increase
        score += 2
    elif change_5m > 0:
        score += 1
    
    # 5. Low Volatility Bonus (0-2 points, 13.3% weight)
    volatility = calculate_volatility(prices)
    if volatility < 2:  # Very stable
        score += 2
    elif volatility < 5:  # Somewhat stable
        score += 1
    
    # 6. Recent Performance (0-2 points, 13.3% weight)
    if symbol in top_performers:
        score += 2
    
    return round(score, 2)
```

**Scoring Breakdown:**
- 0-3 points: **AVOID** (weak signal, don't trade)
- 4-7 points: **CAUTION** (borderline, trade with smaller size)
- 8-11 points: **MODERATE** (decent signal, normal size)
- 12-15 points: **STRONG** (very confident, could increase size)

**Current Threshold:** 4/15 (26.7%) - Conservative entry point

---

### Futures Direction Logic

**Question:** Should we go LONG (bet on price increase) or SHORT (bet on price decrease)?

```python
def determine_futures_direction(symbol, ml_confidence, sentiment):
    # Method 1: ML Confidence (primary)
    if ml_confidence > 0.6:
        return "LONG"
    elif ml_confidence < 0.4:
        return "SHORT"
    
    # Method 2: Sentiment (tiebreaker)
    if sentiment > 0.2:
        return "LONG"
    elif sentiment < -0.2:
        return "SHORT"
    
    # Method 3: Price Momentum (final fallback)
    change_10m = (prices[-1] - prices[-10]) / prices[-10]
    if change_10m > 0:
        return "LONG"
    else:
        return "SHORT"
```

**Why this logic?**
- ML model is most reliable (trained on historical data)
- Sentiment adds real-time context
- Price momentum is last resort (always works)

---

## Trading Logic

### Spot Market (BUY/SELL)

**Entry Conditions:**
```python
if (
    buy_score >= 4.0 and                      # Strong enough signal
    len(spot_positions) < 15 and              # Not maxed out
    symbol not in spot_positions and          # Not already holding
    len(price_snapshots[symbol]) >= 10 and    # Enough history
    cash_balance >= 100                       # Can afford trade
):
    execute_spot_buy(symbol, quantity=100/price)
```

**Exit Conditions:**
```python
if symbol in spot_positions:
    buy_price = positions[symbol]['price']
    current_price = get_price(symbol)
    pnl_percent = (current_price - buy_price) / buy_price * 100
    
    if pnl_percent >= 2.0:
        execute_spot_sell(symbol, reason="TAKE_PROFIT")
    elif pnl_percent <= -2.0:
        execute_spot_sell(symbol, reason="STOP_LOSS")
```

**Position Tracking:**
```json
{
  "BTC-USD": {
    "price": 45000.0,
    "quantity": 0.00222,
    "timestamp": "2026-02-06T10:30:00"
  }
}
```

---

### Futures Market (LONG/SHORT/CLOSE)

**Entry Conditions:**
```python
if (
    buy_score >= 4.0 and
    len(futures_positions) < 10 and
    symbol not in futures_positions and
    len(price_snapshots[symbol]) >= 10
):
    direction = determine_futures_direction(symbol, ml_confidence, sentiment)
    execute_futures_entry(symbol, direction, size=200)
```

**Exit Conditions:**
```python
if symbol in futures_positions:
    entry_price = futures_positions[symbol]['entry_price']
    current_price = get_futures_price(symbol)
    direction = futures_positions[symbol]['direction']
    
    if direction == "LONG":
        pnl_percent = (current_price - entry_price) / entry_price * 100
    else:  # SHORT
        pnl_percent = (entry_price - current_price) / entry_price * 100
    
    if pnl_percent >= 5.0:
        execute_futures_close(symbol, reason="TAKE_PROFIT")
    elif pnl_percent <= -3.0:
        execute_futures_close(symbol, reason="STOP_LOSS")
```

**Position Tracking:**
```json
{
  "PI_XBTUSD": {
    "direction": "SHORT",
    "entry_price": 45000.0,
    "size": 200,
    "timestamp": "2026-02-06T10:32:00"
  }
}
```

---

## Risk Management

### Position Limits

**Why Limits?**
- Prevent over-concentration (too many eggs in one basket)
- Maintain liquidity (can't sell if fully invested)
- Reduce correlation risk (crypto moves together)

**Current Limits:**
- Spot: 15 positions (15 symbols × $100 = $1,500 max exposure)
- Futures: 10 positions (10 symbols × $200 = $2,000 max exposure)
- Total: $3,500 max exposure from $5,000 starting capital (70%)

### Stop-Loss Strategy

**Spot Markets:**
```python
STOP_LOSS_PERCENT = 2.0  # Exit at -2% loss

# Example:
# Buy BTC at $45,000
# Stop-loss triggers at $44,100 (-$900 / -2%)
```

**Futures Markets:**
```python
STOP_LOSS_PERCENT = 3.0  # Exit at -3% loss (higher due to leverage)

# Example:
# SHORT ETH at $3,000
# Price rises to $3,090
# Stop-loss triggers at +3% against us (-$90 loss)
```

**Why 60-Second Intervals?**
- **Problem:** At 120s intervals, prices could crash -15% between checks
- **Solution:** 60s intervals catch stop-losses before catastrophic losses
- **Trade-off:** Slightly higher API usage, but worth it for protection

### Take-Profit Strategy

**Spot Markets:**
```python
TAKE_PROFIT_PERCENT = 2.0  # Exit at +2% gain
```

**Futures Markets:**
```python
TAKE_PROFIT_PERCENT = 5.0  # Exit at +5% gain (higher upside potential)
```

**Risk/Reward Ratios:**
- Spot: 2% TP / 2% SL = **1:1 ratio** (need >50% win rate to profit)
- Futures: 5% TP / 3% SL = **1.67:1 ratio** (need >37.5% win rate)

---

## Learning Systems

### Reinforcement Learning Updates

**When to Update:**
- After every trade close (spot or futures)
- Positive reward for profitable trades
- Negative reward for losing trades

**Update Process:**
```python
def update_rl_agent(symbol, old_price, new_price, action, has_position):
    # Calculate reward
    if action == "SELL" and has_position:
        pnl_percent = (new_price - old_price) / old_price * 100
        reward = pnl_percent
    else:
        reward = 0
    
    # Get old and new states
    old_state = (discretize_price(old_price), has_position)
    new_state = (discretize_price(new_price), False)
    
    # Q-learning update
    old_q = Q[old_state][action]
    max_future_q = max(Q[new_state].values())
    new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
    
    Q[old_state][action] = new_q
    
    # Save to disk
    save_rl_agent()
```

### Meta Learner (Coin Performance Tracker)

**Purpose:** Identify which coins perform best/worst

```python
coin_performance = {
    "BTC-USD": {
        "total_trades": 150,
        "wins": 95,
        "losses": 55,
        "win_rate": 0.633,
        "avg_pnl": 0.15,
        "total_pnl": 22.5
    },
    # ... other coins
}
```

**Usage:**
- Boost buy scores for top performers (+2 points)
- Avoid or reduce allocation to worst performers
- Rebalance portfolio over time

---

## Technical Decisions

### Why 60-Second Intervals?

**Problem Statement:**
At 120-second intervals, the bot was experiencing catastrophic stop-loss failures. Trades would lose -10% to -15% when stop-loss was set at -2% to -3%.

**Root Cause Analysis:**
1. Price check at T=0: BTC at $45,000 (in position)
2. Market crashes between T=0 and T=120
3. Price check at T=120: BTC at $38,250 (-15%)
4. Stop-loss triggers, but too late

**Solution:**
Reduce interval to 60 seconds to catch crashes faster.

**Trade-offs:**
- ✅ Better stop-loss protection
- ✅ Faster reaction to market changes
- ❌ Slightly more API calls (2x)
- ❌ Higher CPU usage (minimal impact)

**Result:**
Expected improvement from -15% worst-case to -3% to -4% worst-case.

---

### Why 4/15 Buy Threshold?

**Problem Statement:**
At 2/15 threshold, the bot was generating 1,300+ trades in 14 hours. Many were low-quality trades that lost money.

**Analysis:**
- Total trades: 1,231 closed
- Win rate: 58.2%
- Average P/L: -0.087% (winning more but losses bigger)

**Solution:**
Increase threshold from 2/15 to 4/15 to filter out weak signals.

**Expected Impact:**
- ~60% reduction in trade frequency
- Higher quality trades (only strong signals)
- Better win rate and average P/L

**Mathematical Justification:**
```
2/15 = 13.3% threshold → Too permissive
4/15 = 26.7% threshold → More selective
8/15 = 53.3% threshold → Very conservative
```

4/15 strikes a balance between activity and quality.

---

### Why Random Forest Over LSTM?

**Considered Alternatives:**
1. **LSTM (Long Short-Term Memory)**: Deep learning for sequences
2. **XGBoost**: Gradient boosting trees
3. **Linear Regression**: Simple baseline

**Decision Matrix:**

| Criterion | Random Forest | LSTM | XGBoost | Linear |
|-----------|--------------|------|---------|--------|
| Training Speed | ✅ Fast | ❌ Slow | ✅ Fast | ✅ Very Fast |
| Data Requirements | ✅ 100+ | ❌ 10,000+ | ✅ 100+ | ✅ 50+ |
| Interpretability | ✅ High | ❌ Low | ⚠️ Medium | ✅ Very High |
| Overfitting Risk | ✅ Low | ⚠️ Medium | ⚠️ Medium | ❌ High |
| Inference Speed | ✅ ms | ❌ 10+ ms | ✅ ms | ✅ <1 ms |

**Winner:** Random Forest

**Reasoning:**
- Only 162 training samples (not enough for LSTM)
- Need fast training (retraining in future)
- Interpretability helps debugging
- Good balance of accuracy vs complexity

---

### Why Paper Trading?

**Reasons:**
1. **Safety:** No risk of losing real money
2. **Testing:** Can experiment with parameters
3. **Learning:** Understand bot behavior before committing capital
4. **Debugging:** Easier to troubleshoot without financial pressure
5. **Compliance:** No regulatory concerns

**When to Go Live:**
- ✅ Bot runs stable for 30+ days
- ✅ Positive P/L over 1,000+ trades
- ✅ Stop-losses working correctly
- ✅ Understand all code behavior
- ✅ Start with $100-$500 real money (not $5,000)

---

### Why Coinbase + Kraken?

**Spot Markets:** Coinbase Advanced Trade API
- ✅ Excellent documentation
- ✅ Reliable uptime
- ✅ Good liquidity on major pairs
- ✅ Simple REST API

**Futures Markets:** Kraken Futures API
- ✅ Supports perpetual contracts
- ✅ Competitive fees
- ✅ Good LONG/SHORT functionality
- ❌ More complex API than Coinbase

**Alternative Considered:**
- Binance (better fees but regulatory issues in US)
- FTX (defunct)

---

## Summary

**Key Architectural Principles:**

1. **Simplicity Over Complexity**
   - Single-threaded loop (no async/threading complexity)
   - Monolithic design (no microservices overhead)
   - Direct file I/O (no database setup required)

2. **Safety First**
   - Paper trading by default
   - Comprehensive stop-losses
   - Position limits to prevent over-exposure
   - Graceful error handling

3. **Observable & Debuggable**
   - Extensive logging to console and file
   - Trade history CSV for analysis
   - JSON state files for inspection
   - Clear scoring breakdown in logs

4. **Adaptive & Learning**
   - RL agent improves from experience
   - Meta learner tracks coin performance
   - Ensemble approach combines multiple signals

5. **Practical & Maintainable**
   - Clean code structure (1110 lines, well-organized)
   - Documented decisions and trade-offs
   - Easy to modify parameters
   - One-file deployment (main loop in trading_engine.py)

**This architecture prioritizes robustness, safety, and learnability over maximum performance. It's designed for a solo developer to understand, debug, and improve over time.**

---

*Last Updated: February 6, 2026*
