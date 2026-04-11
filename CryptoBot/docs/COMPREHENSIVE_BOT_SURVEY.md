# Comprehensive Architecture Survey: AlpacaBot vs CryptoBot

> **Generated from full source-code analysis of both codebases.**
> AlpacaBot: `C:\AlpacaBot` | CryptoBot: `C:\Bot\cryptotrades`

---

## Table of Contents
1. [Signal Generation Pipeline](#1-signal-generation-pipeline)
2. [ML Model Architecture](#2-ml-model-architecture)
3. [RL Agent](#3-rl-agent)
4. [Meta-Learner Ensemble](#4-meta-learner-ensemble)
5. [Risk Management](#5-risk-management)
6. [Market Regime Detection](#6-market-regime-detection)
7. [Data Sources & APIs](#7-data-sources--apis)
8. [Sentiment Analysis](#8-sentiment-analysis)
9. [Execution Logic](#9-execution-logic)
10. [Asset Selection & Universe](#10-asset-selection--universe)
11. [Unique / Differentiating Features](#11-unique--differentiating-features)

---

## 1. Signal Generation Pipeline

### AlpacaBot (Options Scalp)
**File:** `core/trading_engine.py` — `_generate_signals()` + `core/scanner.py`

**Indicator Scoring System:**
14 technical indicators are computed, then scored into `bull_score` and `bear_score` tallies:

| Indicator | Bull Condition → Points | Bear Condition → Points |
|---|---|---|
| RSI(14) | <25 → +2, 35–55 → +1 | >75 → +2, 50–65 → +1 |
| MACD Histogram | >0 → +1, >0.1 → +2 | <0 → +1, <-0.1 → +2 |
| Stochastic %K(14) | <20 → +1 | >80 → +1 |
| BB %B(20,2σ) | <0.10 → +2, <0.30 → +1 | >0.90 → +2, >0.70 → +1 |
| ATR Norm(14) | >0.005 → +1 (to winning side) | — |
| CCI(20) | <-100 → +1 | >100 → +1 |
| ROC(10) | >0.3 → +1 | <-0.3 → +1 |
| Williams %R(14) | <-80 → +1 | >-20 → +1 |
| Vol Ratio(5/20) | >1.3 → +1 (to winning side) | — |
| Z-score(20) | <-2 → +1 | >2 → +1 |
| Trend Strength(14) | >25 → +1 (to winning side) | — |
| Price Δ 1-bar | >0.001 → +1 | <-0.001 → +1 |
| Price Δ 5-bar | >0.003 → +1 | <-0.003 → +1 |
| Price Δ 20-bar | (momentum sanity check, see below) | — |

**Direction Decision:**
- Requires `MIN_SIGNAL_SCORE ≥ 3` AND winning side > losing side + 1
- Calls direction assigned as CALL (bull wins) or PUT (bear wins)

**Quality Gate Chain (sequential — all must pass):**
1. **Direction enable flags** — per-direction on/off
2. **SPY regime filter** — CALL blocked in bear market, PUT blocked in bull
3. **Sentiment direction filter** — sentiment must agree (threshold ±0.15)
4. **ML must agree with direction** — hard gate, ML prediction must match
5. **ML confidence ≥ 0.55** — `MIN_ML_CONFIDENCE`
6. **Meta-learner ensemble gate** — `should_trade()` must return True

**Momentum Sanity Check:**
- Blocks PUTs when `price_change_20 > 1.5%` AND `price_change_5 > 0.5%`
- Blocks CALLs when `price_change_20 < -1.5%` AND `price_change_5 < -0.5%`

---

### CryptoBot (Crypto Spot + Futures)
**File:** `core/trading_engine.py` — `generate_signals()`

**ML-First Approach** (no indicator scoring — ML prediction drives direction):
1. Require ≥50 price history points per symbol
2. ML prediction → `ml_direction` (probability of UP) + `ml_conf`
3. **Strict confidence gate:** `ml_conf ≥ MIN_ML_CONFIDENCE (0.66)` — no bypasses

**Direction Logic (trend-following):**

| ML Direction | Trend | Additional RSI Filter | Confidence Penalty |
|---|---|---|---|
| `ml_direction > 0.62` (long trigger) | UP | RSI < 62 | 1.0× (none) |
| `ml_direction > 0.62` | SIDE | RSI < 62 AND sent ≥ -0.15 | 0.85× |
| `ml_direction > 0.62` | DOWN (counter-trend) | RSI < 62, needs `ml_conf ≥ 0.92` | 0.70× |
| `ml_direction < 0.38` (short trigger) | DOWN | RSI > 38 | 1.0× (none) |
| `ml_direction < 0.38` | SIDE | RSI > 38 AND sent ≤ 0.15 | 0.85× |
| `ml_direction < 0.38` | UP (counter-trend) | RSI > 38, needs `ml_conf ≥ 0.92` | 0.70× |

**Direction Bias (env-configurable):**
- `DIRECTION_BIAS`: neutral / short_lean / long_lean
- `DIRECTION_BIAS_STRENGTH`: 0.04 default (max 0.15)
- Shifts long/short trigger thresholds by ±strength

**Additional Filters:**
- SIDE-market filter: skip sideways unless `ml_conf ≥ 0.75` (`SIDE_MARKET_ML_OVERRIDE`)
- Symbol blacklist check
- Per-symbol auto-pause (consecutive losses)
- Per-direction performance pause (WR < 30% over last 20 trades → 1h pause)
- Direction mode: `long_only` / `short_only` / `both`
- Correlation gate against existing positions

---

## 2. ML Model Architecture

### AlpacaBot
**File:** `utils/ml_model.py`

| Parameter | Value |
|---|---|
| Algorithm | `GradientBoostingClassifier` (sklearn) |
| n_estimators | 200 |
| max_depth | 4 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| min_samples_split | 20 |
| min_samples_leaf | 10 |
| Features | 20 (14 price indicators + 6 context) |
| Target | Price direction over next 6 bars (1 hour) |
| Flat move filter | ±0.05% excluded from training |
| Validation | TimeSeriesSplit CV (2–5 folds) |
| Min accuracy gate | 0.54 |
| Retrain interval | 8 hours |
| Model versioning | Keeps last 5, rollback capability |
| Training data | 10-minute bars, bootstrap from 30d history |

**20 Features:**
```
14 price indicators: rsi_14, macd_histogram, stoch_k, bb_pct_b, atr_norm,
    cci_20, roc_10, williams_r, vol_ratio, z_score, trend_strength,
    price_change_1, price_change_5, price_change_20
6 context features: hour_sin, hour_cos, day_of_week, intraday_range,
    gap_open, vol_regime (low=0/normal=1/high=2)
```

**Vol Regime Buckets:** `<0.003 = low`, `0.003–0.008 = normal`, `>0.008 = high`

**Class Balancing:** Undersamples majority class to match minority count.

**Prediction Output:** `{direction: up_prob, confidence: max(up_prob, down_prob)}`

---

### CryptoBot
**Files:** `core/trading_engine.py` (inline MLModel) + `utils/market_predictor.py`

| Parameter | Value |
|---|---|
| Algorithm | `GradientBoostingClassifier` (sklearn) |
| n_estimators | 200 |
| max_depth | 4 (inline) / 5 (market_predictor.py) |
| learning_rate | 0.05 |
| subsample | 0.8 |
| min_samples_split | 20 |
| min_samples_leaf | 10 |
| Features | 15 (momentum oscillators + complementary) |
| Target | Price direction over next 5 periods |
| Validation | train_test_split (inline) / adaptive window (predictor) |
| Min accuracy gate | 0.58 (inline) / 0.48 (predictor) |
| Retrain interval | Configurable (`MODEL_RETRAIN_HOURS`) |
| Max training points | 2000 (~33 hours of 1-min data) |
| Training data | 1-min → aggregated to 10-min + 1-hr candles |

**15 Features:**
```
10 momentum oscillators: rsi_14, macd_histogram, stoch_k, cci_20, roc_10,
    momentum_10, williams_r, ultimate_osc(7/14/28), trix_15, cmo_14
5 complementary: atr_14 (normalized), trend_strength, bb_position,
    mean_reversion (z-score), vol_ratio
```

**Key Differences from AlpacaBot:**
- No time-of-day features (crypto trades 24/7)
- Adds ultimate oscillator, TRIX, CMO, raw momentum (AlpacaBot uses price changes instead)
- No options-specific features (no IV rank, gap_open)
- Trains on multi-symbol price history simultaneously

---

## 3. RL Agent

### AlpacaBot — Q-Table (Tabular)
**File:** `utils/rl_agent.py` (377 lines)

| Parameter | Value |
|---|---|
| Architecture | Q-table (dictionary-based) |
| State space | 5 dimensions × 3 buckets = 243 states |
| Actions | 4: SKIP(0.0×), SMALL(0.5×), MEDIUM(1.0×), LARGE(1.5×) |
| Learning rate (α) | 0.1 |
| Discount (γ) | 0.95 |
| Epsilon (ε) | 0.15 initial, decay 0.9995, min 0.05 |
| Reward | `pnl_pct × sizing_mult × 100`, clipped [-5, +5] |
| **Mode** | **SHADOW ONLY — never places live trades** |

**State Discretization:**

| Dimension | Low | Neutral | High |
|---|---|---|---|
| RSI | <35 (oversold) | 35–65 | >65 (overbought) |
| Trend | pc5 < -0.003 (down) | sideways | pc5 > 0.003 AND trend > 20 (up) |
| Vol Regime | <0.003 | 0.003–0.008 | >0.008 |
| Sentiment | <-0.2 (bearish) | -0.2 to 0.2 | >0.2 (bullish) |
| ML Confidence | <0.55 | 0.55–0.70 | >0.70 |

**Shadow Mode Tracking:**
- Records hypothetical decisions, tracks shadow equity vs baseline
- Promotion threshold: 50 consecutive trades outperforming baseline required before going live

---

### CryptoBot — Deep Q-Network (DQN)
**File:** `utils/rl_agent.py` (529 lines)

| Parameter | Value |
|---|---|
| Architecture | DQN: 5→64→64→5 (LayerNorm + ReLU) via PyTorch |
| State space | 5 continuous dimensions (no discretization) |
| Actions | 5: [0.25×, 0.5×, 1.0×, 1.5×, 2.0×] size multipliers |
| Learning rate | 0.001 (Adam optimizer) |
| Discount (γ) | 0.95 |
| Epsilon (ε) | 0.15 initial, decay 0.999, min 0.05 |
| Experience replay | Buffer capacity 10,000, batch_size=32 |
| Target network | Syncs every 50 update steps (Double DQN) |
| Loss function | Huber loss |
| Gradient clipping | max_norm = 1.0 |
| **Mode** | **Active in ensemble** (participates in meta-learner with weight 0.15) |

**State Vector (continuous):**
```
[sentiment, volatility/0.1, trend/50, rsi/100, ml_confidence]
```

**Reward Shaping:**
- If `|reward| > 0.5`: amplify × 1.5
- If `|reward| < 0.1`: dampen × 0.5
- Per-coin exploration bonus: `0.1 × (1 - trades/10)` for coins with <10 trades

**Q-Table Migration:** Converts legacy Q-table knowledge into replay buffer experiences on startup.

**Shadow Books:** Both baseline and RL strategies tracked in parallel shadow portfolios with identical slippage/fee model. RL shadow learns from shadow outcomes even when in shadow mode.

---

## 4. Meta-Learner Ensemble

### AlpacaBot
**File:** `utils/meta_learner.py` (294 lines)

| Parameter | Value |
|---|---|
| Sources | ML Model (0.35), Sentiment (0.15), Rule Score (0.50) |
| Min weight floor | 0.10 per source |
| Confidence threshold | 0.68 default |
| Min rule score | 6 default |
| Weight update window | Rolling 50 predictions |
| RL included? | **No** — RL is shadow-only |

**Dynamic Threshold Adjustment:**

| Condition | Action |
|---|---|
| Rolling loss rate ≥ 60% OR ≥3 consecutive losses | Tighten: conf → 0.80, score → 8 |
| Win rate > 60% AND 0 consecutive losses | Loosen: conf → 0.60, score floor stays ≥ 6 |

**Gate:** `should_trade()` requires `ensemble_conf ≥ threshold` AND `rule_score ≥ min_rule_score`

---

### CryptoBot
**File:** `utils/meta_learner.py` (~280 lines)

| Parameter | Value |
|---|---|
| Sources | ML Model (0.50), Sentiment (0.35), RL Agent (0.15) |
| RL inclusion condition | Only if `|rl_prediction - 0.5| > 0.1` |
| Buy threshold | 0.55 default |
| Sell threshold | 0.45 default |
| Weight update | Based on rolling win rate |

**Sentiment Modulation:**
- Global sentiment > 0.3 → buy_threshold -= 0.05, sell_threshold -= 0.05 (easier to buy)
- Global sentiment < -0.3 → buy_threshold += 0.05, sell_threshold += 0.05 (harder to buy)

**Threshold Adaptation:**
- Formula: `adjustment = (win_rate - 50) / 100 × 0.02`
- Buy threshold range: 0.40 – 0.80
- Sell threshold range: 0.20 – 0.60

**Key Difference:** CryptoBot includes RL in the live ensemble; AlpacaBot keeps RL shadow-only.

---

## 5. Risk Management

### AlpacaBot — Graduated Response System
**File:** `core/risk_manager.py` (497 lines)

**Graduated Response Tiers (replaces binary circuit breaker):**

| Consecutive Losses | Size Multiplier | Min Signal Score | Cooldown | Direction Lock |
|---|---|---|---|---|
| 0–2 | 100% | 3 | None | No |
| 3–4 | 50% | 7 | 5 min | Yes |
| 5–7 | 25% | 9 | 10 min | Yes |
| 8+ | 10% | 10 | 15 min | Yes |

**Hard Stops:** Only `HARD_DAILY_LOSS_PCT = -5%` fully stops trading.

**Rapid-Fire Detection:** 3 losses within 20 minutes → extra 10-minute cooldown.

**Direction-Specific Locking:** 3 consecutive same-direction losses → lock that direction only.

**Per-Symbol Pause:** 3 consecutive losses on same symbol → pause that symbol.

**Position Sizing Formula:**
```
max_spend = balance × MAX_POSITION_PCT(4%) × conf_factor(0.5–1.0) × count_factor(0.4–1.0)
min_spend = balance × MIN_POSITION_PCT(2%)
```
- `conf_factor` = 0.5 + (confidence - 0.5) × 2 (linear from 50%→100% confidence)
- `count_factor` = 1.0 - (num_positions × 0.15), floor 0.4

**Rolling Loss Rate:** Tracks last 10 trades across all symbols/days.

**Exit Rules:**

| Rule | Value |
|---|---|
| Stop Loss | -15% |
| Take Profit | +15% |
| Trailing Stop | 12% drawdown from peak, triggers at +6% profit |
| Max Hold (≤2 DTE) | 1 day |
| Max Hold (≤9 DTE) | 3 days |
| Max Hold (16+ DTE) | 7 days |
| Phantom detection | 3 sell failures → API position verification |

---

### CryptoBot — Kelly Criterion + Circuit Breakers
**Files:** `core/trading_engine.py` (inline RiskManager) + `utils/circuit_breaker.py` + `utils/position_sizer.py`

**Circuit Breaker Triggers:**

| Trigger | Spot Default | Futures Default |
|---|---|---|
| Consecutive losses | 5 | 5 |
| Daily loss limit | -4% | -4% |
| Max drawdown | -8% | -8% |
| Cooldown | 60 min | 60 min |

**Position Sizing (Kelly Criterion):**
```python
kelly = max(0.01, 2 × confidence - 1)
vol_adjust = 1 / (1 + volatility × 10)
position_factor = max(0.3, 1 - num_positions × 0.15)
size = balance × kelly × vol_adjust × position_factor
```

**Scaled Sizing System (1–3× multipliers):**

Score points from: confidence (+2), low volatility (+2), trending regime (+1), low correlation (+1)

| Score | Multiplier |
|---|---|
| 0–2 | 1× |
| 3–4 | 2× |
| 5–6 | 3× |

**Per-Symbol Auto-Pause:** Configurable consecutive loss threshold → 2-hour pause.

**Per-Direction Performance Tracker:**
- Lookback: 20 trades
- Pause threshold: WR < 30%
- Pause duration: 1 hour
- Min trades to judge: configurable

**Exit Rules:**

| Rule | Spot | Futures |
|---|---|---|
| Stop Loss | -2% | -3% |
| Take Profit | +2.5% | +2.5% |
| Trailing Stop | 1.5% (trigger at +1% PnL) | 1.5% |
| Stale Profit Decay | Arm at +1.5%, decay to 0.8%, stale after 45 min | After 25 min |
| Max Hold (flat ±0.5%) | 4 hours | 3 hours |
| Max Hold (forced) | 8 hours | 6 hours |

---

## 6. Market Regime Detection

### AlpacaBot
**Regime Source:** SPY trend as market proxy (from Alpaca bars).

**Implementation:**
- Fetches SPY candles and computes trend indicators
- Three regimes: BULL / BEAR / NEUTRAL
- **Filter logic:** CALLs blocked in BEAR regime, PUTs blocked in BULL regime
- Acts as a hard gate in the signal pipeline

---

### CryptoBot
**File:** `utils/config.py` + `core/trading_engine.py` — `MarketData.calculate_trend()`

**Regime Parameters:**

| Parameter | Value |
|---|---|
| `REGIME_TREND_LOOKBACK` | 30 periods |
| `REGIME_VOL_LOOKBACK` | 30 periods |
| `REGIME_TREND_THRESH` | 0.0005 |
| `REGIME_VOL_MULT` | 1.35 |
| `REGIME_COOLDOWN_MIN` | 20 minutes |

**Trend Detection:** Linear regression slope on price history:
- `slope > REGIME_TREND_THRESH` → UP
- `slope < -REGIME_TREND_THRESH` → DOWN
- Otherwise → SIDE

**SIDE-Market Handling:**
- Default: skip sideways markets entirely (`SIDE_MARKET_FILTER = True`)
- ML override: allow trade if `ml_conf ≥ 0.75` (`SIDE_MARKET_ML_OVERRIDE`)
- Counter-trend override: allow if `ml_conf ≥ 0.92` (`COUNTER_TREND_ML_OVERRIDE`)

**Volatility Scaling (regime-adaptive stops):**

| Parameter | Value |
|---|---|
| `VOL_WINDOW` | 20 periods |
| `VOL_MIN` multiplier | 0.5 |
| `VOL_MAX` multiplier | 1.6 |
| `VOL_BASE_STOP` | 1.2% |
| `VOL_BASE_TP` | 2.0% |
| `VOL_BASE_TRAIL` | 1.2% |

---

## 7. Data Sources & APIs

### AlpacaBot

| Data Type | Source | API | Auth Required |
|---|---|---|---|
| Equity bars (10-min) | Alpaca Data API | REST | Yes (API key) |
| Options chains | Alpaca Options API | REST | Yes |
| Options snapshots | Alpaca Options API | REST | Yes |
| News headlines | Alpaca News API (Benzinga) | REST | Yes |
| Market movers | Alpaca Screener API | REST | Yes |
| VIX proxy | VIXY ETF price via Alpaca | REST | Yes |
| SPY regime | SPY bars via Alpaca | REST | Yes |

**Rate Limiting:** 0.4s delay between scanner API calls. Scanner runs every 10 minutes.

**Data Intervals:** 10-minute bars, lookback 50 bars, bootstrap from 30-day history for top 20 scanner symbols.

---

### CryptoBot

| Data Type | Source | API | Auth Required |
|---|---|---|---|
| Spot prices | Coinbase API (auth + public fallback) | REST | Optional |
| Futures prices (primary) | Coinbase product endpoints | REST | No |
| Futures prices (fallback) | Kraken Futures `/tickers` | REST | No |
| Futures orderbook depth | Kraken Futures `/orderbook` | REST | No |
| Spot orderbook depth | Coinbase Exchange `/book` | REST | No |
| Fear & Greed Index | alternative.me | REST | No |
| Global market data | CoinGecko `/global` | REST | No |
| Coin prices & momentum | CoinGecko `/coins/markets` | REST | No |
| Funding rates | Kraken Futures `/tickers` | REST | No |
| Open interest | Kraken Futures `/tickers` | REST | No |
| DXY proxy | floatrates.com | REST | No |
| Mempool congestion | blockchain.info | REST | No |
| Whale transactions | blockchain.info (unconfirmed TXs) | REST | No |
| Stablecoin market cap | CoinGecko `/coins/markets` | REST | No |
| Liquidation proxy | Kraken Futures basis/funding | REST | No |
| News (RSS) | CoinDesk, Cointelegraph, The Block, Bitcoin Magazine, Crypto Briefing | RSS | No |
| Reddit sentiment | r/cryptocurrency, r/bitcoin, r/ethereum, r/CryptoMarkets, r/defi | REST (JSON) | No |
| Twitter | Twitter API v2 | REST | Yes (Bearer) |
| CoinGecko trending | CoinGecko `/search/trending` | REST | No |

**All CryptoBot sentiment sources are free** (no paid API keys required). Fall back to disk-cached data when APIs are unavailable (2-hour TTL).

---

## 8. Sentiment Analysis

### AlpacaBot
**File:** `utils/sentiment.py` (447 lines)

**Market-Wide Sentiment (cached 5 min):**

| Source | Weight | Method |
|---|---|---|
| VIX (via VIXY ETF) | 0.40 | Normalized: VIX<15=bullish, >30=bearish |
| SPY trend | 0.40 | SMA crossover + momentum |
| Market breadth | 0.20 | Alpaca screener movers ratio |

**Per-Symbol Adjustment (±0.35):**

| Source | Weight | Method |
|---|---|---|
| Alpaca News API headlines | 0.40 | Keyword scoring (bullish/bearish word lists), last 4h, max 8 articles, headlines @1.0 weight, summaries @0.5 |
| Price momentum | 0.25 | Recent price action |
| Options flow | 0.35 | Volume/OI ratio, call vs put volume ratio, unusual activity detection (vol > 2× OI) |

**Output:** Single float in [-1, +1] per symbol. Used as both ensemble input (weight 0.15) and directional filter (±0.15 threshold).

---

### CryptoBot
**Files:** `core/trading_engine.py` (inline SentimentAnalyzer) + `utils/enhanced_sentiment.py` (826 lines) + `utils/news_sentiment.py` (480 lines)

**Enhanced Multi-Source System — 11 Free Sources:**

| # | Source | Cache TTL | Signal Derivation |
|---|---|---|---|
| 1 | Fear & Greed Index | 5 min | Normalized 0–100 → signal: (value-50)/50 |
| 2 | CoinGecko Global Market | 10 min | BTC dominance + market cap change |
| 3 | Kraken Funding Rates | 5 min | Contrarian: `signal = -avg_rate × 1000` |
| 4 | Kraken Open Interest | 5 min | OI change direction |
| 5 | DXY Proxy (floatrates) | 30 min | `signal = -(dxy_proxy - 1.0) × 10` |
| 6 | Mempool Congestion | 10 min | `congestion = unconfirmed_txs / 50000` |
| 7 | Stablecoin Ratio | 30 min | `signal = (ratio% - 7) / 5` |
| 8 | Top Coin Momentum | 5 min | 24h avg change of 9 major coins, `signal = avg/5` |
| 9 | Long/Short Ratio | 5 min | Basis spread (mark vs index): `signal = -basis × 20` |
| 10 | Whale Transactions | 10 min | Accumulation vs distribution heuristic from blockchain.info |
| 11 | Liquidation Proxy | 5 min | Mark/index basis × funding rate direction |

**Composite Weighting:**
```
fear_greed:     0.20    coin_momentum:   0.15
whale_flow:     0.12    funding_rates:   0.12
liquidations:   0.10    dollar_strength: 0.08
long_short:     0.08    stablecoin_ratio:0.08
mempool:        0.04    global_market:   0.03
```

**News NLP (separate module):**
- Primary: DistilBERT transformer (`distilbert-base-uncased-finetuned-sst-2-english`)
- Fallback: Keyword scoring (positive/negative word lists)
- Aggregation: Recency-weighted average (newer articles weighted more)

**Per-Coin Sentiment:**
- Keyword matching from article text → map to 16 tracked coins
- Blend: 70% coin-specific + 30% global for matched coins

**Stale Decay:** After 60 minutes without refresh, sentiment linearly decays to 0 over the next 60 minutes.

---

## 9. Execution Logic

### AlpacaBot

| Parameter | Value |
|---|---|
| Order type | Market orders (options) |
| DTE selection | Per-symbol lookup table from 150-test backtest sweep (33 symbols mapped, default DTE=2) |
| Strike selection | ATM ± MAX_OTM_PCT (4%) |
| Options quality gates | MIN_OPEN_INTEREST ≥ 50, MIN_VOLUME ≥ 10, MAX_BID_ASK_SPREAD ≤ 15% |
| Position sizing | Confidence × count scaling, graduated response multiplier |
| Half-size mode | When ML model not yet ready |
| Fees | Implicit in bid-ask spread |
| MAX_OPENS_PER_CYCLE | 1 (throttled entry) |
| BAR_INTERVAL | 600s (10 minutes) |

**Shadow Scanner:** Runs during circuit breaker pauses. Logs would-be trades without executing.

**AI Learning on Exit:** Feeds trade outcome to RL shadow agent + meta-learner for weight updates.

---

### CryptoBot

| Parameter | Spot | Futures |
|---|---|---|
| Order type | Paper/simulated (market) | Paper/simulated (market) |
| Slippage model | 5 bps | 8 bps |
| Fee rate | 0.1% (0.001) | 0.1% (0.001) |
| Partial fill probability | 10% | 10% |
| Partial fill range | 70%–100% of order | 70%–100% |
| Funding cost | — | `notional × rate_per_8h × (hours/8)` |
| Quiet hours (UTC 0–3) | 50% size reduction | 50% size reduction |
| Leverage | 1× | 2× |

**Slippage Formula:**
```
BUY: fill_price = mid × (1 + bps/10000)
SELL: fill_price = mid × (1 - bps/10000)
```

**Liquidity Gate (orderbook depth):**

| Parameter | Spot | Futures |
|---|---|---|
| MIN_DEPTH_USD | $5,000 | $15,000 |
| DEPTH_TO_TRADE_RATIO | 3× trade size | 3× trade size |
| Fail-open behavior | Configurable (default: allow) | Configurable |

**Correlation Gate:**
```
max_abs_correlation(candidate, all_open_positions) ≤ MAX_CORRELATION (0.85)
```
Using Pearson correlation on rolling 120-period returns.

**Stress Correlation Mode:** Activates on 8-trade window with -1.5% PnL — caps correlation at 0.70, limits to 8 positions.

---

## 10. Asset Selection & Universe

### AlpacaBot
**File:** `core/config.py` + `core/scanner.py`

**Active Watchlist (6 symbols):** MA, NFLX, GM, IWM, LLY, SNOW

**Scanner Universe (33 symbols in 3 tiers):**

| Tier | Symbols | Criteria |
|---|---|---|
| TIER1 (10) | High-confidence symbols | Profit Factor ≥ 1.2 |
| TIER2 (12) | Medium-confidence symbols | PF 1.0–1.2 |
| TIER3 (11) | Watch-only | Under evaluation |

**Position Limits:**

| Parameter | Value |
|---|---|
| MAX_POSITIONS | 5 |
| MAX_POSITION_PCT | 4% of balance |
| MIN_POSITION_PCT | 2% of balance |
| MAX_OPENS_PER_CYCLE | 1 |

**DTE Map:** Per-symbol optimized DTE from 150-run backtest sweep (e.g., AAPL→DTE=3, TSLA→DTE=2, etc.)

---

### CryptoBot
**File:** `core/trading_engine.py` (TradingConfig) + `utils/config.py`

**Spot Universe (18–19 pairs):**
```
BTC-USD, ETH-USD, SOL-USD, ADA-USD, AVAX-USD, DOGE-USD,
MATIC-USD, LINK-USD, LTC-USD, XRP-USD, DOT-USD, ATOM-USD,
SHIB-USD, UNI-USD, AAVE-USD, XLM-USD, NEAR-USD, PAXG-USD
+ watchlist extras: OP-USD, ARB-USD, SUI-USD
```

**Futures Universe (12–15 PI_ symbols):**
```
PI_XBTUSD, PI_ETHUSD, PI_SOLUSD, PI_ADAUSD, PI_DOGEUSD,
PI_LINKUSD, PI_AVAXUSD, PI_DOTUSD, PI_BCHUSD, PI_LTCUSD,
PI_XRPUSD, PI_ATOMUSD (+ potentially MATIC, SHIB, UNI)
```

**Position Limits:**

| Parameter | Spot | Futures |
|---|---|---|
| MAX_POSITIONS | 12 | 8 |
| PER_SYMBOL limit | Configurable | 1 per direction (long + short allowed) |

**Futures Probation System:**
- Core symbols: PI_ETHUSD, PI_XRPUSD (always active)
- Probation candidates rotate every 24 hours
- Cohort size: 6 symbols
- Promote: WR > 55% AND avg > 0.20%
- Demote: WR < 45%

**Coin Performance Tracker:** (`utils/coin_performance.py`)
- Composite score (0–100) per coin: win_rate (30pts) + avg_profit (25pts) + volatility (20pts) + trade_frequency (15pts) + sentiment (±10pts) + ML_confidence (10pts)
- Portfolio rotation: identify bottom-N underperformers, replace with top-N watchlist candidates if score 20% better

---

## 11. Unique / Differentiating Features

### AlpacaBot Only

| Feature | Description |
|---|---|
| **Options Trading** | Trades calls/puts instead of underlying — per-symbol DTE optimization from 150-run backtest sweep |
| **Graduated Response** | 4-tier loss response (100%→50%→25%→10%) instead of binary circuit breaker |
| **Direction-Specific Locking** | 3 consecutive losses in one direction → lock that direction, other stays open |
| **Rapid-Fire Detection** | 3 losses in 20 minutes → extra 10-min cooldown |
| **Rolling Cross-Symbol Loss Rate** | Last 10 trades across all symbols and days |
| **SPY Regime Gate** | Market-wide directional filter using SPY trend |
| **Phantom Position Detection** | 3 sell failures → API position verification |
| **Scanner Shadow Mode** | Scanner runs during circuit breaker pauses, logs hypothetical trades |
| **Options Quality Filters** | Min OI ≥ 50, min volume ≥ 10, max bid-ask spread ≤ 15% |
| **IV Rank & IV Percentile** | Options volatility context in indicator suite |

### CryptoBot Only

| Feature | Description |
|---|---|
| **DQN Neural Network RL** | PyTorch Double DQN with experience replay (vs AlpacaBot's tabular Q-table) |
| **RL in Live Ensemble** | RL agent participates in meta-learner with 0.15 weight (AlpacaBot keeps RL shadow-only) |
| **11-Source Sentiment** | Fear & Greed, funding rates, OI, whale tracking, liquidations, mempool, stablecoins, DXY, long/short, momentum, global market |
| **DistilBERT NLP** | Transformer-based news sentiment analysis (vs AlpacaBot's keyword scoring) |
| **Futures Trading** | Dual-market (Coinbase spot + Kraken futures) with leverage and per-direction position tracking |
| **Correlation Gate** | 120-period return correlation matrix; blocks new entries if max|corr| > 0.85 |
| **Stress Correlation Mode** | Tightens to 0.70 cap + 8 max positions during drawdown |
| **Futures Probation System** | Performance-based rotation of futures symbols every 24h |
| **Coin Performance Tracker** | Composite scoring (0–100) for automatic portfolio rotation |
| **Kelly Criterion Sizing** | Mathematical sizing with quarter-Kelly safety fraction |
| **Volatility-Scaled Stops** | Dynamic SL/TP/trail adjusted by real-time volatility regime |
| **Execution Realism Profiles** | `off` / `normal` / `strict` — configurable simulation fidelity |
| **Locked Profile System** | Persist config overrides to JSON for reproducible runs |
| **Runtime Fingerprint** | SHA256 hash of engine + config snapshot saved at every startup |
| **Direction Bias (env-tunable)** | Shift long/short trigger thresholds via env vars |
| **Quiet Hours** | UTC 0–3: automatic 50% size reduction |
| **Whale Transaction Analysis** | Heuristic accumulation/distribution detection from Bitcoin mempool |
| **Liquidation Squeeze Proxy** | Basis × funding direction → squeeze pressure signal |
| **Per-Direction Performance Pause** | Pauses LONG or SHORT independently if WR drops below 30% over last 20 trades |
| **Alerting** | Webhook + SMTP email alerts for drawdown and daily-loss thresholds |
| **Stale Profit Decay Exit** | Arm at +1.5%, decay to +0.8%, auto-close after configurable minutes |

---

## Summary Comparison Table

| Dimension | AlpacaBot | CryptoBot |
|---|---|---|
| **Asset Class** | US equity options (calls/puts) | Crypto spot + futures |
| **Exchanges** | Alpaca | Coinbase (spot) + Kraken (futures) |
| **Bar Interval** | 10-min | 1-min → aggregated 10-min |
| **ML Features** | 20 (14 price + 6 context) | 15 (10 oscillators + 5 complementary) |
| **ML Depth** | max_depth=4 | max_depth=4 (inline) / 5 (predictor) |
| **ML Min Accuracy** | 0.54 | 0.58 |
| **RL Architecture** | Q-table (243 states, 4 actions) | DQN (5 continuous → 64→64 → 5 actions) |
| **RL Mode** | Shadow only | Active in ensemble (weight 0.15) |
| **Meta-Learner Weights** | ML:0.35 Sent:0.15 Rules:0.50 | ML:0.50 Sent:0.35 RL:0.15 |
| **Signal Source** | 14-indicator scoring (≥3 needed) | ML probability thresholds |
| **Stop Loss** | -15% (options) | -2% spot / -3% futures |
| **Take Profit** | +15% (options) | +2.5% |
| **Trailing Stop** | 12% (trigger at +6%) | 1.5% (trigger at +1%) |
| **Max Positions** | 5 | 12 spot + 8 futures |
| **Risk System** | 4-tier graduated response | Kelly criterion + 3-trigger circuit breaker |
| **Sentiment Sources** | 4 (VIX, SPY, news, options flow) | 11 free APIs + DistilBERT NLP |
| **Unique Strength** | Options-specific (DTE opt, IV, spread) | Multi-exchange, correlation mgmt, DQN RL |
