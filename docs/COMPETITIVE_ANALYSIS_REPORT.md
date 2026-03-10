# Competitive Analysis: AlpacaBot vs CryptoBot — Full Technical Survey

> **Generated**: 2025 | **Scope**: All core engine files, config, ML, RL, sentiment, risk, execution, backtesting  
> **Methodology**: Line-by-line code analysis — every parameter, formula, and gate extracted verbatim.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Signal Generation](#2-signal-generation)
3. [ML Model](#3-ml-model)
4. [RL Agent](#4-rl-agent)
5. [Meta-Learner Ensemble](#5-meta-learner-ensemble)
6. [Risk Management](#6-risk-management)
7. [Market Regime Detection](#7-market-regime-detection)
8. [Data Sources & Sentiment Analysis](#8-data-sources--sentiment-analysis)
9. [Execution Model](#9-execution-model)
10. [Position Sizing](#10-position-sizing)
11. [Asset Selection & Watchlist](#11-asset-selection--watchlist)
12. [Timeframe & Cycle Structure](#12-timeframe--cycle-structure)
13. [Backtesting Infrastructure](#13-backtesting-infrastructure)
14. [Unique Features](#14-unique-features)

---

## 1. Architecture Overview

| Dimension | AlpacaBot (`C:\AlpacaBot\`) | CryptoBot (`C:\Bot\`) |
|---|---|---|
| **Asset Class** | US equity options (calls & puts) | Cryptocurrency spot + perpetual futures |
| **Exchanges** | Alpaca (paper & live) | Coinbase (spot), Kraken Futures (perpetuals) |
| **Language** | Python 3 | Python 3 |
| **Key Libraries** | scikit-learn, numpy, alpaca-py SDK | scikit-learn, numpy, pandas, PyTorch, requests |
| **File Structure** | Modular: separate files per concern (`core/`, `utils/`, `tools/`) | Monolithic engine: `trading_engine.py` = 3122 lines containing TradingConfig, MarketData, MLModel, SentimentAnalyzer, RiskManager, TradingBot all inline; separate files for RL/meta/sentiment |
| **Initial Balance** | $100,000 (paper) | $2,500 spot + $2,500 futures (paper) |
| **Dashboard** | Flask on port 5555 (`dashboard.py`, `unified_dashboard.py`) | Flask on port 5001 |
| **State Persistence** | JSON positions + risk state + CSV trade log | JSON positions + risk state + CSV trade log + RL `.pt` weights |
| **Runtime Fingerprint** | None | SHA256 of engine file + config + model files at startup |

**Shared AI Stack** (same architecture, ported from CryptoBot → AlpacaBot):
```
Price Bars → 14-15 Indicators → Bull/Bear Score
                                      ↓
              ML Model (GradientBoosting) → Direction + Confidence
                                      ↓
              Sentiment Analyzer → Market Bias (-1 to +1)
                                      ↓
              RL Agent (Shadow) → Position Size Multiplier (observe only)
                                      ↓
              Meta-Learner Ensemble → Weighted Score → Trade / Skip
```

---

## 2. Signal Generation

### AlpacaBot — 14 Indicators → Max Score 16

Scored into `bull` and `bear` counters. Direction = call if `bull ≥ 3 AND bull > bear + 1`; put if `bear ≥ 3 AND bear > bull + 1`.

| # | Indicator | Bullish Trigger | Points | Bearish Trigger | Points |
|---|-----------|----------------|--------|-----------------|--------|
| 1 | RSI(14) | < 25 | +2 | > 75 | +2 |
|   |   | 35-55 | +1 | 50-65 | +1 |
| 2 | MACD Histogram | > 0 (+1), > 0.1 (+1) | +2 | < 0 (+1), < -0.1 (+1) | +2 |
| 3 | Stochastic %K | < 20 | +1 | > 80 | +1 |
| 4 | Bollinger Band Position | < 0.10 (+2), < 0.30 (+1) | +2 | > 0.90 (+2), > 0.70 (+1) | +2 |
| 5 | ATR Normalized | > 0.005 & bull leads | +1 | > 0.005 & bear leads | +1 |
| 6 | CCI(20) | < -100 | +1 | > 100 | +1 |
| 7 | ROC(10) | > 0.3 | +1 | < -0.3 | +1 |
| 8 | Williams %R | < -80 | +1 | > -20 | +1 |
| 9 | Volatility Ratio | > 1.3 & leading | +1 | > 1.3 & leading | +1 |
| 10 | Z-Score | < -2.0 | +1 | > 2.0 | +1 |
| 11 | Trend Strength | > 25 & leading | +1 | > 25 & leading | +1 |
| 12 | Price Change 1-bar | > 0.001 | +1 | < -0.001 | +1 |
| 13 | Price Change 5-bar | > 0.003 | +1 | < -0.003 | +1 |

**Quality Gates (sequential — all must pass):**
1. Direction enable flags (call_enabled / put_enabled from graduated response)
2. SPY regime filter: calls blocked in bear market (SPY < 20d MA), puts blocked in bull
3. Sentiment direction filter: calls blocked if sentiment < -0.15, puts blocked if sentiment > 0.15
4. ML agreement gate: ML predicted direction must match signal direction
5. ML confidence gate: `MIN_ML_CONFIDENCE = 0.55`
6. Meta-learner ensemble gate: `meta_learner.should_trade()` must return True

**Scanner Universe** (unique to AlpacaBot):
- `MarketScanner` class scans up to 50 symbols per cycle from a 33-symbol universe
- Universe is tiered by backtested profitability:
  - TIER1: AVGO, PYPL, NFLX, NKE, SBUX, F, COST, SHOP, IWM, META (PF ≥ 1.20)
  - TIER2: JPM, AMZN, DIS, SMH, RIVN, XLK, GM, SOFI, QCOM, MARA, MA, AAPL (PF 1.0-1.2)
  - TIER3: WMT, V, LLY, INTC, ABBV, COIN, JNJ, AMD, MU, UBER, NVDA (marginal)
  - Eliminated: SNOW, CVX, ARKK, MSFT, TSLA, GOOGL, SPY, QQQ (negative P&L)
- **Momentum sanity check**: Won't buy puts on stocks up >1.5% over 20 bars AND still rising (and vice versa)

### CryptoBot — 15 Indicators → Max Score 15

Uses `compute_all_indicators()` from `technical_indicators.py`. Signal scoring feeds into ML confidence.

| # | Indicator | Implementation |
|---|-----------|---------------|
| 1 | RSI(14) | Same formula as AlpacaBot |
| 2 | MACD Histogram | Same |
| 3 | Stochastic %K(14,3) | Same |
| 4 | CCI(20) | Same |
| 5 | ROC(10) | Same |
| 6 | Momentum(10) | `prices[-1] - prices[-period]` |
| 7 | Williams %R(14) | Same |
| 8 | Ultimate Oscillator(7,14,28) | Three-period weighted average |
| 9 | TRIX(15) | Triple EMA rate of change |
| 10 | CMO (Chande Momentum, 14) | `(sum_up - sum_down) / (sum_up + sum_down) × 100` |
| 11 | ATR (normalized) | Same |
| 12 | Trend Strength(20) | Same |
| 13 | Bollinger Band Position(20) | Same |
| 14 | Mean Reversion Z-Score(20) | Same |
| 15 | Volatility Ratio | Same |

**Quality Gates (sequential):**
1. ML confidence: `MIN_ML_CONFIDENCE = 0.66` (stricter than AlpacaBot's 0.55)
2. Ensemble score: `MIN_ENSEMBLE_SCORE = 0.62`
3. Trend verification: `trend == "UP"` for LONG, `trend == "DOWN"` for SHORT
4. RSI filter: `MAX_RSI_LONG = 62` (don't long overbought), `MIN_RSI_SHORT = 38` (don't short oversold)
5. SIDE market filter: skip if no clear trend
6. Counter-trend ML override: threshold = 0.92 (nearly disabled — allows against-trend only with extreme confidence)
7. Direction performance tracker: auto-pause direction at <30% WR over last 20 trades
8. Direction mode filter: configurable `both` / `short_only` / `long_only`
9. Symbol blacklist check
10. Per-symbol auto-pause: 3 consecutive losses = 2-hour pause

---

## 3. ML Model

| Parameter | AlpacaBot | CryptoBot |
|-----------|-----------|-----------|
| **Algorithm** | GradientBoostingClassifier | GradientBoostingClassifier |
| **n_estimators** | 200 | 200 |
| **max_depth** | 4 | 4 |
| **learning_rate** | 0.05 | 0.05 |
| **subsample** | 0.8 | 0.8 |
| **min_samples_split** | 20 | (default) |
| **min_samples_leaf** | 10 | 10 |
| **Features** | 20 (14 price + 6 options-context) | 15 (all technical indicators) |
| **Training Target** | Price direction over next 6 bars (1 hour) | Price direction with configurable thresholds per timeframe |
| **Flat Filter** | Moves < 0.05% excluded | Per-timeframe thresholds |
| **Min Accuracy** | 54% (code) / 51% (config) | 58% OOS |
| **Validation** | TimeSeriesSplit (2-5 folds) | TimeSeriesSplit |
| **Class Balance** | Undersampling + balanced sample weights | Balanced sample weights |
| **Model Versioning** | Last 5 versions, rollback support | Single model |
| **Retrain Schedule** | Every 6 hours (`MODEL_RETRAIN_HOURS`) | Periodic (runtime) |
| **Bootstrap** | Fetches 30d bars for 20 symbols via API | Trains on hourly + 10-min + 1-min candles |
| **Prediction Output** | `{direction, confidence, up_prob, down_prob}` | `{direction, confidence, up_prob, down_prob}` |
| **Storage** | joblib serialization | joblib serialization |

**AlpacaBot Extra Features (6 options-context):**
- `hour_sin`, `hour_cos` — time-of-day cyclical encoding
- `day_of_week` — weekday number
- `intraday_range` — session high-low range
- `gap_open` — overnight gap %
- `vol_regime` — ATR-based volatility regime (0/1/2)

---

## 4. RL Agent

| Parameter | AlpacaBot | CryptoBot |
|-----------|-----------|-----------|
| **Algorithm** | Tabular Q-Learning | Deep Q-Network (DQN) |
| **Network** | Q-table (243 entries) | 2-layer NN (64 units, LayerNorm + ReLU) |
| **State Space** | 5 dims × 3 buckets = 243 states | 5 continuous dims normalized [0,1] or [-1,1] |
| **State Dims** | RSI bucket, trend bucket, vol_regime, sentiment bucket, ML confidence bucket | sentiment, volatility/0.1, trend/50, rsi/100, ml_confidence |
| **Actions** | 4: SKIP(0.0×), SMALL(0.5×), MEDIUM(1.0×), LARGE(1.5×) | 5: [0.25×, 0.5×, 1.0×, 1.5×, 2.0×] |
| **Learning Rate** | 0.1 | 0.001 |
| **Discount (γ)** | 0.95 | 0.95 |
| **Exploration (ε)** | 0.15, decay 0.9995 → min 0.05 | 0.15, decay 0.9995 → min 0.05 |
| **Replay Buffer** | None (tabular) | 10,000 capacity, batch 32 |
| **Target Network** | None | Synced every 50 updates |
| **Double DQN** | No | Yes (policy net selects, target net evaluates) |
| **Loss Function** | N/A | Huber (SmoothL1) + gradient clipping max_norm=1.0 |
| **Reward Shaping** | PnL% × sizing_mult × 100, clipped [-5, 5] | Amplify >0.5 (1.5×), dampen <0.1 (0.5×) |
| **Per-Coin Exploration** | No | Yes (bonus for less-traded coins) |
| **Migration** | N/A | Can seed replay buffer from old Q-table format |
| **Mode** | **Shadow only** | **Shadow only** (with optional live size control) |
| **Promotion** | Must outperform baseline for 50 consecutive trades | Same concept, but with `rl_live_size_control` flag |
| **Persistence** | JSON Q-table | PyTorch `.pt` + JSON metadata |

**Key Code Comment** (from AlpacaBot `rl_agent.py`):
> "The crypto bot's RL performed WORSE than baseline (+$39 vs +$49 over 200 trades)"

Both bots keep RL in shadow mode — it observes and learns but does NOT affect live trades unless explicitly promoted.

---

## 5. Meta-Learner Ensemble

| Parameter | AlpacaBot | CryptoBot |
|-----------|-----------|-----------|
| **Sources** | ml_model, sentiment, rule_score | ml_model, sentiment, rl_agent |
| **Default Weights** | ml=0.35, sentiment=0.15, rules=0.50 | ml=0.50, sentiment=0.35, rl=0.15 |
| **Weight Floor** | 0.10 per source | 0.10 per source |
| **RL Safeguard** | N/A (RL not in ensemble) | RL only included if `|prediction - 0.5| > 0.1` |
| **Confidence Threshold** | 0.68 (raised from 0.62) | buy=0.55, sell=0.45 (dynamic) |
| **Min Rule Score** | 6 (raised from 4) | N/A |
| **Dynamic Adjustment** | Tightens on losing streaks (≥3 consecutive or ≥60% loss rate), loosens when WR>60%, mean-reverts | Thresholds shift by sentiment × 0.05 |
| **Weight Updates** | Per-source accuracy over rolling 50 predictions | Same mechanism |
| **Input Normalization** | `rule_normalized = score/16`, `ml_for_ensemble = ml_direction` (flipped for puts), `sent_for_ensemble = (sentiment+1)/2` | Direct confidence values |

---

## 6. Risk Management

### AlpacaBot — Graduated Response System (NOT binary)

| Tier | Consecutive Losses | Size Multiplier | Min Score | Cooldown | Special |
|------|-------------------|-----------------|-----------|----------|---------|
| NORMAL | 0-2 | 100% | 3 | None | — |
| CAUTION | 3-4 | 50% | 7 | 5 min | Block losing direction |
| DEFENSIVE | 5-7 | 25% | 9 | 10 min | — |
| PROBE | 8+ | 10% | 10 | 15 min | — |
| HARD STOP | N/A | 0% | — | Rest of day | Only at -5% daily loss |

**Additional Rules:**
- **Direction-specific lock**: 3 consecutive losses in calls → calls locked, puts still open (and vice versa). Unlocks on first win in that direction.
- **Rapid-fire detection**: 3 losses in 20 minutes = extra 10-min cooldown
- **Per-symbol pause**: 3 consecutive losses on same symbol = paused
- **Rolling loss rate**: Tracked over last 10 trades → meta-learner threshold adjustment
- **New day reset**: All counters (daily P&L, consecutive losses, direction locks, hard stop) reset at midnight

**Exit Rules:**
| Exit Type | Trigger |
|-----------|---------|
| Stop Loss | -15% of premium |
| Take Profit | +15% of premium |
| Trailing Stop | 12% from peak, activated at +6% profit |
| DTE Exit | Close when DTE ≤ `MIN_DTE_EXIT` (avoid assignment) |
| Max Hold | 1-7 days (DTE-dependent) |

### CryptoBot — Binary Circuit Breaker + Kelly Sizing

| Parameter | Spot | Futures |
|-----------|------|---------|
| Stop Loss | -2% (-3% config) | -2.5% (-3% config) |
| Take Profit | +2.5% | +4% |
| Trailing Stop | 1.5% from peak (armed at +1% profit) | Same |
| Stale Profit Decay | Armed at 1.5%, decays 0.8%, 45 min | Armed at 1.5%, decays 0.8%, 25 min |
| Max Hold | 4 hours | 3 hours |
| Flat Band | ±0.5% for max hold exit | Same |

**Circuit Breakers:**
- 5 consecutive losses → 60 min pause
- -5% daily loss → halt
- -10% drawdown from peak → halt
- **Fast guards**: 3 losses in 5-trade window at -2% P/L → 30 min pause

**Position Sizing**: Kelly Criterion
```
kelly = max(0.01, 2 * confidence - 1) × vol_adjust × position_factor
```

**Quiet Hours**: 0-3 UTC → 50% size reduction

---

## 7. Market Regime Detection

### AlpacaBot
- **SPY Regime Filter**: Computes SPY 20-day MA. If SPY < 20d MA → "bear market" → calls blocked. If SPY > 20d MA → "bull market" → puts blocked. Uses VIXY ETF as VIX proxy.
- **VIX Signal**: VIXY price mapped to VIX estimate: price < $15 = low vol (bullish), price > $30 = high vol (bearish), in between = neutral. Weight: 40% of market-wide sentiment.
- **Volatility Regime**: ATR-based regime classification (0/1/2) fed as ML feature.

### CryptoBot
- **Trend Detection**: Based on indicator consensus — `"UP"`, `"DOWN"`, or `"SIDE"`. SIDE trades are blocked.
- **Direction Bias**: Configurable `neutral` / `short_lean` / `long_lean` with bias strength (default 0.04, max 0.15). Adjusts confidence thresholds.
- **Volatility Scaling**: `VOL_WINDOW = 20`, base stops/TPs scale with realized volatility (base_stop=1.2%, base_tp=2.0%, base_trail=1.2%).
- **Direction Performance**: Auto-pause direction if WR < 30% over last 20 trades.

---

## 8. Data Sources & Sentiment Analysis

### AlpacaBot — 5 Equity-Specific Sources

| Source | API | Weight | Signal |
|--------|-----|--------|--------|
| News Headlines | Alpaca News API | 40% per-symbol | Keyword match (bullish/bearish word lists) |
| VIX (via VIXY ETF) | Alpaca Market Data | 40% market-wide | Price → VIX estimate → bullish/bearish |
| SPY Trend | Alpaca Market Data | 40% market-wide | 5d + 1d return vs 20d MA |
| Market Breadth | Alpaca Screener | 20% market-wide | Gainers vs losers count |
| Options Flow | Alpaca Options API | 35% per-symbol | Volume vs OI ratio → unusual activity detection |

**Per-Symbol Blend**: `40% news + 25% momentum + 35% options_flow`, clamped to ±0.35  
**Stale Decay**: Sentiment decays toward 0 after 60 minutes without refresh

### CryptoBot — 11 Free Crypto-Specific Sources

| # | Source | API | TTL | Signal Logic |
|---|--------|-----|-----|-------------|
| 1 | Fear & Greed Index | alternative.me | 30 min | 0-100 → normalized. <25 extreme fear (bullish), >75 extreme greed (bearish) |
| 2 | Global Crypto Market | CoinGecko | 30 min | BTC dominance, total market cap, 24h volume |
| 3 | Funding Rates | Kraken Futures | 5 min | Negative = shorts pay (contrarian bullish). 6 perps tracked. |
| 4 | Open Interest | Kraken Futures | 10 min | Change from baseline → positioning signal |
| 5 | Dollar Strength (DXY proxy) | FloatRates | 30 min | EUR/USD + GBP/USD basket. Strong dollar = bearish crypto |
| 6 | BTC Mempool | blockchain.info | 10 min | Unconfirmed txs > 50k = congestion (extreme moves) |
| 7 | Stablecoin Ratio | CoinGecko | 30 min | Stable MC / Total MC. >10% = dry powder (bearish fear), <5% = deployed (bullish) |
| 8 | Top Coin Momentum | CoinGecko | 5 min | 24h % change for BTC, ETH, SOL, etc. Avg ±5% → ±1.0 |
| 9 | Long/Short Ratio | Kraken Futures | 5 min | Mark vs index price spread. Positive basis = contrarian bearish |
| 10 | Whale Transactions | blockchain.info | 10 min | Large BTC txs (>10 BTC). Many inputs→few outputs = accumulation (bullish) |
| 11 | Liquidation Proxy | Kraken Futures | 5 min | Basis + funding direction → squeeze detection |

**Composite Weights:**
```
fear_greed:      0.20    coin_momentum:    0.15
funding_rates:   0.12    whale_flow:       0.12
liquidations:    0.10    long_short:       0.08
dollar_strength: 0.08    stablecoin_ratio: 0.08
mempool:         0.04    global_market:    0.03
```

**All sources are 100% free, no API keys required.** In-memory cache with TTL + disk fallback (2-hour TTL).

---

## 9. Execution Model

### AlpacaBot
- **Order Type**: Market orders for all entries/exits
- **Contract Selection**: `OptionsHandler` scores contracts by: DTE proximity to target, open interest (min 50), bid-ask spread, strike distance from ATM (slightly OTM preferred, max 4% OTM)
- **Per-Symbol DTE Optimization**: 33 symbols mapped to backtested optimal DTE (1-30 days)
- **Slippage**: Not modeled (relies on market order fills via Alpaca)
- **Rate Limiting**: 0.4s delay between symbol scans in scanner
- **Cooldown**: `COOLDOWN_BARS × BAR_INTERVAL_SEC` per underlying after exit

### CryptoBot
- **Order Type**: Simulated market fills (paper trading) with execution realism layer
- **Slippage Model**: 
  - Spot: 5-8 bps (configurable `SLIPPAGE_BPS`)
  - Futures: 3-5 bps
  - Applied directionally: entry slippage against you, exit slippage against you
- **Partial Fills**: 10% probability of partial fill
- **Funding Costs**: 0.01% per 8 hours for futures positions
- **Liquidity Gate**: Checks orderbook depth before entry. Skip if depth < trade size.
- **Correlation Gate**: `MAX_CORRELATION = 0.75` — won't open two highly correlated positions
- **Fee Model**: 0.1% per trade (spot), applied to balance at entry

---

## 10. Position Sizing

### AlpacaBot
```
max_spend = balance × MAX_POSITION_PCT(4%) × conf_factor(0.5-1.0) × count_factor(0.4-1.0)
contracts = max_spend / (premium × 100)
```
- `conf_factor`: Scales 0.5-1.0 based on ML confidence
- `count_factor`: Reduces size as position count grows (5 positions → 0.4×)
- Capped by portfolio risk limit
- Further scaled by graduated response tier (100% / 50% / 25% / 10%)

### CryptoBot
```
kelly = max(0.01, 2 × confidence - 1) × vol_adjust × position_factor
size = balance × kelly
```
- **Scaled sizing multipliers**: 1-3× based on setup quality score (0-6 scale)
- **Quiet hours**: 0-3 UTC → 50% size reduction
- **Futures leverage**: `LEVERAGE = 2×` → capital requirement divided by leverage
- **Min trade size**: `max($10, balance × MIN_POSITION_PCT)`
- **RL live sizing** (optional): When `rl_live_size_control = True`, RL multiplier (0.25-2.0×) applied after base sizing. Clamped by `rl_live_size_min_mult` and `rl_live_size_max_mult`.

---

## 11. Asset Selection & Watchlist

### AlpacaBot
- **Universe**: 33 US equities, tiered by 365-day backtest results
- **Discovery Method**: DTE sweep backtester tested 61 stocks × all available DTEs
- **Selection Criteria**: Profit Factor ≥ 1.0 over 1 year of simulated options trading
- **Top Performers**: AVGO (+116%, PF 1.34), PYPL (+105%, PF 2.13), NFLX (+102%, PF 1.57)
- **Eliminated**: TSLA, GOOGL, SPY, QQQ, MSFT (all negative P&L in backtest)
- **Dynamic Scanning**: `MarketScanner` scans universe each cycle, ranks by signal strength, feeds top signals to engine
- **Per-Symbol DTE Map**: Each of the 33 symbols has an individually backtested optimal DTE (e.g., AVGO=2, SBUX=9, F=30, IWM=1)

### CryptoBot
- **Spot Watchlist**: 19 crypto pairs: BTC, ETH, SOL, ADA, AVAX, DOGE, MATIC, LTC, LINK, SHIB, XRP, DOT, UNI, ATOM, XLM, AAVE, OP, ARB, SUI
- **Futures**: 15 Kraken perpetuals (PI_XBTUSD, PI_ETHUSD, etc.)
- **Probation System** (futures):
  - Core symbols: PI_ETHUSD, PI_XRPUSD (always active)
  - Probation candidates rotate every 24 hours
  - Cohort size: 6 symbols
  - Promote at >55% WR, demote at <45% WR
- **Symbol Blacklist**: Runtime blacklist for persistently losing symbols

---

## 12. Timeframe & Cycle Structure

### AlpacaBot
```
BAR_INTERVAL = 600 seconds (10-minute bars)
LOOKBACK_BARS = 50 (≈ 8.3 hours of history)
RISK_CHECK_INTERVAL = 15 seconds (exit checks)
SCANNER_INTERVAL = 10 minutes
MODEL_RETRAIN_HOURS = 6
```

**Main Loop Flow**:
1. Wait for market open (`_is_market_open()`)
2. Fetch 10-min bars for all positions + scanned symbols
3. Check exits (stop loss, TP, trailing, DTE, max hold)
4. Run scanner → rank signals
5. Apply quality gates → execute best signals
6. ML retrain check (every 6 hours)
7. Sleep 10 minutes → repeat

### CryptoBot
```
CHECK_INTERVAL = 60 seconds (price collection)
TRADE_CYCLE_INTERVAL = 10 cycles (trade every 10 minutes)
RISK_CHECK_INTERVAL = 60 seconds (positions checked every cycle)
```

**Main Loop Flow** (`run()` method):
1. Fetch prices (every cycle, 60s)
2. Check risk / exits (every cycle)
3. Shadow system risk check (every cycle)
4. Check alerts (every cycle)
5. On trade cycle (every 10th iteration):
   - Ensure model ready → maybe retrain
   - Generate signals → execute signals
6. Log status (every 5 cycles)
7. Save state (every 10 cycles)
8. Persist RL shadow report (every 10 cycles)

**Key Difference**: CryptoBot runs 24/7 (crypto markets never close). AlpacaBot only runs during US market hours (9:30 AM - 4:00 PM ET, Mon-Fri).

---

## 13. Backtesting Infrastructure

### AlpacaBot — 8 Backtesting Tools

| Tool | Purpose | Key Details |
|------|---------|-------------|
| `tools/backtest.py` | Core options backtester | Black-Scholes approximated delta for options P&L. ATM premium ≈ `price × IV × √(DTE/365) × 0.4`. Delta starts at 0.50, adjusts by price move. Theta decay modeled. |
| `tools/backtest_v2.py` | Improved version | (Iteration on v1) |
| `tools/backtest_v3.py` | Latest version | (Further refinements) |
| `tools/backtest_scalp.py` | Scalp-specific testing | Optimized for short-hold options trades |
| `tools/backtest_scanner.py` | Scanner evaluation | Tests scanner signal quality across universe |
| `tools/backtest_dte_sweep.py` | DTE optimization sweep | Tests 61 symbols × all available DTEs. Found optimal DTE per symbol. Universe of daily-expiry vs weekly-expiry symbols. Uses API probe data or defaults. |
| `tools/backtest_mtf.py` | Multi-timeframe testing | Tests across multiple bar intervals |
| `tools/backtest_stocks.py` | Stock-only backtester | Tests underlying equity signals without options simulation |

**Options P&L Simulation** (since historical options data isn't available):
```python
IV = annualized std(log returns) over 20 days
premium = price × IV × √(T) × 0.4
delta ≈ 0.50 + price_move% × 5, clamped [0.05, 0.95]
theta_decay = time_held / total_DTE × premium × 0.4
option_PnL = delta_PnL - theta_decay
```

### CryptoBot — Backtester Module

| File | Lines | Purpose |
|------|-------|---------|
| `cryptotrades/utils/backtester.py` | 1166 | Full replay engine with dual-timeframe simulation |
| `tools/backtest_6mo.py` | Variant | 6-month focused testing |
| `tools/backtests/backtest_6mo_v10_baseline.py` | Variant | Baseline comparison version |
| `tools/backtests/backtest_new_coins.py` | Variant | New coin evaluation |
| `backtest_harness.py` | Harness | Test runner wrapper |

**Features**:
- Simulates 1-minute risk monitoring + 10-minute trade decisions
- Models slippage, partial fills, funding costs via `execution_model.py`
- Reports: total return, win rate, max drawdown, Sharpe ratio, profit factor, avg trade duration, win/lose streaks
- Trade breakdown by exit reason

---

## 14. Unique Features

### AlpacaBot Only

| Feature | Details |
|---------|---------|
| **Options Trading** | Calls and puts on equities — unique instrument class vs crypto spot/futures |
| **Options Handler** | Contract selection by DTE, OI, bid-ask, moneyness. Per-symbol optimal DTE from backtest sweep. |
| **Per-Symbol DTE Optimization** | 33 symbols each have individually backtested optimal DTE (1-30 days). Data from 365-day DTE sweep with 150 test runs. |
| **Graduated Response** | 4-tier progressive risk reduction (not binary circuit breaker) with direction-specific locking |
| **Direction-Specific Locking** | Locks only the losing direction (calls OR puts), keeps the other open |
| **Rapid-Fire Detection** | 3 losses in 20 minutes → extra cooldown |
| **Options Flow Sentiment** | Detects unusual options volume/OI ratio across the chain (call-heavy = bullish, put-heavy = bearish) |
| **SPY Regime Filter** | Market-wide direction filter based on SPY vs 20d MA |
| **Scanner Universe** | Dynamic opportunity ranking across 33-symbol backtested universe |
| **Momentum Sanity Check** | Won't short stocks ripping up (>1.5% over 20 bars AND accelerating) |
| **Black-Scholes Options Sim** | Backtester approximates options P&L via delta/theta without historical chain data |
| **Model Versioning** | Keeps last 5 ML models with rollback capability |

### CryptoBot Only

| Feature | Details |
|---------|---------|
| **Dual Market** | Spot + futures simultaneously with separate balance tracking |
| **Perpetual Futures** | 2× leverage, funding cost modeling (0.01%/8h), Kraken API |
| **DQN RL Agent** | Deep Q-Network with experience replay (10k buffer), target network, double DQN, Huber loss — vs AlpacaBot's tabular Q-table |
| **11 Free Sentiment Sources** | Fear & Greed, CoinGecko, funding rates, OI, DXY proxy, mempool, stablecoin ratio, coin momentum, long/short ratio, whale tracking, liquidation proxy — all 100% free |
| **Whale Transaction Tracking** | Monitors BTC blockchain for >10 BTC transactions, classifies accumulation vs distribution patterns |
| **Liquidation Squeeze Detector** | Combines mark-index basis + funding direction to identify short/long squeeze potential |
| **Execution Realism** | Slippage model (3-8 bps), partial fill simulation (10% chance), funding costs |
| **Futures Probation System** | Symbols rotate 24h, promote at >55% WR, demote at <45% WR |
| **Correlation Gate** | Won't open positions with >0.75 correlation |
| **Liquidity Gate** | Orderbook depth check before entry |
| **Kelly Criterion Sizing** | `kelly = max(0.01, 2×conf - 1) × vol_adjust × position_factor` |
| **Stale Profit Decay** | Take-profit threshold decays over time if position goes flat |
| **Quiet Hours** | 0-3 UTC auto 50% size reduction |
| **Direction Bias System** | Configurable neutral/short_lean/long_lean with strength parameter |
| **Locked Profile System** | Config overrides from JSON for reproducibility |
| **Runtime Fingerprint** | SHA256 hash of engine + config + model files at startup |
| **Per-Coin RL Exploration Bonus** | DQN incentivizes trading less-explored coins |
| **Shadow Dual-Book** | "baseline" book (mirrors actual trades) + "rl" book (uses RL sizing), both tracked with identical fill/slippage model |

---

## Comparative Summary Table

| Category | AlpacaBot | CryptoBot |
|----------|-----------|-----------|
| **Market** | US equity options (Mon-Fri 9:30-4) | Crypto spot+futures (24/7) |
| **Capital** | $100,000 paper | $5,000 paper ($2,500 spot + $2,500 futures) |
| **Instruments** | Calls/puts on 33 stocks | 19 spot pairs + 15 perpetual futures |
| **Indicators** | 14 → max score 16 | 15 → max score 15 |
| **ML Threshold** | 55% confidence, 54% accuracy | 66% confidence, 58% accuracy |
| **RL Sophistication** | Tabular Q-table (243 states, 4 actions) | DQN (continuous state, 5 actions, replay buffer, target net) |
| **Sentiment Sources** | 5 (Alpaca API-dependent) | 11 (100% free, no keys) |
| **Risk System** | Graduated 4-tier + direction locks | Binary circuit breaker + Kelly sizing |
| **Position Sizing** | Balance × 4% × confidence × count_factor × tier | Kelly criterion × vol_adjust × quality multiplier |
| **Max Positions** | 5 | 15 spot + 10 futures |
| **Execution Realism** | Direct API market orders | Simulated with slippage, partial fills, funding |
| **Backtesting** | 8 specialized tools | 1 main + 4 variants |
| **Unique Strength** | Options-specific features (DTE optimization, chain scoring, delta simulation) | Breadth of sentiment data, execution realism, DQN RL, dual-market |

---

*End of survey. All values extracted directly from source code.*
