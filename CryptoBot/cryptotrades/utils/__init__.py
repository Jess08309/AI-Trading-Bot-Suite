"""Utility modules for AI-driven trading bot."""

from .news_sentiment import (
    fetch_crypto_news, fetch_crypto_news_rss, get_news_sentiment,
    get_coin_sentiment, get_pair_sentiment,
)
from .performance_tracker import PerformanceTracker
try:
    from .rl_agent import RLTradingAgent
except ImportError:
    RLTradingAgent = None
from .meta_learner import MetaLearner
from .coin_performance import CoinPerformanceTracker
from .technical_indicators import compute_all_indicators
from .feature_engine import FeatureEngine
from .market_predictor import MarketPredictor
from .position_sizer import PositionSizer
from .correlation_tracker import CorrelationTracker
from .circuit_breaker import CircuitBreaker
from .config import config
from .retry import retry_with_backoff
from . import alerting

__all__ = [
    'CoinPerformanceTracker',
    'fetch_crypto_news',
    'fetch_crypto_news_rss',
    'get_news_sentiment',
    'get_coin_sentiment',
    'get_pair_sentiment',
    'PerformanceTracker',
    'RLTradingAgent',
    'MetaLearner',
    'compute_all_indicators',
    'FeatureEngine',
    'MarketPredictor',
    'PositionSizer',
    'CorrelationTracker',
    'CircuitBreaker',
    'config',
    'retry_with_backoff',
    'alerting',
]
