"""
Retry decorator with exponential backoff for API calls.

Used for:
- Coinbase spot price fetches
- Kraken futures price fetches
- RSS news fetches
- Any network call that might transiently fail
"""
from __future__ import annotations
import functools
import time
import logging
from typing import Optional, Callable, Type, Tuple

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    Decorator: retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        backoff_factor: Multiply delay by this after each retry
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(attempt, exception, delay) called before each retry

    Usage:
        @retry_with_backoff(max_retries=3, base_delay=0.5)
        def fetch_price(client, pair):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(attempt + 1, e, delay)
                        else:
                            logger.debug(
                                f"Retry {attempt + 1}/{max_retries} for "
                                f"{func.__name__}: {e} (waiting {delay:.1f}s)"
                            )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        # Final attempt failed
                        logger.warning(
                            f"{func.__name__} failed after {max_retries + 1} "
                            f"attempts: {last_exception}"
                        )

            # Re-raise or return None depending on the function's pattern
            # Most price fetchers return None on failure, so we keep that pattern
            return None

        return wrapper
    return decorator
