"""
Retry utility — exponential backoff for external API calls.

Usage:
    from retry_utils import async_retry

    @async_retry(max_retries=3, base_delay=1.0, retryable_exc=(openai.APIConnectionError, openai.RateLimitError))
    async def call_llm(...):
        ...
"""

import asyncio
import logging
from functools import wraps
from typing import Type, Tuple

logger = logging.getLogger(__name__)


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exc: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator: retry an async function with exponential backoff.

    Args:
        max_retries: Maximum retry attempts (0 = no retry).
        base_delay: Initial delay in seconds; doubles each attempt.
        max_delay: Cap on delay between retries.
        retryable_exc: Exception types that trigger a retry.
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1 + max_retries):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exc as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"{fn.__name__} retry {attempt}/{max_retries} after {delay}s: {exc}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{fn.__name__} exhausted {max_retries} retries: {exc}"
                        )
            raise last_exc

        return wrapper
    return decorator