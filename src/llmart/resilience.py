"""LLM call resilience: exponential backoff retry via tenacity."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("llmart.resilience")

F = TypeVar("F", bound=Callable[..., object])

# Common transient exceptions from LLM providers
_TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
)

try:
    import openai

    _TRANSIENT_EXCEPTIONS = (
        *_TRANSIENT_EXCEPTIONS,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.APIConnectionError,
    )
except ImportError:  # pragma: no cover
    pass

try:
    import anthropic

    _TRANSIENT_EXCEPTIONS = (
        *_TRANSIENT_EXCEPTIONS,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
    )
except ImportError:  # pragma: no cover
    pass


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts for observability."""
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "LLM call retry attempt=%d exc=%s",
        attempt,
        type(exc).__name__ if exc else "unknown",
    )


def llm_retry(fn: F) -> F:  # noqa: UP047
    """Decorator: exponential backoff for LLM API calls (max 3 retries)."""
    decorated = retry(
        retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=_log_retry,
        reraise=True,
    )(fn)
    return decorated
