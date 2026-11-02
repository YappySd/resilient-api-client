"""Pure helper functions for retry timing (no network, trivial to test)."""
from __future__ import annotations

from typing import Callable


def backoff_delay(attempt: int, base: float, cap: float, rng: Callable[[], float]) -> float:
    """Exponential backoff with 'full jitter': a random value in [0, min(cap, base * 2**attempt)].

    attempt is zero-based. Jitter prevents many clients from retrying at the same instant.
    """
    return rng() * min(cap, base * (2**attempt))


def parse_retry_after(value: str | None, max_wait: float) -> float | None:
    """Read a numeric Retry-After header (seconds), capped at max_wait.

    Returns None when the header is missing or not a plain number of seconds
    (the HTTP-date form is intentionally not supported, so the caller falls back to backoff).
    """
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, max_wait)
