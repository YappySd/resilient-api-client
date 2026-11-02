"""Token-bucket rate limiter for asyncio."""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable


class AsyncRateLimiter:
    """Allows `rate_per_sec` requests per second on average, with bursts up to `burst`.

    Waiters queue on a lock, so requests are released in order and the limit holds
    even when many coroutines call acquire() at once.
    """

    def __init__(
        self,
        rate_per_sec: float,
        burst: int = 1,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if burst < 1:
            raise ValueError("burst must be at least 1")
        self._rate = rate_per_sec
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1 - 1e-9:
                    self._tokens -= 1
                    return
                await self._sleep((1 - self._tokens) / self._rate)
