"""An async HTTP client that is polite to APIs: rate limit, retries, backoff, Retry-After."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable

import httpx

from .backoff import backoff_delay, parse_retry_after
from .errors import ApiError
from .ratelimit import AsyncRateLimiter

logger = logging.getLogger(__name__)

# Temporary conditions worth retrying. Other 4xx errors are our fault and never retried.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class ResilientClient:
    def __init__(
        self,
        base_url: str = "",
        *,
        rate_per_sec: float = 5.0,
        burst: int = 1,
        max_retries: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 30.0,
        max_retry_after: float = 60.0,
        timeout: float = 10.0,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[], float] = random.random,
    ) -> None:
        """`sleep`, `clock`, `rng` and `transport` can be replaced in tests to make them instant."""
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, headers=headers, transport=transport
        )
        self._limiter = AsyncRateLimiter(rate_per_sec, burst, clock=clock, sleep=sleep)
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._max_retry_after = max_retry_after
        self._sleep = sleep
        self._rng = rng

    async def __aenter__(self) -> "ResilientClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send a request, retrying temporary failures. Returns a response with status < 400."""
        attempt = 0
        while True:
            await self._limiter.acquire()
            try:
                response = await self._client.request(method, url, **kwargs)
            except httpx.TransportError as exc:  # timeouts, connection errors, resets
                if attempt >= self._max_retries:
                    raise ApiError(
                        f"{method} {url} failed after {attempt + 1} attempts: {exc!r}"
                    ) from exc
                delay = self._backoff(attempt)
                reason = repr(exc)
            else:
                status = response.status_code
                if status < 400:
                    return response
                if status not in RETRY_STATUSES or attempt >= self._max_retries:
                    raise ApiError(
                        f"{method} {url} -> HTTP {status}",
                        status_code=status,
                        body=response.text[:500],
                    )
                retry_after = parse_retry_after(
                    response.headers.get("retry-after"), self._max_retry_after
                )
                delay = retry_after if retry_after is not None else self._backoff(attempt)
                reason = f"HTTP {status}"

            logger.warning(
                "%s %s: %s, retry %d/%d in %.2fs",
                method, url, reason, attempt + 1, self._max_retries, delay,
            )
            await self._sleep(delay)
            attempt += 1

    def _backoff(self, attempt: int) -> float:
        return backoff_delay(attempt, self._backoff_base, self._backoff_cap, self._rng)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    async def get_json(self, url: str, **kwargs: Any) -> Any:
        response = await self.get(url, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(f"GET {url}: response is not valid JSON") from exc
