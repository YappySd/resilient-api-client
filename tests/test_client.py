"""All tests run without network and without real waiting: time and HTTP are faked."""
import httpx
import pytest

from resilient_client import ApiError, AsyncRateLimiter, ResilientClient, backoff_delay, parse_retry_after


class FakeTime:
    """A clock whose sleep() just moves time forward and records the request."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(handler, fake: FakeTime, **kwargs) -> ResilientClient:
    options = dict(rate_per_sec=1000.0, burst=10, backoff_base=0.5, rng=lambda: 1.0)
    options.update(kwargs)
    return ResilientClient(
        "https://api.test",
        transport=httpx.MockTransport(handler),
        sleep=fake.sleep,
        clock=fake.clock,
        **options,
    )


def scripted(*responses):
    """Handler that returns the given responses (or raises exceptions) one per call."""
    queue = list(responses)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    handler.calls = calls
    return handler


# --- pure helpers -----------------------------------------------------------


def test_backoff_grows_exponentially_and_respects_cap():
    assert backoff_delay(0, 0.5, 30, lambda: 1.0) == 0.5
    assert backoff_delay(3, 0.5, 30, lambda: 1.0) == 4.0
    assert backoff_delay(10, 0.5, 30, lambda: 1.0) == 30


def test_backoff_applies_jitter():
    assert backoff_delay(2, 1.0, 30, lambda: 0.5) == 2.0
    assert backoff_delay(2, 1.0, 30, lambda: 0.0) == 0.0


@pytest.mark.parametrize(
    ("header", "expected"),
    [("3", 3.0), ("2.5", 2.5), ("9999", 60.0), ("", None), (None, None), ("-1", None), ("soon", None)],
)
def test_parse_retry_after(header, expected):
    assert parse_retry_after(header, 60.0) == expected


# --- rate limiter -----------------------------------------------------------


async def test_rate_limiter_spaces_out_requests():
    fake = FakeTime()
    limiter = AsyncRateLimiter(2.0, burst=1, clock=fake.clock, sleep=fake.sleep)
    for _ in range(3):
        await limiter.acquire()
    # 2 requests per second means 0.5 s between requests: the 2nd and 3rd each wait 0.5 s
    assert sum(fake.sleeps) == pytest.approx(1.0)


async def test_rate_limiter_allows_a_burst_first():
    fake = FakeTime()
    limiter = AsyncRateLimiter(2.0, burst=3, clock=fake.clock, sleep=fake.sleep)
    for _ in range(3):
        await limiter.acquire()
    assert fake.sleeps == []
    await limiter.acquire()
    assert sum(fake.sleeps) == pytest.approx(0.5)


def test_rate_limiter_validates_arguments():
    with pytest.raises(ValueError):
        AsyncRateLimiter(0)
    with pytest.raises(ValueError):
        AsyncRateLimiter(1, burst=0)


# --- client -----------------------------------------------------------------


async def test_success_returns_json_without_waiting():
    fake = FakeTime()
    handler = scripted(httpx.Response(200, json={"ok": True}))
    async with make_client(handler, fake) as client:
        assert await client.get_json("/ping") == {"ok": True}
    assert fake.sleeps == []
    assert len(handler.calls) == 1


async def test_retries_503_with_exponential_backoff_then_succeeds():
    fake = FakeTime()
    handler = scripted(httpx.Response(503), httpx.Response(503), httpx.Response(200, json={"n": 1}))
    async with make_client(handler, fake) as client:
        assert await client.get_json("/data") == {"n": 1}
    assert fake.sleeps == [0.5, 1.0]
    assert len(handler.calls) == 3


async def test_honors_retry_after_header_on_429():
    fake = FakeTime()
    handler = scripted(httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, json={}))
    async with make_client(handler, fake) as client:
        await client.get("/data")
    assert fake.sleeps == [3.0]


async def test_retry_after_is_capped():
    fake = FakeTime()
    handler = scripted(httpx.Response(429, headers={"Retry-After": "9999"}), httpx.Response(200, json={}))
    async with make_client(handler, fake, max_retry_after=60.0) as client:
        await client.get("/data")
    assert fake.sleeps == [60.0]


async def test_gives_up_after_max_retries():
    fake = FakeTime()
    handler = scripted(*[httpx.Response(503, text="down")] * 3)
    async with make_client(handler, fake, max_retries=2) as client:
        with pytest.raises(ApiError) as info:
            await client.get("/data")
    assert info.value.status_code == 503
    assert len(handler.calls) == 3  # first try + 2 retries
    assert len(fake.sleeps) == 2


async def test_does_not_retry_client_errors():
    fake = FakeTime()
    handler = scripted(httpx.Response(404, text="no such thing"))
    async with make_client(handler, fake) as client:
        with pytest.raises(ApiError) as info:
            await client.get("/missing")
    assert info.value.status_code == 404
    assert "no such thing" in info.value.body
    assert len(handler.calls) == 1
    assert fake.sleeps == []


async def test_retries_network_errors():
    fake = FakeTime()
    handler = scripted(httpx.ConnectError("boom"), httpx.Response(200, json={"ok": 1}))
    async with make_client(handler, fake) as client:
        assert await client.get_json("/data") == {"ok": 1}
    assert fake.sleeps == [0.5]


async def test_network_errors_eventually_raise_api_error():
    fake = FakeTime()
    handler = scripted(*[httpx.ConnectError("boom")] * 2)
    async with make_client(handler, fake, max_retries=1) as client:
        with pytest.raises(ApiError):
            await client.get("/data")
    assert len(handler.calls) == 2


async def test_invalid_json_raises_api_error():
    fake = FakeTime()
    handler = scripted(httpx.Response(200, text="<html>not json</html>"))
    async with make_client(handler, fake) as client:
        with pytest.raises(ApiError):
            await client.get_json("/data")


async def test_client_applies_rate_limit_to_every_attempt():
    fake = FakeTime()
    handler = scripted(httpx.Response(503), httpx.Response(200, json={}))
    async with make_client(handler, fake, rate_per_sec=1.0, burst=1, backoff_base=0.0) as client:
        await client.get("/data")
    # attempt 2 happens right after a zero-length backoff, so the limiter must hold it for ~1 s
    assert sum(fake.sleeps) == pytest.approx(1.0)
