from __future__ import annotations


class ApiError(Exception):
    """Raised when a request ultimately fails (non-retryable status, or retries exhausted)."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
