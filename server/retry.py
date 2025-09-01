from __future__ import annotations

from typing import Any, Callable, Coroutine, TypeVar, Union, cast

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential


T = TypeVar("T")


class RetryPolicy:
    """Simple async retry decorator with exponential backoff using tenacity.

    Designed to be minimal and composable. Use for flaky external calls.
    """

    def __init__(
        self,
        attempts: int = 3,
        wait_min_seconds: float = 0.5,
        wait_max_seconds: float = 4.0,
        exception_types: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.attempts = attempts
        self.wait_min_seconds = wait_min_seconds
        self.wait_max_seconds = wait_max_seconds
        self.exception_types = exception_types

    def __call__(self, fn: Callable[..., Coroutine[Any, Any, T]]):
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.attempts),
                wait=wait_exponential(min=self.wait_min_seconds, max=self.wait_max_seconds),
                retry=retry_if_exception_type(self.exception_types),
                reraise=True,
            ):
                with attempt:
                    return await fn(*args, **kwargs)
            # Type hinting; flow never reaches here due to reraise=True
            return cast(T, None)

        return wrapper


# A sensible default policy for network calls
default_retry = RetryPolicy(attempts=3, wait_min_seconds=0.5, wait_max_seconds=3.0)


