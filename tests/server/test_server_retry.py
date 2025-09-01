import asyncio

import pytest

from server.retry import RetryPolicy


@pytest.mark.asyncio
async def test_retry_policy_retries_then_succeeds():
    calls = {"n": 0}

    policy = RetryPolicy(attempts=3, wait_min_seconds=0.01, wait_max_seconds=0.02)

    @policy
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("fail once")
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert calls["n"] == 2


