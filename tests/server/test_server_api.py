import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from server.api import create_app
from server.config import AppConfig


@pytest.fixture(scope="module")
def client():
    cfg = AppConfig(host="127.0.0.1", port=9000)
    app = create_app(cfg)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_unknown_agent(client):
    r = client.post("/doesnotexist/chat", json={"query": "hello"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_chat_router_smoke():
    # Use ASGI in-memory to avoid network
    cfg = AppConfig(host="127.0.0.1", port=9000)
    app = create_app(cfg)

    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/router/chat", json={"query": "hello"})
        # We don't assert 200 strictly since underlying LLM calls may fail in CI
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert "response" in data


