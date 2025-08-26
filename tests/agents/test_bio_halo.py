import asyncio
from unittest.mock import AsyncMock

import pytest

from bioagents.agents.bio_halo import BioHALOAgent
from bioagents.agents.common import AgentResponse, AgentRouteType


@pytest.mark.asyncio
async def test_bio_halo_basic_flow_monocapability():
    agent = BioHALOAgent(name="BioHALO")
    # Mock sub-agent to avoid heavy calls
    agent._rag_agent = AsyncMock()
    agent._rag_agent.achat = AsyncMock(return_value=AgentResponse(
        response_str="[RAG] NCCN recommendation ...",
        route=AgentRouteType.LLAMARAG,
    ))

    # Force planner to pick only RAG
    agent._plan = lambda q: ["rag"]

    resp = await agent.achat("What do NCCN guidelines say?")
    assert isinstance(resp, AgentResponse)
    assert resp.response_str.startswith("[HALO]")
    assert resp.judge_response.startswith("HALO Judge Summary:")


@pytest.mark.asyncio
async def test_bio_halo_multicapability_merge_and_citations():
    agent = BioHALOAgent(name="BioHALO")

    # Two sub-agents respond, one with citations
    rag_resp = AgentResponse(response_str="[RAG] Guideline text", route=AgentRouteType.LLAMARAG)
    graph_resp = AgentResponse(response_str="[Graph] Relationship analysis", route=AgentRouteType.GRAPH)
    agent._rag_agent = AsyncMock()
    agent._graph_agent = AsyncMock()
    agent._rag_agent.achat = AsyncMock(return_value=rag_resp)
    agent._graph_agent.achat = AsyncMock(return_value=graph_resp)

    agent._plan = lambda q: ["graph", "rag"]

    resp = await agent.achat("How do HER2 status and HR status interact under NCCN?")
    assert "[HALO]" in resp.response_str
    # Primary should prefer graph for relationship queries
    assert "[From" in resp.response_str or "Relationship" in resp.response_str


@pytest.mark.asyncio
async def test_bio_halo_handles_subagent_error():
    agent = BioHALOAgent(name="BioHALO")
    failing = AsyncMock()
    failing.achat = AsyncMock(side_effect=RuntimeError("boom"))
    agent._web_agent = failing
    agent._plan = lambda q: ["web"]

    resp = await agent.achat("latest news")
    assert "[HALO]" in resp.response_str
    assert resp.route == AgentRouteType.REASONING


@pytest.mark.asyncio
async def test_planner_llm_success(monkeypatch):
    agent = BioHALOAgent(name="BioHALO")

    async def fake_achat_completion(self, query_str: str, **kwargs):
        return '{"capabilities": ["graph", "rag"]}'

    from bioagents.agents import bio_halo as mod
    monkeypatch.setattr(mod.LLM, "achat_completion", fake_achat_completion, raising=False)

    caps = await agent._plan_async("How do HER2 and HR status interact under NCCN?")
    assert caps == ["graph", "rag"]


@pytest.mark.asyncio
async def test_planner_llm_invalid_json(monkeypatch):
    agent = BioHALOAgent(name="BioHALO")

    async def fake_achat_completion(self, query_str: str, **kwargs):
        return 'not-json'

    from bioagents.agents import bio_halo as mod
    monkeypatch.setattr(mod.LLM, "achat_completion", fake_achat_completion, raising=False)

    caps = await agent._plan_async("latest news")
    assert caps == ["chitchat"]


@pytest.mark.asyncio
async def test_planner_llm_validation_failure(monkeypatch):
    agent = BioHALOAgent(name="BioHALO")

    async def fake_achat_completion(self, query_str: str, **kwargs):
        return '{"capabilities": ["unknown_capability"]}'

    from bioagents.agents import bio_halo as mod
    monkeypatch.setattr(mod.LLM, "achat_completion", fake_achat_completion, raising=False)

    caps = await agent._plan_async("something")
    assert caps == ["chitchat"]


