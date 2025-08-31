import asyncio
from unittest.mock import AsyncMock

import pytest

from bioagents.agents.bio_halo import BioHALOAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.source import Source


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
    agent._plan = lambda q: ["llama_rag"]

    resp = await agent.achat("What do NCCN guidelines say?")
    assert isinstance(resp, AgentResponse)
    assert resp.response_str.startswith("[HALO]")
    # Structured evaluation is provided in judge_response for frontend expander
    assert resp.judge_response
    assert "**Score**:" in resp.judge_response
    assert "**Assessment**:" in resp.judge_response


@pytest.mark.asyncio
async def test_bio_halo_multicapability_merge_and_citations():
    agent = BioHALOAgent(name="BioHALO")

    # Two sub-agents respond, one with citations
    rag_resp = AgentResponse(
        response_str="[RAG] Guideline text",
        route=AgentRouteType.LLAMARAG,
        citations=[Source(url="https://example.com/nccn", title="NCCN", snippet="", source="web")],
    )
    graph_resp = AgentResponse(
        response_str="[Graph] Relationship analysis",
        route=AgentRouteType.GRAPH,
        citations=[Source(url="https://example.com/graphref", title="Graph Ref", snippet="", source="web")],
    )
    agent._rag_agent = AsyncMock()
    agent._graph_agent = AsyncMock()
    agent._rag_agent.achat = AsyncMock(return_value=rag_resp)
    agent._graph_agent.achat = AsyncMock(return_value=graph_resp)

    agent._plan = lambda q: ["graph", "llama_rag"]

    resp = await agent.achat("How do HER2 status and HR status interact under NCCN?")
    assert "[HALO]" in resp.response_str
    # Inline markers should appear referencing merged citations
    assert "[1" in resp.response_str or ",1" in resp.response_str
    # Only used citations should be retained
    assert len(resp.citations) >= 1


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
async def test_structured_evaluation_block_and_counts():
    agent = BioHALOAgent(name="BioHALO")

    # Prepare responses with citations
    from bioagents.models.source import Source
    graph_resp = AgentResponse(
        response_str="[Graph] Graph supports relation X.",
        route=AgentRouteType.GRAPH,
        citations=[Source(url="https://s1", title="S1", snippet="", source="web")],
    )
    web_resp = AgentResponse(
        response_str="[Web] Web adds details.",
        route=AgentRouteType.REASONING,
        citations=[Source(url="https://s2", title="S2", snippet="", source="web")],
    )

    agent._graph_agent = AsyncMock()
    agent._web_agent = AsyncMock()
    agent._graph_agent.achat = AsyncMock(return_value=graph_resp)
    agent._web_agent.achat = AsyncMock(return_value=web_resp)
    agent._plan = lambda q: ["graph", "web"]

    # Monkeypatch judge to deterministic outputs returning a valid AgentJudgment
    from bioagents.agents import bio_halo as mod

    async def fake_judge_response(self, capability, response, query):
        score = 0.9 if capability == "web" else 0.8
        return mod.AgentJudgment(
            prose_summary="Good and useful.",
            scores={
                "accuracy": score,
                "completeness": score,
                "groundedness": score,
                "professional_tone": score,
                "clarity_coherence": score,
                "relevance": score,
                "usefulness": score,
            },
            overall_score=score,
            justifications={
                "accuracy": "Reasonable and grounded.",
                "completeness": "Covers key aspects.",
                "groundedness": "Uses sources where relevant.",
                "professional_tone": "Professional tone maintained.",
                "clarity_coherence": "Clear and coherent.",
                "relevance": "Directly relevant.",
                "usefulness": "Actionable guidance.",
            },
        )

    mod.HALOJudge.judge_response = fake_judge_response

    resp = await agent.achat("test")
    assert resp.judge_response
    assert "**Score**:" in resp.judge_response
    # Check per-subagent lines exist
    assert "- graph:" in resp.judge_response
    assert "- web:" in resp.judge_response
    # Citations only for used indices
    assert len(resp.citations) in (1, 2)


@pytest.mark.asyncio
async def test_planner_llm_success(monkeypatch):
    agent = BioHALOAgent(name="BioHALO")

    async def fake_achat_completion(self, query_str: str, **kwargs):
        return '{"capabilities": ["graph", "llama_rag"]}'

    from bioagents.agents import bio_halo as mod
    monkeypatch.setattr(mod.LLM, "achat_completion", fake_achat_completion, raising=False)

    caps = await agent._plan_async("How do HER2 and HR status interact under NCCN?")
    assert caps == ["graph", "llama_rag"]


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


