# ------------------------------------------------------------------------------
# bio_halo.py
#
# BioHALOAgent: A HALO-style hierarchical orchestrator for multi-agent LLM systems
# Complementary to BioRouterAgent, this agent can invoke any subset of
# specialized sub-agents per query (parallel and/or sequential), evaluate them,
# and synthesize a final response with citations.
#
# Architecture inspired by HALO: https://arxiv.org/abs/2505.13516
# High-level planning -> Role design/selection -> Low-level execution & synthesis
#
# Author: Theodore Mui
# Date: 2025-08-20
# ------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Tuple
from pydantic import BaseModel, ValidationError
from typing import Literal

from loguru import logger

from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.agents.graph_agent import GraphAgent
from bioagents.agents.llamarag_agent import LlamaRAGAgent
from bioagents.agents.biomcp_agent import BioMCPAgent
from bioagents.agents.web_agent import WebReasoningAgent
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.llamamcp_agent import LlamaMCPAgent
from bioagents.models.llms import LLM


class BioHALOAgent(BaseAgent):
    """
    HALO-style hierarchical orchestrator that can invoke multiple subagents based on 
    the query, returning a final synthesized response.
    
    Supports:
    - Knowledge graph, RAG, MCP, Web, and chit-chat agents
    - Dynamic subagent instantiation
    - Consequential tool-calling evaluation

    Tiered flow (HALO):
    1) Plan: analyze query and propose required capabilities
    2) Role design/selection: map capabilities to concrete sub-agents
    3) Execute: run selected agents in parallel and collect results
    4) Judge: critique agent outputs (lightweight heuristic)
    5) Synthesize: produce a concise, well-structured answer with citations
    """

    def __init__(
        self,
        name: str = "BioHALO",
        model_name: str = LLM.GPT_4_1_MINI,
    ) -> None:
        instructions = (
            "You are a hierarchical orchestrator (HALO). You plan, select, and coordinate"
            " multiple specialists to answer complex biomedical and general questions.\n\n"
            "## Response Instructions:\n"
            "- Prepend the final response with '[HALO]'\n"
            "- Be concise and structured; provide citations where available\n"
        )
        super().__init__(name, model_name, instructions)

        self._started = False
        self._agent = None  # Unused intentionally; orchestration is manual

        # Sub-agents created in start(); kept as attributes for reuse
        self._graph_agent: GraphAgent | None = None
        self._rag_agent: LlamaRAGAgent | None = None
        self._biomcp_agent: BioMCPAgent | None = None
        self._web_agent: WebReasoningAgent | None = None
        self._chat_agent: ChitChatAgent | None = None
        self._llamamcp_agent: LlamaMCPAgent | None = None

    async def start(self) -> None:
        if self._started:
            return
        # Instantiate sub-agents (lazy; their own achat handles per-call init)
        self._graph_agent = GraphAgent(name="Graph Agent")
        self._rag_agent = LlamaRAGAgent(name="LlamaCloud RAG Agent")
        self._biomcp_agent = BioMCPAgent(name="Bio MCP Agent")
        self._web_agent = WebReasoningAgent(name="Web Reasoning Agent")
        self._chat_agent = ChitChatAgent(name="Chit Chat Agent")
        self._llamamcp_agent = LlamaMCPAgent(name="LlamaCloud MCP Agent")
        self._started = True

    async def stop(self) -> None:
        # Sub-agents manage their own lifecycles; provide a clean entry point
        self._started = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    # -----------------------------
    # HALO tier 1: Planning
    # -----------------------------
    class _CapabilityPlan(BaseModel):
        capabilities: List[
            Literal["graph", "rag", "biomcp", "web", "llama_mcp", "chitchat"]
        ]

    async def _plan_with_llm(self, query: str) -> List[str]:
        """Optional LLM-backed planner returning capability tags.

        Uses a small model to classify which specialists are helpful,
        returning strictly validated capabilities via Pydantic.
        """
        try:
            llm = LLM(model_name=LLM.GPT_4_1_MINI, timeout=10)
            prompt = (
                "You are a planner deciding which specialists to invoke for a user query.\n"
                "Available capabilities: graph, rag, biomcp, web, llama_mcp, chitchat.\n"
                "Return ONLY JSON with a key 'capabilities' whose value is an ordered list of any of these strings.\n"
                "No prose.\n\n"
                f"Query: {query}\n"
                "Example: {\"capabilities\": [\"graph\", \"rag\"]}"
            )
            # Prefer JSON responses; rely on strict schema validation
            content = await llm.achat_completion(
                prompt,
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads((content or "{}").strip())
            plan = BioHALOAgent._CapabilityPlan.model_validate(data)
            filtered = plan.capabilities
            # Deduplicate while preserving order
            seen = set()
            ordered: List[str] = []
            for c in filtered:
                if c not in seen:
                    ordered.append(c)
                    seen.add(c)
            return ordered or ["chitchat"]
        except (ValidationError, json.JSONDecodeError, Exception):
            # Minimal safe fallback to ensure progress; avoid keyword heuristics
            return ["chitchat"]

    async def _plan_async(self, query: str) -> List[str]:
        """Async planner using LLM structured output; minimal fallback to chitchat."""
        caps = await self._plan_with_llm(query)
        return caps if caps else ["chitchat"]

    def _plan(self, query: str) -> List[str]:
        """Synchronous wrapper to enable test monkeypatching; uses LLM planner."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._plan_with_llm(query))
        except RuntimeError:
            # If already in an event loop, rely on async path from caller
            return []
        except Exception:
            return ["chitchat"]

    # --------------------------------------
    # HALO tier 2: Role selection/mapping
    # --------------------------------------
    def _select_roles(self, capabilities: List[str]) -> List[Tuple[str, BaseAgent]]:
        mapping: Dict[str, BaseAgent] = {}
        if self._graph_agent:
            mapping["graph"] = self._graph_agent
        if self._rag_agent:
            mapping["rag"] = self._rag_agent
        if self._biomcp_agent:
            mapping["biomcp"] = self._biomcp_agent
        if self._web_agent:
            mapping["web"] = self._web_agent
        if self._llamamcp_agent:
            mapping["llama_mcp"] = self._llamamcp_agent
        if self._chat_agent:
            mapping["chitchat"] = self._chat_agent

        selected: List[Tuple[str, BaseAgent]] = []
        for cap in capabilities:
            agent = mapping.get(cap)
            if agent is not None:
                selected.append((cap, agent))
        return selected

    # --------------------------------------
    # HALO tier 3: Execution & Synthesis
    # --------------------------------------
    async def _execute_roles(self, query: str, roles: List[Tuple[str, BaseAgent]]) -> List[Tuple[str, AgentResponse]]:
        tasks = []
        for cap, agent in roles:
            tasks.append(self._execute_one(cap, agent, query))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: List[Tuple[str, AgentResponse]] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"HALO sub-agent failed: {r}")
                continue
            outputs.append(r)  # (capability, AgentResponse)
        return outputs

    async def _execute_one(self, capability: str, agent: BaseAgent, query: str) -> Tuple[str, AgentResponse]:
        try:
            resp = await agent.achat(query)
            return capability, resp
        except Exception as e:
            logger.error(f"Sub-agent '{capability}' error: {e}")
            return capability, AgentResponse(response_str=f"[{capability}] unavailable", route=AgentRouteType.REASONING)

    def _judge(self, outputs: List[Tuple[str, AgentResponse]], query: str) -> str:
        """Produce a lightweight critique across sub-agent outputs.

        Heuristics: prefer responses with citations, reward domain-aligned routes,
        penalize empty/short answers. Returns a concise textual judge summary.
        """
        lines: List[str] = []
        for cap, resp in outputs:
            score = 0
            if resp.citations:
                score += min(3, len(resp.citations))
            if len(resp.response_str or "") > 120:
                score += 1
            # Domain hint alignment
            if cap == "graph" and any(k in query.lower() for k in ["interact", "relationship", "correlat"]):
                score += 1
            if cap == "rag" and any(k in query.lower() for k in ["nccn", "guideline", "pdf", "document"]):
                score += 1
            lines.append(f"- {cap}: score={score} citations={len(resp.citations)}")

        return "\n".join(["HALO Judge Summary:"] + lines)

    def _synthesize(self, outputs: List[Tuple[str, AgentResponse]], judge_text: str) -> AgentResponse:
        """Merge multiple AgentResponse objects into a single response.

        Strategy: prioritize Graph and RAG content; append supplementary points
        from others succinctly. Merge and de-duplicate citations.
        """
        # Prioritization
        priority_order = {"graph": 0, "rag": 1, "biomcp": 2, "llama_mcp": 3, "web": 4, "chitchat": 5}
        sorted_out = sorted(outputs, key=lambda t: priority_order.get(t[0], 99))

        primary_text_parts: List[str] = []
        supplementary_parts: List[str] = []
        citations = []
        seen_urls = set()

        for cap, resp in sorted_out:
            # Gather citations with de-dupe
            for c in resp.citations:
                url = getattr(c, "url", None)
                if url and url in seen_urls:
                    continue
                citations.append(c)
                if url:
                    seen_urls.add(url)

        for idx, (cap, resp) in enumerate(sorted_out):
            text = (resp.response_str or "").strip()
            if not text:
                continue
            if idx == 0:
                primary_text_parts.append(text)
            else:
                supplementary_parts.append(f"\n\n[From {cap}] {text}")

        final_text = "[HALO] " + (" ".join(primary_text_parts) if primary_text_parts else "No primary answer available.")
        if supplementary_parts:
            final_text += "\n\nAdditional insights:" + "".join(supplementary_parts)

        return AgentResponse(
            response_str=final_text,
            citations=citations,
            judge_response=judge_text,
            route=AgentRouteType.REASONING,
        )

    # -----------------------------
    # Public entry
    # -----------------------------
    async def achat(self, query_str: str) -> AgentResponse:
        logger.info(f"=> halo: {self.name}: {query_str}")
        if not getattr(self, "_started", False):
            await self.start()

        # Respect test overrides that monkeypatch _plan(); otherwise use async planner
        try:
            capabilities = self._plan(query_str)
        except Exception:
            capabilities = []
        if not isinstance(capabilities, list) or not capabilities:
            capabilities = await self._plan_async(query_str)
        roles = self._select_roles(capabilities)
        outputs = await self._execute_roles(query_str, roles)
        judge_text = self._judge(outputs, query_str)
        return self._synthesize(outputs, judge_text)


