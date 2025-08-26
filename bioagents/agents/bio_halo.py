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
import re
from typing import Dict, List, Tuple
from pydantic import BaseModel, ValidationError
from typing import Literal

from loguru import logger

from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.agents.graph_agent import GraphAgent
from bioagents.agents.biomcp_agent import BioMCPAgent
from bioagents.agents.web_agent import WebReasoningAgent
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.llamamcp_agent import LlamaMCPAgent
from bioagents.agents.llamarag_agent import LlamaRAGAgent
from bioagents.models.llms import LLM


PLANNING_PROMPT = """
You are a planner deciding which specialists to invoke for a user query.
## Available capabilities

capabilities: graph, rag, biomcp, web, llama_mcp, chitchat.

## Guidelines
You should strongly prefer at least 2 or more complementary capabilities when in doubt,
unless you are sure that the query is chitchat or exactly 1 capability is enough.

The selected capabilities should involve diversified sources 
(e.g., combine biomcp with rag or web).

For biomedical clinical queries, include biomcp plus one of rag/web.
For document/guideline queries include rag.
For relationship/network questions include graph.

## Query
Query: {query}

## Output instructions
Return ONLY JSON with a key 'capabilities' whose value is an ordered list 
of any of these strings.  No prose. No other text.

## Example Output
{"capabilities": ["graph", "rag"]}
"""

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
        # Instantiate sub-agents only if not pre-injected (to allow tests/mocks)
        if self._graph_agent is None:
            self._graph_agent = GraphAgent(name="Graph Agent")
        # Lazy import LlamaRAGAgent to avoid optional dependency import errors at module import time
        if self._rag_agent is None:
            try:
                from bioagents.agents.llamarag_agent import LlamaRAGAgent  # type: ignore
                self._rag_agent = LlamaRAGAgent(name="LlamaCloud RAG Agent")
            except Exception:
                self._rag_agent = None
        if self._biomcp_agent is None:
            self._biomcp_agent = BioMCPAgent(name="Bio MCP Agent")
        if self._web_agent is None:
            self._web_agent = WebReasoningAgent(name="Web Reasoning Agent")
        if self._chat_agent is None:
            self._chat_agent = ChitChatAgent(name="Chit Chat Agent")
        if self._llamamcp_agent is None:
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
        """LLM-backed planner returning capability tags -- which are the names 
        of the sub-agents to invoke.

        Uses a small model to classify which specialists are helpful,
        returning strictly validated capabilities via Pydantic.
        
        Args:
            query: The user query.
            
        Returns:
            A list of capability tags.
        """
        try:
            llm = LLM(model_name=LLM.GPT_4_1_MINI, timeout=10)
            prompt = PLANNING_PROMPT.format(query=query)
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
            if loop.is_running():
                # Let caller await the async planner path
                return []
            return loop.run_until_complete(self._plan_with_llm(query))
        except Exception:
            return ["chitchat"]

    # --------------------------------------
    # HALO tier 2: Role selection/mapping
    # --------------------------------------
    def _select_roles(self, capabilities: List[str]) -> List[Tuple[str, BaseAgent]]:
        """
        Select the appropriate sub-agents based on the capabilities.
        
        Args:
            capabilities: A list of capabilities (e.g. graph, rag, biomcp, web, chitchat) 
                          to select from.
            
        Returns:
            A list of tuples, each containing a capability and the corresponding sub-agent.
        """
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
                
        logger.info(f"***> selected {len(selected)} roles: {selected}")
        return selected

    # --------------------------------------
    # HALO tier 3: Execution & Synthesis
    # --------------------------------------
    async def _execute_roles(
        self, 
        query: str, 
        roles: List[Tuple[str, BaseAgent]]
    ) -> List[Tuple[str, AgentResponse]]:
        """
        Execute the roles (sub-agents) and return the results.
        
        Args:
            query: The user query.
            roles: A list of tuples, each containing a capability and the corresponding sub-agent.
            
        Returns:
            A list of tuples, each containing a capability and the corresponding AgentResponse.
        """
        
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

    async def _execute_one(
        self, 
        capability: str, 
        agent: BaseAgent, 
        query: str
    ) -> Tuple[str, AgentResponse]:
        """
        Execute one sub-agent and return the result.
        
        Args:
            capability: The capability to execute.
            agent: The sub-agent to execute.
            query: The user query.
            
        Returns:
            A tuple containing the capability (e.g. graph, rag, biomcp, web, chitchat) 
            and the AgentResponse.
        """
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
        
        Args:
            outputs: A list of tuples, each containing a capability and the corresponding AgentResponse.
            query: The user query.
            
        Returns:
            A string containing the judge summary.
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

        def _strip_leading_tag(text: str) -> str:
            # Remove leading tags like [Graph], [RAG], [MCP], [HALO]
            return re.sub(r"^\s*\[[^\]]+\]\s*", "", text or "").strip()

        primary_text: str = ""
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

        # Build a global inline citation index based on merged citations
        citation_index_by_url: Dict[str, int] = {}
        idx_counter = 1
        for c in citations:
            url = getattr(c, "url", None)
            if url and url not in citation_index_by_url:
                citation_index_by_url[url] = idx_counter
                idx_counter += 1

        def _append_citation_markers(text_segment: str, resp: AgentResponse) -> str:
            indices: List[int] = []
            for c in getattr(resp, "citations", []) or []:
                url = getattr(c, "url", None)
                if url and url in citation_index_by_url:
                    indices.append(citation_index_by_url[url])
            # Deduplicate and sort
            indices = sorted(list(dict.fromkeys(indices)))
            if indices:
                return f"{text_segment} [{','.join(str(i) for i in indices)}]"
            return text_segment

        # Build unified answer: start from primary, then weave in unique sentences from others with inline markers
        if sorted_out:
            primary_cap, primary_resp = sorted_out[0]
            primary_text = _strip_leading_tag(primary_resp.response_str)
            primary_text = _append_citation_markers(primary_text, primary_resp)
        final_text = "[HALO] " + (primary_text if primary_text else "No primary answer available.")

        def _unique_sentence_addition(current: str, additional: str, max_new: int = 2) -> str:
            added = 0
            for sent in re.split(r"(?<=[.!?])\s+", additional):
                clean = sent.strip()
                if not clean:
                    continue
                if clean not in current:
                    current += (" " if not current.endswith(" ") else "") + clean
                    added += 1
                    if added >= max_new:
                        break
            return current

        for idx in range(1, len(sorted_out)):
            cap, resp = sorted_out[idx]
            text = _strip_leading_tag(resp.response_str)
            if text:
                text_with_markers = _append_citation_markers(text, resp)
                final_text = _unique_sentence_addition(final_text, text_with_markers)

        # Append overall judge summary for transparency
        if judge_text:
            final_text += f"\n\nJudge: {judge_text}"

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


#-------------------------------------
# Test code
#-------------------------------------

async def main():
    agent = BioHALOAgent()
    await agent.start()
    
    # -----Test 1------
    query = "What is the best way to treat an elderly patient with HER2 breast cancer?"
    resp = await agent.achat(query)
    print(f">> BioHALOAgent response: {resp.response_str}")
    print(f">> # citations: {len(resp.citations)}")

    # -----Test 2------
    query = "For elderly patients who have complex history with variety of health " +\
        "problems, how should we first assess their risk of remission after they " +\
        "have completed a full dose of chemotherapy?"
    resp = await agent.achat(query)
    print(f">> BioHALOAgent response: {resp.response_str}")
    print(f">> # citations: {len(resp.citations)}")
    await agent.stop()
        
if __name__ == "__main__":
    asyncio.run(main())
