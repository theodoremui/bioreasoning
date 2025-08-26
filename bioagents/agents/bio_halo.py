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
from pydantic import BaseModel, ValidationError, Field
from typing import Literal

from loguru import logger

from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.agents.graph_agent import GraphAgent
from bioagents.agents.biomcp_agent import BioMCPAgent
from bioagents.agents.web_agent import WebReasoningAgent
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.llamamcp_agent import LlamaMCPAgent
from bioagents.models.llms import LLM
from bioagents.agents.llamarag_agent import LlamaRAGAgent


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
Return ONLY JSON with key 'capabilities' under one of two conditions:
- Exactly ["chitchat"] if and only if the user query is clearly small-talk.
- Otherwise an array of 2 or more items, each in the array:
  ["graph","rag","biomcp","web","llama_mcp","chitchat"].
No prose. No other text.

## Example Output
{{"capabilities": ["graph", "rag"]}}
"""

JUDGE_PROMPT = """You are an impartial judge, evaluating one subagent's response to a user query. Your task has two parts:

1. **Prose Summary** (2–3 sentences):
   Provide a concise, professional summary highlighting the response's main strengths and areas for improvement.

2. **Structured Score (JSON)**:
   Assign each criterion a score between 0.0 and 1.0, with 1.0 being perfect. Then compute an overall weighted score. Provide a brief justification for each score.

Return a JSON object **and nothing else**, with the following schema:

{{
  "prose_summary": "<2-3 sentence summary>",
  "scores": {{
    "accuracy": float,
    "completeness": float,
    "groundedness": float,
    "professional_tone": float,
    "clarity_coherence": float,
    "relevance": float,
    "usefulness": float
  }},
  "overall_score": float,
  "justifications": {{
    "accuracy": "<one-sentence justification>",
    "completeness": "<one-sentence justification>",
    "groundedness": "<one-sentence justification>",
    "professional_tone": "<one-sentence justification>",
    "clarity_coherence": "<one-sentence justification>",
    "relevance": "<one-sentence justification>",
    "usefulness": "<one-sentence justification>"
  }}
}}

Guidelines:
	•	Prose Summary should be objective, highlighting what the response does well and what could be improved.
	•	Scores must all be between 0.0 and 1.0; if unsure, choose a sober estimate (e.g., 0.5).
	•	Overall score should be a weighted average of the criteria—weights are equal unless otherwise specified.
	•	Justifications should be clear, succinct, referencing specific elements of the response (e.g. "The answer cites accurate stats," "Tone is overly casual," etc.).
	•	Do not output any extra content beyond the JSON object.

## User Query
{query}

## Subagent Response
Capability: {capability}
Response: {response}

## Citations
{citations}

## Instructions
Evaluate this response and return ONLY the JSON object as specified above."""


class AgentJudgment(BaseModel):
    """Structured judgment of a subagent's response quality."""
    
    prose_summary: str = Field(..., description="2-3 sentence summary of strengths and areas for improvement")
    scores: Dict[str, float] = Field(..., description="Individual criterion scores from 0.0 to 1.0")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall weighted score")
    justifications: Dict[str, str] = Field(..., description="One-sentence justification for each score")
    
    class Config:
        extra = "forbid"  # Strict validation - no extra fields allowed


class HALOJudgmentSummary(BaseModel):
    """Summary of all subagent judgments for HALO orchestration."""
    
    individual_judgments: Dict[str, AgentJudgment] = Field(..., description="Judgment for each capability")
    overall_halo_score: float = Field(..., ge=0.0, le=1.0, description="Overall HALO system score")
    synthesis_notes: str = Field(..., description="Notes for response synthesis")
    
    class Config:
        extra = "forbid"


class CapabilityPlanner:
    """Plan capabilities from an LLM JSON object output.

    Simplicity-first: parse JSON, validate allowed values and lengths, otherwise fallback.
    """

    CAPABILITY_ENUM = ["graph", "rag", "biomcp", "web", "llama_mcp", "chitchat"]

    def __init__(self, model_name: str = LLM.GPT_4_1_MINI, timeout: int = 10) -> None:
        self._llm = LLM(model_name=model_name, timeout=timeout)
        

    async def plan(self, query: str, available_caps: List[str]) -> List[str]:
        """Return ["chitchat"] when clearly small-talk, else 2 or more allowed capabilities.

        If JSON is invalid or no valid capabilities remain after filtering, return ["chitchat"].
        
        Args:
            query: The user query.
            available_caps: A list of capabilities (e.g. graph, rag, biomcp, web, chitchat) 
                          to select from.
            
        Returns:
            A list of capabilities.
        """
        prompt = PLANNING_PROMPT.format(query=query)
        try:
            content = await self._llm.achat_completion(
                prompt,
                temperature=0,
                response_format={"type": "json_object"},
            )
            logger.debug(f"CapabilityPlanner raw content (truncated): {str(content)[:300]}")
            data = json.loads((content or "{}").strip())
            raw_caps = data.get("capabilities", []) or []
            # Normalize: lowercase, trim, unique order-preserving
            seen: set[str] = set()
            caps: List[str] = []
            for item in raw_caps:
                s = str(item).strip().lower()
                if s in CapabilityPlanner.CAPABILITY_ENUM and s not in seen:
                    caps.append(s)
                    seen.add(s)

            if caps == ["chitchat"]:
                return ["chitchat"]
            if len(caps) >= 2:
                return caps
            return ["chitchat"]
        except Exception as e:
            logger.warning(f"CapabilityPlanner parsing failed: {e}")
            return ["chitchat"]


class HALOJudge:
    """LLM-based judge for evaluating subagent responses using structured output."""
    
    def __init__(self, model_name: str = LLM.GPT_4_1_MINI, timeout: int = 15) -> None:
        self._llm = LLM(model_name=model_name, timeout=timeout)
    
    async def judge_response(self, capability: str, response: AgentResponse, query: str) -> AgentJudgment:
        """Judge a single subagent response using LLM with structured output."""
        
        # Prepare citations text
        citations_text = "None"
        if response.citations:
            citations_text = "\n".join([
                f"- {getattr(c, 'url', 'Unknown source')}: {getattr(c, 'title', 'No title')}"
                for c in response.citations
            ])
        
        # Format prompt
        prompt = JUDGE_PROMPT.format(
            query=query,
            capability=capability,
            response=response.response_str or "No response",
            citations=citations_text
        )
        
        try:
            # Get structured LLM response
            content = await self._llm.achat_completion(
                prompt,
                temperature=0.1,  # Low temperature for consistent judgment
                response_format={"type": "json_object"},
            )
            
            # Parse and validate JSON
            data = json.loads((content or "{}").strip())
            judgment = AgentJudgment.model_validate(data)
            
            logger.debug(f"HALO Judge for {capability}: overall_score={judgment.overall_score}")
            return judgment
            
        except Exception as e:
            logger.warning(f"HALO Judge failed for {capability}: {e}")
            # Fallback to basic scoring
            return self._fallback_judgment(capability, response, query)
    
    def _fallback_judgment(self, capability: str, response: AgentResponse, query: str) -> AgentJudgment:
        """Fallback judgment when LLM judging fails."""
        
        # Basic heuristic scoring similar to original method
        score = 0.0
        if response.citations:
            score += min(0.3, len(response.citations) * 0.1)
        if len(response.response_str or "") > 120:
            score += 0.1
        
        # Domain alignment
        if capability == "graph" and any(k in query.lower() for k in ["interact", "relationship", "correlat"]):
            score += 0.1
        if capability == "rag" and any(k in query.lower() for k in ["nccn", "guideline", "pdf", "document"]):
            score += 0.1
        
        # Cap at 1.0
        score = min(1.0, score)
        
        return AgentJudgment(
            prose_summary=f"Fallback judgment for {capability} due to LLM evaluation failure",
            scores={
                "accuracy": score,
                "completeness": score,
                "groundedness": score,
                "professional_tone": 0.5,
                "clarity_coherence": score,
                "relevance": score,
                "usefulness": score
            },
            overall_score=score,
            justifications={
                "accuracy": "Fallback scoring due to LLM evaluation failure",
                "completeness": "Fallback scoring due to LLM evaluation failure",
                "groundedness": "Fallback scoring due to LLM evaluation failure",
                "professional_tone": "Fallback scoring due to LLM evaluation failure",
                "clarity_coherence": "Fallback scoring due to LLM evaluation failure",
                "relevance": "Fallback scoring due to LLM evaluation failure",
                "usefulness": "Fallback scoring due to LLM evaluation failure"
            }
        )


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
    4) Judge: critique agent outputs using LLM-based structured evaluation
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
        
        # LLM-based judge for response evaluation
        self._halo_judge = HALOJudge(model_name=model_name)

    async def start(self) -> None:
        if self._started:
            return
        # Instantiate sub-agents only if not pre-injected (to allow tests/mocks)
        if self._graph_agent is None:
            self._graph_agent = GraphAgent(name="Graph Agent")
        if self._rag_agent is None:
            self._rag_agent = LlamaRAGAgent(name="LlamaCloud RAG Agent")
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
        """Delegates to CapabilityPlanner with schema-constrained outputs and fallback."""
        # Plan against the full capability set; role selection handles availability
        available = list(CapabilityPlanner.CAPABILITY_ENUM)

        planner = CapabilityPlanner(model_name=LLM.GPT_4_1_MINI, timeout=10)
        return await planner.plan(query, available)

    async def _plan_async(self, query: str) -> List[str]:
        """Async planner using schema-based CapabilityPlanner; minimal deterministic fallback."""
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

    async def _judge(self, outputs: List[Tuple[str, AgentResponse]], query: str) -> HALOJudgmentSummary:
        """Produce LLM-based structured judgments across sub-agent outputs.

        Uses HALOJudge to evaluate each response with structured scoring and justification.
        Returns a comprehensive judgment summary for synthesis guidance.
        
        Args:
            outputs: A list of tuples, each containing a capability and the corresponding AgentResponse.
            query: The user query.
            
        Returns:
            HALOJudgmentSummary containing individual judgments and overall system score.
        """
        
        # Judge each subagent response individually
        individual_judgments: Dict[str, AgentJudgment] = {}
        
        for capability, response in outputs:
            try:
                judgment = await self._halo_judge.judge_response(capability, response, query)
                individual_judgments[capability] = judgment
                logger.info(f"HALO Judge {capability}: score={judgment.overall_score:.3f}")
            except Exception as e:
                logger.warning(f"HALO Judge failed for {capability}: {e}")
                # Use fallback judgment
                individual_judgments[capability] = self._halo_judge._fallback_judgment(
                    capability, response, query
                )
        
        # Calculate overall HALO system score
        if individual_judgments:
            overall_score = sum(j.overall_score for j in individual_judgments.values()) / len(individual_judgments)
        else:
            overall_score = 0.0
        
        # Generate synthesis notes based on judgments
        synthesis_notes = self._generate_synthesis_notes(individual_judgments, query)
        
        return HALOJudgmentSummary(
            individual_judgments=individual_judgments,
            overall_halo_score=overall_score,
            synthesis_notes=synthesis_notes
        )
    
    def _generate_synthesis_notes(self, judgments: Dict[str, AgentJudgment], query: str) -> str:
        """Generate synthesis guidance notes based on individual judgments."""
        
        if not judgments:
            return "No subagent responses available for synthesis."
        
        # Find best and worst performing agents
        best_agent = max(judgments.items(), key=lambda x: x[1].overall_score)
        worst_agent = min(judgments.items(), key=lambda x: x[1].overall_score)
        
        notes = [
            f"Best performing agent: {best_agent[0]} (score: {best_agent[1].overall_score:.3f})",
            f"Lowest performing agent: {worst_agent[0]} (score: {worst_agent[1].overall_score:.3f})"
        ]
        
        # Add specific guidance based on scores
        high_performers = [cap for cap, j in judgments.items() if j.overall_score >= 0.8]
        if high_performers:
            notes.append(f"High-quality responses from: {', '.join(high_performers)}")
        
        low_performers = [cap for cap, j in judgments.items() if j.overall_score < 0.5]
        if low_performers:
            notes.append(f"Consider supplementing responses from: {', '.join(low_performers)}")
        
        return "; ".join(notes)

    def _synthesize(self, outputs: List[Tuple[str, AgentResponse]], judgment_summary: HALOJudgmentSummary) -> AgentResponse:
        """Merge multiple AgentResponse objects into a single response.

        Strategy: prioritize Graph and RAG content; append supplementary points
        from others succinctly. Merge and de-duplicate citations that are actually used
        in the final composed answer, and add inline markers.
        """
        # Prioritization
        priority_order = {"graph": 0, "rag": 1, "biomcp": 2, "llama_mcp": 3, "web": 4, "chitchat": 5}
        sorted_out = sorted(outputs, key=lambda t: priority_order.get(t[0], 99))

        def _strip_leading_tag(text: str) -> str:
            # Remove leading tags like [Graph], [RAG], [MCP], [HALO]
            return re.sub(r"^\s*\[[^\]]+\]\s*", "", text or "").strip()

        # Inline citation machinery: assign indices lazily only for citations that appear in final text
        used_url_to_index: Dict[str, int] = {}
        used_citations: List = []
        used_per_capability: Dict[str, set] = {}
        next_index = 1

        def _append_citation_markers(text_segment: str, resp: AgentResponse, capability: str) -> str:
            indices: List[int] = []
            cap_set = used_per_capability.get(capability, set())
            for c in getattr(resp, "citations", []) or []:
                url = getattr(c, "url", None)
                if not url:
                    continue
                if url not in used_url_to_index:
                    # assign new index lazily on first use
                    used_url_to_index[url] = next_index_nonlocal()
                    used_citations.append(c)
                idx = used_url_to_index[url]
                indices.append(idx)
                cap_set.add(idx)
            used_per_capability[capability] = cap_set
            # Deduplicate and sort
            indices = sorted(list(dict.fromkeys(indices)))
            if indices:
                return f"{text_segment} [{','.join(str(i) for i in indices)}]"
            return text_segment

        def next_index_nonlocal() -> int:
            nonlocal next_index
            current = next_index
            next_index += 1
            return current

        primary_text: str = ""

        # Build unified answer: start from primary, then weave in unique sentences from others with inline markers
        if sorted_out:
            primary_cap, primary_resp = sorted_out[0]
            primary_text = _strip_leading_tag(primary_resp.response_str)
            primary_text = _append_citation_markers(primary_text, primary_resp, primary_cap)
        final_text = "[HALO] " + (primary_text if primary_text else "No primary answer available.")

        def _unique_sentence_addition(current: str, additional: str, capability: str, resp: AgentResponse, max_new: int = 2) -> str:
            added = 0
            for sent in re.split(r"(?<=[.!?])\s+", additional):
                clean = sent.strip()
                if not clean:
                    continue
                if clean not in current:
                    # append with markers tied to this fragment
                    fragment_with_markers = _append_citation_markers(clean, resp, capability)
                    current += (" " if not current.endswith(" ") else "") + fragment_with_markers
                    added += 1
                    if added >= max_new:
                        break
            return current

        for idx in range(1, len(sorted_out)):
            cap, resp = sorted_out[idx]
            text = _strip_leading_tag(resp.response_str)
            if text:
                final_text = _unique_sentence_addition(final_text, text, cap, resp)

        # Build structured evaluation block
        def _first_sentence(text: str) -> str:
            parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
            return parts[0].strip() if parts else ""

        overall_score = getattr(judgment_summary, "overall_halo_score", 0.0)
        assessment_lines: List[str] = []
        # Heuristic assessment using synthesis notes and best/worst
        assessment_lines.append(f"Overall Score: {overall_score:.2f}")
        # Convert synthesis notes into a cogent sentence if available
        synthesis_notes = (judgment_summary.synthesis_notes or "").strip()
        if synthesis_notes:
            assessment_lines.append(f"Assessment: {synthesis_notes}")
        else:
            assessment_lines.append("Assessment: Responses were synthesized to balance breadth and grounding across agents.")

        # Per-subagent lines in the order they contributed
        judgments = getattr(judgment_summary, "individual_judgments", {})
        for cap, resp in sorted_out:
            j = judgments.get(cap)
            if not j:
                continue
            short_just = _first_sentence(j.prose_summary)
            src_count = len(used_per_capability.get(cap, set()))
            assessment_lines.append(f"- {cap}: {j.overall_score:.2f} - {short_just} (with {src_count} sources)")

        evaluation_block = "\n" + "\n".join(assessment_lines)

        final_text = final_text + "\n\n" + evaluation_block

        # Only include citations that were actually used (in the order of assigned indices)
        citations = used_citations

        return AgentResponse(
            response_str=final_text,
            citations=citations,
            judge_response=evaluation_block.strip(),
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
        judgment_summary = await self._judge(outputs, query_str)
        return self._synthesize(outputs, judgment_summary)


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
    print(f">> Judge response: {resp.judge_response}")

    # -----Test 2------
    query = "For elderly patients who have complex history with variety of health " +\
        "problems, how should we first assess their risk of remission after they " +\
        "have completed a full dose of chemotherapy?"
    resp = await agent.achat(query)
    print(f">> BioHALOAgent response: {resp.response_str}")
    print(f">> # citations: {len(resp.citations)}")
    print(f">> Judge response: {resp.judge_response}")
    await agent.stop()
        
if __name__ == "__main__":
    asyncio.run(main())
