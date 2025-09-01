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
from bioagents.judge import ResponseJudge, HALOJudgmentSummary, AgentJudgment

# Back-compat shim for tests that monkeypatch HALOJudge
class HALOJudge(ResponseJudge):
    pass


PLANNING_PROMPT = """\
You are a planner deciding which specialists to invoke for a user query.

## Available capabilities

capabilities: graph, llama_rag, biomcp, web, llama_mcp, chitchat.

## Guidelines
You should strongly prefer at least 2 or more complementary capabilities when in doubt,
unless you are sure that the query is chitchat or exactly 1 capability is enough.

## Capability Descriptions:
- **graph**: NCCN breast cancer guidelines knowledge graph, treatment pathways, drug interactions, biomarker relationships
- **llama_rag**: Document retrieval from NCCN breast cancer guidelines and related medical documents
- **biomcp**: PubMed articles, biomedical research, genetic variants, clinical studies, ICD codes
- **web**: Current web search, latest news, recent developments, real-time information
- **llama_mcp**: Queries about NCCN breast cancer guidelines using LlamaCloud processing
- **chitchat**: Casual conversation, greetings, non-medical small talk

## Selection Guidelines:

**For NCCN/breast cancer queries:** graph + llama_rag (and optionally biomcp)
- Example: "NCCN breast cancer guidelines", "HER2-positive treatment", "treatment protocols"

**For biomedical research/PubMed queries:** biomcp + graph (if cancer-related) or biomcp + llama_rag
- Example: "latest research on gene X", "clinical studies for drug Y", "genetic variants"

**For document/guideline questions:** llama_rag + graph (if NCCN-related) or just llama_rag
- Example: "what does the PDF say about...", "guidelines for condition X"

**For current events/news:** web + biomcp (if medical) or just web
- Example: "latest COVID developments", "recent FDA approvals", "current medical news"

**For relationship/interaction queries:** graph + biomcp
- Example: "drug interactions", "biomarker relationships", "treatment combinations"

**For general medical questions:** biomcp + llama_rag + graph (comprehensive coverage)
- Example: "how to treat condition X", "what causes disease Y"

## Query
Query: {query}

## Output instructions
Return ONLY JSON with key 'capabilities' under one of two conditions:
- Exactly ["chitchat"] if and only if the user query is clearly small-talk.
- Otherwise an array of 2 or more items, each in the array:
  ["graph","llama_rag","biomcp","web","llama_mcp","chitchat"].
No prose. No other text.

## Example Output
{{"capabilities": ["graph", "llama_rag"]}}
"""

class CapabilityPlanner:
    """Plan capabilities from an LLM JSON object output.

    Simplicity-first: parse JSON, validate allowed values and lengths, otherwise fallback.
    """

    CAPABILITY_ENUM = ["graph", "llama_rag", "biomcp", "web", "llama_mcp", "chitchat"]

    def __init__(self, model_name: str = LLM.GPT_4_1_MINI, timeout: int = 10) -> None:
        self._llm = LLM(model_name=model_name, timeout=timeout)
        

    async def plan(self, query: str, available_caps: List[str]) -> List[str]:
        """Return ["chitchat"] when clearly small-talk, else 2 or more allowed capabilities.

        If JSON is invalid or no valid capabilities remain after filtering, return ["chitchat"].
        
        Args:
            query: The user query.
            available_caps: A list of capabilities (e.g. graph, llama_rag, biomcp, web, chitchat) 
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





class BioHALOAgent(BaseAgent):
    """
    HALO-style hierarchical orchestrator that can invoke multiple subagents based on 
    the query, returning a final synthesized response.
    
    Supports:
    - Knowledge graph, LlamaRAG, MCP, Web, and chit-chat agents
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
            "- Prepend the final response with '[HALO]'"
            "- Be concise and structured; provide citations where available"
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
            Literal["graph", "llama_rag", "biomcp", "web", "llama_mcp", "chitchat"]
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
            capabilities: A list of capabilities (e.g. graph, llama_rag, biomcp, web, chitchat) 
                          to select from.
            
        Returns:
            A list of tuples, each containing a capability and the corresponding sub-agent.
        """
        mapping: Dict[str, BaseAgent] = {}
        if self._graph_agent:
            mapping["graph"] = self._graph_agent
        if self._rag_agent:
            mapping["llama_rag"] = self._rag_agent
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
            A tuple containing the capability (e.g. graph, llama_rag, biomcp, web, chitchat) 
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

        Uses ResponseJudge to evaluate each response with structured scoring and justification.
        Returns a comprehensive judgment summary for synthesis guidance.
        
        Args:
            outputs: A list of tuples, each containing a subagent and the corresponding AgentResponse.
            query: The user query.
            
        Returns:
            HALOJudgmentSummary containing individual judgments and overall system score.
        """
        
        # Judge each subagent response individually
        individual_judgments: Dict[str, AgentJudgment] = {}
        
        for subagent_name, response in outputs:
            try:
                judgment = await self._response_judge.judge_response(subagent_name, response, query)
                individual_judgments[subagent_name] = judgment
                logger.info(f"ResponseJudge {subagent_name}: score={judgment.overall_score:.3f}")
            except Exception as e:
                logger.warning(f"ResponseJudge failed for {subagent_name}: {e}")
                # Use fallback judgment
                individual_judgments[subagent_name] = self._response_judge.create_fallback_judgment(
                    subagent_name, response, query
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
        """Generate assessment prose about the overall response quality."""
        
        if not judgments:
            return "No subagent responses were available for analysis."
        
        # Calculate score distribution
        scores = [j.overall_score for j in judgments.values()]
        avg_score = sum(scores) / len(scores)
        high_performers = [cap for cap, j in judgments.items() if j.overall_score >= 0.8]
        low_performers = [cap for cap, j in judgments.items() if j.overall_score < 0.5]
        
        # Generate qualitative assessment based on scores
        if avg_score >= 0.8:
            quality_assessment = "The overall response demonstrates high quality with strong evidence grounding and comprehensive coverage."
        elif avg_score >= 0.6:
            quality_assessment = "The response provides solid information with good coverage of the topic, though some areas could be strengthened."
        elif avg_score >= 0.4:
            quality_assessment = "The response addresses the query adequately but has notable gaps in completeness or grounding."
        else:
            quality_assessment = "The response has significant limitations in accuracy, completeness, or evidence support."
        
        # Add specific observations
        observations = []
        if high_performers:
            observations.append(f"Strong contributions from {', '.join(high_performers)} with excellent grounding and relevance.")
        if low_performers:
            observations.append(f"Areas for improvement identified in {', '.join(low_performers)} responses.")
        
        # Combine assessment
        if observations:
            return f"{quality_assessment} {' '.join(observations)}"
        else:
            return quality_assessment

    def _synthesize(self, outputs: List[Tuple[str, AgentResponse]], judgment_summary: HALOJudgmentSummary) -> AgentResponse:
        """Create a well-structured markdown response from multiple subagent outputs.

        Strategy: Organize content thematically with proper markdown structure,
        extract key insights from each agent, and present in a coherent narrative flow.
        """
        if not outputs:
            return AgentResponse(
                response_str="[HALO] No subagent responses available.",
                citations=[],
                route=AgentRouteType.REASONING,
            )

        # Citation management
        # Build a union of citations from ALL subagents and assign stable indices
        citation_index: Dict[str, int] = {}
        used_citations: List = []
        used_per_capability: Dict[str, set] = {}
        next_index = 1

        def _citation_identity(citation) -> str:
            """Return a robust identity key for a citation.

            Prefers URL; falls back to file name + page labels + title. This ensures
            citations without URLs (e.g., PDF page refs) still get indexed and
            can be referenced by inline markers.
            """
            try:
                url = str(getattr(citation, "url", "") or "").strip()
                file_name = str(getattr(citation, "file_name", "") or "").strip()
                start_page = str(getattr(citation, "start_page_label", "") or "").strip()
                end_page = str(getattr(citation, "end_page_label", "") or "").strip()
                page = str(getattr(citation, "page", "") or "").strip()
                title = str(getattr(citation, "title", "") or "").strip()
                if url:
                    return f"url={url}"
                page_part = f"{start_page}-{end_page}" if start_page or end_page else (page or "")
                return f"file={file_name}|page={page_part}|title={title}"
            except Exception:
                return repr(citation)

        # Pre-index all citations from all subagents to ensure the Sources list includes union
        for agent_name, resp in outputs:
            for c in getattr(resp, "citations", []) or []:
                key = _citation_identity(c)
                if key and key not in citation_index:
                    citation_index[key] = next_index
                    used_citations.append(c)
                    next_index += 1

        def _append_citation_markers(text_segment: str, resp: AgentResponse, capability: str) -> str:
            indices: List[int] = []
            cap_set = used_per_capability.get(capability, set())
            for c in getattr(resp, "citations", []) or []:
                # Use robust identity; if not present yet (unexpected), register now
                key = _citation_identity(c)
                if key not in citation_index:
                    citation_index[key] = next_index_nonlocal()
                    used_citations.append(c)
                idx = citation_index[key]
                indices.append(idx)
                cap_set.add(idx)
            used_per_capability[capability] = cap_set
            indices = sorted(list(dict.fromkeys(indices)))
            if indices:
                return f"{text_segment} [{','.join(str(i) for i in indices)}]"
            return text_segment

        def next_index_nonlocal() -> int:
            nonlocal next_index
            current = next_index
            next_index += 1
            return current

        def _strip_leading_tag(text: str) -> str:
            return re.sub(r"^\s*\[[^\]]+\]\s*", "", text or "").strip()

        def _extract_key_points(text: str) -> List[str]:
            """Extract key sentences/points from text for structured presentation."""
            if not text:
                return []
            
            # Split into sentences and filter meaningful ones
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            key_points = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and not sentence.startswith("However") and not sentence.startswith("Additionally"):
                    # Prioritize sentences with key medical terms, recommendations, or specific information
                    if any(keyword in sentence.lower() for keyword in [
                        "recommend", "should", "treatment", "therapy", "guideline", "study", "research",
                        "patient", "clinical", "evidence", "effective", "consider", "important"
                    ]):
                        key_points.append(sentence)
                    elif len(key_points) < 3:  # Include first few sentences if no keywords found
                        key_points.append(sentence)
                        
                if len(key_points) >= 4:  # Limit to avoid overwhelming output
                    break
                    
            return key_points

        # Organize responses by capability type
        agent_groups = {
            "guidelines": [],  # graph, llama_rag
            "research": [],    # biomcp
            "current": [],     # web
            "analysis": [],    # llama_mcp
            "other": []        # chitchat, etc.
        }

        for agent_name, resp in outputs:
            if agent_name in ["graph", "llama_rag"]:
                agent_groups["guidelines"].append((agent_name, resp))
            elif agent_name == "biomcp":
                agent_groups["research"].append((agent_name, resp))
            elif agent_name == "web":
                agent_groups["current"].append((agent_name, resp))
            elif agent_name == "llama_mcp":
                agent_groups["analysis"].append((agent_name, resp))
            else:
                agent_groups["other"].append((agent_name, resp))

        # Find best performing agent for executive summary
        judgments = getattr(judgment_summary, "individual_judgments", {})
        best_agent = None
        best_score = 0.0
        for agent_name, resp in outputs:
            judgment = judgments.get(agent_name)
            if judgment and judgment.overall_score > best_score:
                best_score = judgment.overall_score
                best_agent = (agent_name, resp)

        # Build structured markdown response
        markdown_sections = []
        
        # Executive Summary from best-performing agent (ensure HALO prefix)
        if best_agent:
            agent_name, resp = best_agent
            summary_text = _strip_leading_tag(resp.response_str)
            if summary_text:
                # Take first 2-3 sentences as executive summary
                summary_sentences = re.split(r"(?<=[.!?])\s+", summary_text.strip())
                executive_summary = ". ".join(summary_sentences[:2]) + "."
                executive_summary = _append_citation_markers(executive_summary, resp, agent_name)
                markdown_sections.append(f"[HALO] {executive_summary}")
        else:
            markdown_sections.append("[HALO] Response synthesis in progress.")

        # Clinical Guidelines & Protocols
        if agent_groups["guidelines"]:
            markdown_sections.append("\n## Clinical Guidelines & Protocols")
            for agent_name, resp in agent_groups["guidelines"]:
                content = _strip_leading_tag(resp.response_str)
                key_points = _extract_key_points(content)
                if key_points:
                    source_name = "NCCN Guidelines" if agent_name == "graph" else "Clinical Documents"
                    markdown_sections.append(f"\n### {source_name}")
                    for point in key_points:
                        point_with_citations = _append_citation_markers(point, resp, agent_name)
                        markdown_sections.append(f"- {point_with_citations}")

        # Research Evidence & Literature  
        if agent_groups["research"]:
            markdown_sections.append("\n## Research Evidence & Literature")
            for agent_name, resp in agent_groups["research"]:
                content = _strip_leading_tag(resp.response_str)
                key_points = _extract_key_points(content)
                if key_points:
                    for point in key_points:
                        point_with_citations = _append_citation_markers(point, resp, agent_name)
                        markdown_sections.append(f"- {point_with_citations}")

        # Current Developments
        if agent_groups["current"]:
            markdown_sections.append("\n## Current Developments")
            for agent_name, resp in agent_groups["current"]:
                content = _strip_leading_tag(resp.response_str)
                key_points = _extract_key_points(content)
                if key_points:
                    for point in key_points:
                        point_with_citations = _append_citation_markers(point, resp, agent_name)
                        markdown_sections.append(f"- {point_with_citations}")

        # Document Analysis
        if agent_groups["analysis"]:
            markdown_sections.append("\n## Document Analysis")
            for agent_name, resp in agent_groups["analysis"]:
                content = _strip_leading_tag(resp.response_str)
                key_points = _extract_key_points(content)
                if key_points:
                    for point in key_points:
                        point_with_citations = _append_citation_markers(point, resp, agent_name)
                        markdown_sections.append(f"- {point_with_citations}")

        # Additional Considerations
        if agent_groups["other"]:
            markdown_sections.append("\n## Additional Considerations")
            for agent_name, resp in agent_groups["other"]:
                if agent_name != "chitchat":  # Skip chitchat content
                    content = _strip_leading_tag(resp.response_str)
                    key_points = _extract_key_points(content)
                    if key_points:
                        for point in key_points:
                            point_with_citations = _append_citation_markers(point, resp, agent_name)
                            markdown_sections.append(f"- {point_with_citations}")

        # Key Recommendations (from highest-scoring agents)
        high_scoring_agents = [cap for cap, judgment in judgments.items() if judgment.overall_score >= 0.7]
        if high_scoring_agents and len(outputs) > 1:
            markdown_sections.append("\n## Key Recommendations")
            for agent_name in high_scoring_agents[:2]:  # Top 2 performers
                for agent_cap, resp in outputs:
                    if agent_cap == agent_name:
                        content = _strip_leading_tag(resp.response_str)
                        # Extract recommendation-style sentences
                        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
                        recommendations = [s for s in sentences if any(word in s.lower() for word in 
                                         ["recommend", "should", "consider", "advised", "suggested"])]
                        for rec in recommendations[:2]:  # Limit recommendations
                            rec_with_citations = _append_citation_markers(rec.strip(), resp, agent_name)
                            markdown_sections.append(f"- **{rec_with_citations}**")
                        break

        final_text = "\n".join(markdown_sections)
        citations = used_citations
        logger.info(f"HALO synthesis: {len(used_citations)} union citations included; {len(outputs)} subagents processed")

        # Build structured judge_response block (no HTML) for frontend expander
        def _first_sentence(text: str) -> str:
            parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
            return parts[0].strip() if parts else ""

        judgments = getattr(judgment_summary, "individual_judgments", {})
        overall_score = getattr(judgment_summary, "overall_halo_score", 0.0)
        synthesis_notes = (judgment_summary.synthesis_notes or "").strip()

        # Header lines
        judge_lines: List[str] = []
        judge_lines.append(f"**Score**: {overall_score:.2f}")

        # Build compact per-subagent summary: "cap 0.82 (N sources)"
        per_agent_summaries: List[str] = []
        agent_scores = []
        for agent_name, resp in outputs:
            j = judgments.get(agent_name)
            if not j:
                continue
            num_sources = len(getattr(resp, "citations", []) or [])
            per_agent_summaries.append(f"{agent_name} {j.overall_score:.2f} ({num_sources} sources)")
            agent_scores.append(f"{agent_name}: {j.overall_score:.2f}")

        if agent_scores:
            judge_lines.append(f"Agent Scores: {', '.join(agent_scores)}")

        # Single-line Assessment so the header renderer shows content inline
        summary_text = synthesis_notes if synthesis_notes else "Responses were synthesized to balance breadth and grounding across agents."
        if per_agent_summaries:
            judge_lines.append(f"**Assessment**: {summary_text}\n_{', '.join(per_agent_summaries)}_")
        else:
            judge_lines.append(f"**Assessment**: {summary_text}")

        # Per-subagent blocks: Response, Score, Justification, Citations  
        max_snippet_len = 800  # Increased from 400 to show more complete responses
        for agent_name, resp in outputs:
            j = judgments.get(agent_name)
            if not j:
                continue
            short_just = _first_sentence(j.prose_summary)
            # sources used in composed answer
            used_src_count = len(used_per_capability.get(agent_name, set()))
            if agent_name == "llama_rag" and used_src_count == 0 and getattr(resp, "citations", None):
                used_src_count = len(resp.citations)
            
            # Full response text (not truncated for judge display)
            full_response = _strip_leading_tag(resp.response_str)
            if len(full_response) > max_snippet_len:
                snippet = full_response[:max_snippet_len].rstrip() + "…"
            else:
                snippet = full_response
            
            judge_lines.append(f"- {agent_name}:")
            judge_lines.append(f"  - Response: {snippet}")
            judge_lines.append(f"  - Score: {j.overall_score:.2f}")
            judge_lines.append(f"  - Justification: {short_just} (with {used_src_count} sources)")
            
            # Add individual citations for this subagent
            subagent_citations = getattr(resp, "citations", []) or []
            if subagent_citations:
                judge_lines.append(f"  - Citations:")
                for i, citation in enumerate(subagent_citations, 1):
                    try:
                        # Extract all available metadata
                        title = getattr(citation, "title", None)
                        url = getattr(citation, "url", None)
                        file_name = getattr(citation, "file_name", None)
                        source = getattr(citation, "source", None)
                        start_page = getattr(citation, "start_page_label", None)
                        end_page = getattr(citation, "end_page_label", None)
                        # GraphRAG uses 'page' field instead of start_page_label/end_page_label  
                        page = getattr(citation, "page", None)
                        score = getattr(citation, "score", None)
                        snippet = getattr(citation, "snippet", None)
                        text = getattr(citation, "text", None)
                        
                        # GraphRAG-specific fields that may provide more distinguishing info
                        doc_id = getattr(citation, "doc_id", None)
                        paragraph = getattr(citation, "paragraph", None)
                        triplet_key = getattr(citation, "triplet_key", None)
                        provenance_id = getattr(citation, "provenance_id", None)
                        file_path = getattr(citation, "file_path", None)
                        
                        # Build a meaningful title with fallbacks
                        display_title = title
                        if not display_title or display_title.strip() == "":
                            display_title = doc_id or file_name or "Source"
                        if display_title:
                            display_title = str(display_title).strip()
                        
                        # Start building citation display
                        citation_parts = []
                        
                        # Main title/header
                        if url and str(url).strip():
                            citation_parts.append(f"**[{display_title}]({url})**")
                        else:
                            citation_parts.append(f"**{display_title}**")
                        
                        # Build detailed source information with location data
                        location_parts = []
                        
                        # File information (prefer file_name, fallback to extracting from file_path)
                        display_file = None
                        if file_name and str(file_name).strip():
                            display_file = str(file_name).strip()
                        elif file_path and str(file_path).strip():
                            # Extract filename from path for GraphRAG citations
                            path_str = str(file_path).strip()
                            display_file = path_str.split("/")[-1] if "/" in path_str else path_str
                        
                        if display_file:
                            location_parts.append(f"*{display_file}*")
                        
                        # Page information with enhanced detail
                        page_added = False
                        # Check GraphRAG page field first
                        if page and str(page).strip():
                            location_parts.append(f"p. {page}")
                            page_added = True
                        # Fallback to start_page/end_page for other agent types
                        elif start_page and str(start_page).strip():
                            if end_page and str(end_page).strip() and start_page != end_page:
                                location_parts.append(f"pp. {start_page}-{end_page}")
                            else:
                                location_parts.append(f"p. {start_page}")
                            page_added = True
                        
                        # Paragraph/section information for GraphRAG
                        if paragraph is not None and str(paragraph).strip():
                            location_parts.append(f"¶ {paragraph}")
                        
                        # Source type information
                        if source and str(source).strip() and source != "knowledge_graph":
                            location_parts.append(f"via {source}")
                        elif source == "knowledge_graph":
                            location_parts.append("knowledge graph")
                            
                        # Relevance score
                        if score and score > 0.0:
                            location_parts.append(f"relevance: {score:.2f}")
                        
                        # Add location info if we have any
                        if location_parts:
                            citation_parts.append(f"({', '.join(location_parts)})")
                        
                        # Enhanced snippet handling - prioritize longer, more informative text
                        best_text = None
                        
                        # Priority 1: snippet (usually most relevant excerpt)
                        if snippet and str(snippet).strip() and len(str(snippet).strip()) > 15:
                            best_text = str(snippet).strip()
                            
                        # Priority 2: text field (may have more context)
                        elif text and str(text).strip() and len(str(text).strip()) > 15:
                            best_text = str(text).strip()[:300]  # Limit very long text
                            
                        # Priority 3: generate contextual info from metadata
                        elif triplet_key:
                            best_text = f"Knowledge graph triplet: {triplet_key}"
                        
                        # Display the best available text
                        if best_text:
                            # Clean up whitespace and format nicely
                            best_text = ' '.join(best_text.split())
                            
                            # Ensure reasonable length for display
                            if len(best_text) > 200:
                                # Try to break at sentence boundary
                                sentences = best_text.split('. ')
                                if len(sentences) > 1 and len(sentences[0]) < 180:
                                    best_text = sentences[0] + '.'
                                else:
                                    best_text = best_text[:197] + "..."
                            
                            citation_parts.append(f"\n      > \"{best_text}\"")
                        else:
                            # Fallback: show that this is a graph citation with limited text
                            if source == "knowledge_graph":
                                citation_parts.append(f"\n      > *[Knowledge graph reference]*")
                        
                        # Add debug info for GraphRAG citations if available  
                        debug_parts = []
                        if doc_id and doc_id != display_title:
                            debug_parts.append(f"doc_id: {doc_id}")
                        if provenance_id:
                            debug_parts.append(f"prov_id: {provenance_id}")
                            
                        if debug_parts and len(debug_parts) < 3:  # Only add if not too verbose
                            citation_parts.append(f"\n      *{', '.join(debug_parts)}*")
                        
                        # Combine all parts
                        judge_lines.append(f"    {i}. {' '.join(citation_parts)}")
                        
                    except Exception as e:
                        logger.warning(f"Error processing citation {i} for {agent_name}: {e}")
                        judge_lines.append(f"    {i}. [Citation processing error]")
            else:
                judge_lines.append(f"  - Citations: None")

        # Do not append judge lines into the primary response string; expose via AgentResponse.judge_response
        return AgentResponse(
            response_str=final_text,
            citations=citations,
            judgement="\n".join(judge_lines),
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
            subagents = self._plan(query_str)
        except Exception:
            subagents = []
        if not isinstance(subagents, list) or not subagents:
            subagents = await self._plan_async(query_str)
        roles = self._select_roles(subagents)
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
    print(f">> Judge response: {resp.judgement}")

    # -----Test 2------
    query = "For elderly patients who have complex history with variety of health " +\
        "problems, how should we first assess their risk of remission after they " +\
        "have completed a full dose of chemotherapy?"
    resp = await agent.achat(query)
    print(f">> BioHALOAgent response: {resp.response_str}")
    print(f">> # citations: {len(resp.citations)}")
    print(f">> Judge response: {resp.judgement}")
    await agent.stop()
        
if __name__ == "__main__":
    asyncio.run(main())
