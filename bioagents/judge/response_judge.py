# ------------------------------------------------------------------------------
# bioagents/judge/response_judge.py
#
# ResponseJudge: LLM-based evaluator for agent responses with structured scoring.
# Implements comprehensive response evaluation using multiple criteria with
# transparent justifications and fallback mechanisms.
#
# Follows SOLID principles:
# - SRP: Single responsibility for response judgment
# - OCP: Open for extension with custom scoring criteria
# - LSP: Substitutable for ResponseJudgeInterface
# - ISP: Focused interface for judgment operations
# - DIP: Depends on abstractions (LLM interface)
#
# Author: Theodore Mui
# Date: 2025-01-31
# ------------------------------------------------------------------------------

import json
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from loguru import logger

from bioagents.models.llms import LLM
from .interfaces import ResponseJudgeInterface, JudgmentError
from .models import AgentJudgment, JudgmentScores, JudgmentJustifications
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

# Avoid runtime circular import; only import for type checking
if TYPE_CHECKING:
    from bioagents.agents.common import AgentResponse


# Judgment prompt template for structured LLM evaluation
JUDGMENT_PROMPT = """You are an impartial expert evaluator assessing an AI agent's response quality.

## Evaluation Criteria (0.0 = Poor, 1.0 = Excellent)

**Accuracy**: Factual correctness and reliability of information
**Completeness**: Coverage of query requirements and thoroughness  
**Groundedness**: Quality of evidence, citations, and source support
**Professional Tone**: Appropriateness and clarity of communication
**Clarity & Coherence**: Readability, logical flow, and organization
**Relevance**: Alignment with user query intent and context
**Usefulness**: Practical value and actionability for the user

## Task Instructions

1. Write a concise 2-3 sentence summary highlighting strengths and improvement areas
2. Assign numerical scores (0.0-1.0) for each criterion with brief justifications
3. Calculate overall score as equal-weighted average of all criteria
4. Return ONLY a JSON object conforming to the specified schema

## Input Context

**User Query**: {query}

**Agent Type**: {agent_name}  
**Agent Response**: {response}

**Available Citations**:
{citations}

## Additional Context
{context_info}

## Output Requirements
Return a JSON object with exactly these fields:
- prose_summary: string (2-3 sentences)  
- scores: object with all 7 criteria as numbers 0.0-1.0
- overall_score: number 0.0-1.0 (average of all scores)
- justifications: object with explanations for each criterion
"""


class ResponseJudge(ResponseJudgeInterface):
    """LLM-based response judge implementing comprehensive structured evaluation.
    
    Uses large language models to evaluate agent responses across multiple
    quality dimensions with transparent scoring and justifications.
    Includes robust error handling and fallback scoring mechanisms.
    
    Key Features:
    - Multi-criteria evaluation (accuracy, completeness, groundedness, etc.)
    - Structured JSON output with score justifications
    - Configurable LLM backends with timeout handling
    - Heuristic fallback scoring for reliability
    - Domain-specific context awareness
    
    Example Usage:
        judge = ResponseJudge(model_name=LLM.GPT_4_1_MINI, timeout=15)
        judgment = await judge.judge_response('graph', response, query)
        print(f"Overall score: {judgment.overall_score:.2f}")
    """

    def __init__(
        self, 
        model_name: str = LLM.GPT_4_1_MINI, 
        timeout: int = 15,
        temperature: float = 0.1,
        enable_schema_mode: bool = True,
        max_citations_in_prompt: int = 6,
        max_response_chars_in_prompt: int = 1024,
    ) -> None:
        """Initialize the response judge with configurable parameters.
        
        Args:
            model_name: LLM model identifier for evaluation.
            timeout: Timeout in seconds for LLM requests.
            temperature: Sampling temperature for LLM (0.0-2.0, lower = more deterministic).
            enable_schema_mode: Whether to use JSON schema constraints when available.
            max_citations_in_prompt: Maximum number of citations to include in the prompt.
            max_response_chars_in_prompt: Maximum number of characters to include in the prompt.
        """
        self._model_name = model_name
        self._timeout = timeout
        self._temperature = temperature
        self._enable_schema_mode = enable_schema_mode
        self._llm = LLM(model_name=model_name, timeout=timeout)
        self._max_citations_in_prompt = max_citations_in_prompt
        self._max_response_chars_in_prompt = max_response_chars_in_prompt
        
        logger.info(
            f"ResponseJudge initialized: model={model_name}, timeout={timeout}s, "
            f"temp={temperature}, schema_mode={enable_schema_mode}"
        )

    @property
    def model_name(self) -> str:
        """Get the LLM model name used for judgment."""
        return self._model_name

    @property
    def timeout(self) -> int:
        """Get the timeout value for judgment operations."""
        return self._timeout

    async def judge_response(
        self, 
        agent_name: str, 
        response: "AgentResponse", 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentJudgment:
        """Judge a single agent response using structured LLM evaluation.
        
        Evaluates responses across multiple criteria using an LLM with structured
        output formatting. Includes fallback handling for robustness.
        
        Args:
            agent_name: The agent name being evaluated (e.g., 'graph', 'web').
            response: The agent response containing text and citations.
            query: The original user query for relevance assessment.
            context: Optional context for specialized evaluation.
            
        Returns:
            Complete structured judgment with scores and justifications.
            
        Raises:
            JudgmentError: When critical evaluation failures occur.
        """
        try:
            # Prepare citations summary for evaluation context (limited)
            limited_citations = (response.citations or [])[: self._max_citations_in_prompt]
            citations_text = self._format_citations(limited_citations)
            
            # Format context information
            context_info = self._format_context(context) if context else "None provided"
            
            # Format the judgment prompt (limit response length)
            display_response = (response.response_str or "")[: self._max_response_chars_in_prompt]
            prompt = JUDGMENT_PROMPT.format(
                query=query,
                agent_name=agent_name,
                response=display_response or "No response generated",
                citations=citations_text,
                context_info=context_info
            )
            
            # Attempt structured LLM evaluation
            try:
                use_schema = self._should_use_schema_mode()
                judgment_data = await self._get_structured_judgment(prompt, use_schema)
                # Ensure required fields introduced in AgentJudgment are present
                if not isinstance(judgment_data, dict):
                    judgment_data = {}
                judgment_data.setdefault("agent_name", agent_name)
                judgment_data.setdefault("response_str", response.response_str or "")
                judgment = self._validate_and_normalize_judgment(judgment_data)
                
                logger.debug(
                    f"ResponseJudge completed for {agent_name}: "
                    f"score={judgment.overall_score:.3f}"
                )
                return judgment
                
            except Exception as llm_error:
                logger.warning(
                    f"LLM judgment failed for {agent_name}: {llm_error}. "
                    f"Using fallback evaluation."
                )
                return self.create_fallback_judgment(
                    agent_name, response, query, 
                    f"LLM evaluation failed: {str(llm_error)}"
                )
                
        except Exception as e:
            error_msg = f"Critical judgment failure for {agent_name}"
            logger.error(f"{error_msg}: {e}")
            raise JudgmentError(error_msg, agent_name, e)

    async def _get_structured_judgment(self, prompt: str, use_schema: bool) -> Dict[str, Any]:
        """Get structured judgment from LLM with schema validation when possible.
        
        Args:
            prompt: Formatted evaluation prompt.
            
        Returns:
            Raw judgment data as dictionary.
            
        Raises:
            Exception: When LLM request fails.
        """
        schema = self._minimal_schema()
        
        # Try JSON schema mode first if enabled and allowed for model
        if use_schema:
            try:
                content = await self._achat_with_retry(
                    prompt,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "AgentJudgment", "schema": schema},
                    },
                )
                return json.loads((content or "{}").strip())
            except Exception as e:
                logger.debug(f"JSON schema mode failed ({e}), falling back to json_object")

        # Fallback to JSON object mode
        schema_prompt = prompt + f"\n\nReturn ONLY a JSON object conforming to this schema:\n{json.dumps(schema)}"
        content = await self._achat_with_retry(
            schema_prompt,
            response_format={"type": "json_object"},
        )
        return json.loads((content or "{}").strip())

    def _minimal_schema(self) -> Dict[str, Any]:
        """Return a compact JSON schema sufficient for AgentJudgment validation."""
        score_fields = [
            "accuracy",
            "completeness",
            "groundedness",
            "professional_tone",
            "clarity_coherence",
            "relevance",
            "usefulness",
        ]
        return {
            "type": "object",
            "properties": {
                "prose_summary": {"type": "string"},
                "scores": {
                    "type": "object",
                    "properties": {f: {"type": "number"} for f in score_fields},
                    "required": score_fields,
                },
                "overall_score": {"type": "number"},
                "justifications": {
                    "type": "object",
                    "properties": {f: {"type": "string"} for f in score_fields},
                    "required": score_fields,
                },
            },
            "required": ["prose_summary", "scores", "overall_score", "justifications"],
            "additionalProperties": True,
        }

    def _should_use_schema_mode(self) -> bool:
        """Gate JSON schema mode based on settings (tests expect it when enabled)."""
        return bool(self._enable_schema_mode)

    async def _achat_with_retry(self, prompt: str, response_format: Dict[str, Any]) -> str:
        """LLM achat with short retry and exponential backoff."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_exponential(min=0.2, max=1.0),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                return await self._llm.achat_completion(
                    prompt,
                    temperature=self._temperature,
                    response_format=response_format,
                )
        return "{}"

    def _validate_and_normalize_judgment(self, raw_data: Dict[str, Any]) -> AgentJudgment:
        """Validate and normalize raw LLM judgment data.
        
        Args:
            raw_data: Raw judgment dictionary from LLM.
            
        Returns:
            Validated and normalized AgentJudgment object.
        """
        # Ensure basic structure
        if not isinstance(raw_data, dict):
            raw_data = {}
            
        raw_data.setdefault("prose_summary", "")
        raw_data.setdefault("scores", {})
        raw_data.setdefault("justifications", {})

        # Normalize and clamp scores
        scores = raw_data.get("scores", {})
        score_fields = [
            "accuracy", "completeness", "groundedness", 
            "professional_tone", "clarity_coherence", 
            "relevance", "usefulness"
        ]
        
        for field in score_fields:
            scores[field] = self._clamp_score(scores.get(field, 0.0))
        raw_data["scores"] = scores

        # Calculate overall score if missing
        if "overall_score" not in raw_data:
            raw_data["overall_score"] = self._clamp_score(
                sum(scores.values()) / len(score_fields) if scores else 0.0
            )

        # Ensure justifications for all criteria
        justifications = raw_data.get("justifications", {})
        for field in score_fields:
            justifications[field] = str(justifications.get(field, "No justification provided"))
        raw_data["justifications"] = justifications

        # Validate using Pydantic model
        return AgentJudgment.model_validate(raw_data)

    def create_fallback_judgment(
        self, 
        agent_name: str, 
        response: "AgentResponse", 
        query: str,
        reason: str = "Evaluation system unavailable"
    ) -> AgentJudgment:
        """Create a heuristic-based fallback judgment when LLM evaluation fails.
        
        Uses rule-based scoring to provide reasonable evaluation when the primary
        LLM-based system is unavailable. Scores are conservative and transparent.
        
        Args:
            capability: The agent capability being evaluated.
            response: The agent response to evaluate.
            query: The original user query.
            reason: Explanation for why fallback was used.
            
        Returns:
            Basic judgment using heuristic scoring methods.
        """
        logger.info(f"Creating fallback judgment for {agent_name}: {reason}")
        
        # Heuristic scoring based on observable response characteristics
        base_score = 0.0
        
        # Citation quality scoring (up to 0.3 points)
        if response.citations:
            citation_score = min(0.3, len(response.citations) * 0.1)
            base_score += citation_score
        
        # Response length/substance scoring (up to 0.2 points)
        response_text = response.response_str or ""
        if len(response_text) > 50:
            length_score = min(0.2, len(response_text) / 1000)  # Scale by length
            base_score += length_score
        
        # Domain alignment scoring (up to 0.2 points)
        domain_score = self._calculate_domain_alignment(agent_name, query, response_text)
        base_score += domain_score
        
        # Conservative baseline (0.3 points for any response)
        base_score += 0.3
        
        # Cap at 0.8 to indicate this is heuristic scoring
        final_score = min(0.8, base_score)
        
        citations_count = len(response.citations or [])
        groundedness_score = (
            min(1.0, final_score + min(0.2, 0.05 * citations_count))
            if citations_count > 0
            else max(0.0, final_score * 0.6)
        )

        return AgentJudgment(
            agent_name=agent_name,
            response_str=response.response_str or "",
            prose_summary=f"Evaluation for {agent_name} (reason: {reason}). "
                          f"Response shows {'good' if final_score >= 0.6 else 'basic'} characteristics.",
            scores=JudgmentScores(
                accuracy=final_score,
                completeness=final_score,
                groundedness=groundedness_score,
                professional_tone=0.7,  # Assume reasonable tone
                clarity_coherence=final_score,
                relevance=final_score,
                usefulness=final_score
            ),
            overall_score=final_score,
            justifications=JudgmentJustifications(
                accuracy=f"Heuristic scoring based on response characteristics ({reason})",
                completeness=f"Estimated completeness from response length and structure ({reason})",
                groundedness=f"Citation quality assessment: {len(response.citations or [])} sources ({reason})",
                professional_tone=f"Assumed professional tone for system response ({reason})",
                clarity_coherence=f"Estimated from response structure and length ({reason})",
                relevance=f"Domain alignment scoring for {agent_name} ({reason})",
                usefulness=f"Estimated practical value from response characteristics ({reason})"
            )
        )

    def _format_citations(self, citations: Optional[List]) -> str:
        """Format citations for inclusion in judgment prompt.
        
        Args:
            citations: List of citation objects.
            
        Returns:
            Formatted citations string for LLM evaluation.
        """
        if not citations:
            return "None provided"
            
        formatted = []
        for i, citation in enumerate(citations, 1):
            url = getattr(citation, 'url', 'No URL')
            title = getattr(citation, 'title', 'No title')
            source = getattr(citation, 'source', 'Unknown source')
            formatted.append(f"{i}. {title} ({source}): {url}")
            
        return "\n".join(formatted)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format additional context information for judgment.
        
        Args:
            context: Context dictionary with additional information.
            
        Returns:
            Formatted context string.
        """
        if not context:
            return "None provided"
            
        formatted_items = []
        for key, value in context.items():
            formatted_items.append(f"{key}: {str(value)}")
            
        return "\n".join(formatted_items)

    def _calculate_domain_alignment(self, capability: str, query: str, response: str) -> float:
        """Calculate heuristic domain alignment score based on keywords.
        
        Args:
            capability: The agent capability being evaluated.
            query: The user query.
            response: The agent response text.
            
        Returns:
            Domain alignment score (0.0 to 0.2).
        """
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Domain-specific keyword matching
        domain_keywords = {
            'graph': ['relationship', 'interact', 'correlate', 'network', 'connect'],
            'llama_rag': ['guideline', 'nccn', 'pdf', 'document', 'protocol'],
            'biomcp': ['pubmed', 'research', 'study', 'clinical', 'gene'],
            'web': ['recent', 'latest', 'current', 'news', 'update'],
            'chitchat': ['hello', 'hi', 'thanks', 'help', 'please']
        }
        
        keywords = domain_keywords.get(capability, [])
        matches = sum(1 for keyword in keywords if keyword in query_lower or keyword in response_lower)
        
        return min(0.2, matches * 0.05)  # Max 0.2 points for domain alignment

    @staticmethod
    def _clamp_score(value: Any) -> float:
        """Clamp a value to valid score range [0.0, 1.0].
        
        Args:
            value: Value to clamp (will be converted to float).
            
        Returns:
            Clamped float value between 0.0 and 1.0.
        """
        try:
            return max(0.0, min(1.0, float(value)))
        except (ValueError, TypeError):
            return 0.0
