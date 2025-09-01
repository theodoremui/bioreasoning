# ------------------------------------------------------------------------------
# tests/judge/test_response_judge.py
#
# Comprehensive test suite for ResponseJudge and related components.
# Tests all functionality including normal operation, error handling,
# fallback mechanisms, and edge cases.
#
# Author: Theodore Mui
# Date: 2025-01-31
# ------------------------------------------------------------------------------

import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.judge import (
    ResponseJudge, 
    AgentJudgment, 
    JudgmentScores, 
    JudgmentJustifications,
    HALOJudgmentSummary,
    ResponseJudgeInterface
)
from bioagents.judge.interfaces import JudgmentError
from bioagents.models.llms import LLM


class MockCitation:
    """Mock citation object for testing."""
    def __init__(self, title: str = "Test Title", url: str = "https://example.com", source: str = "test"):
        self.title = title
        self.url = url
        self.source = source


@pytest.fixture
def mock_llm():
    """Fixture providing a mocked LLM instance."""
    with patch('bioagents.judge.response_judge.LLM') as mock_llm_class:
        mock_instance = AsyncMock()
        mock_llm_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def response_judge(mock_llm):
    """Fixture providing a ResponseJudge instance with mocked LLM."""
    return ResponseJudge(model_name=LLM.GPT_4_1_MINI, timeout=15)


@pytest.fixture
def sample_response():
    """Fixture providing a sample AgentResponse for testing."""
    citations = [MockCitation("Test Paper", "https://example.com/paper", "pubmed")]
    return AgentResponse(
        response_str="This is a test response with good information.",
        citations=citations,
        route=AgentRouteType.REASONING
    )


@pytest.fixture
def valid_judgment_data():
    """Fixture providing valid judgment data structure."""
    return {
        "prose_summary": "Good response with accurate information and proper citations.",
        "scores": {
            "accuracy": 0.8,
            "completeness": 0.7,
            "groundedness": 0.9,
            "professional_tone": 0.8,
            "clarity_coherence": 0.8,
            "relevance": 0.9,
            "usefulness": 0.8
        },
        "overall_score": 0.81,
        "justifications": {
            "accuracy": "Response contains factually correct information",
            "completeness": "Covers most aspects but could be more comprehensive",
            "groundedness": "Strong citation support",
            "professional_tone": "Appropriate and professional",
            "clarity_coherence": "Clear and well-structured",
            "relevance": "Highly relevant to the query",
            "usefulness": "Very useful for the user"
        }
    }


class TestResponseJudgeInterface:
    """Test the ResponseJudgeInterface compliance."""
    
    def test_interface_implementation(self, response_judge):
        """Test that ResponseJudge properly implements the interface."""
        assert isinstance(response_judge, ResponseJudgeInterface)
        
        # Check required methods exist
        assert hasattr(response_judge, 'judge_response')
        assert hasattr(response_judge, 'create_fallback_judgment')
        assert hasattr(response_judge, 'model_name')
        assert hasattr(response_judge, 'timeout')
        
    def test_properties(self, response_judge):
        """Test that properties return expected values."""
        assert response_judge.model_name == LLM.GPT_4_1_MINI
        assert response_judge.timeout == 15


class TestResponseJudgeInitialization:
    """Test ResponseJudge initialization and configuration."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        with patch('bioagents.judge.response_judge.LLM') as mock_llm:
            judge = ResponseJudge()
            assert judge.model_name == LLM.GPT_4_1_MINI
            assert judge.timeout == 15
            mock_llm.assert_called_once_with(model_name=LLM.GPT_4_1_MINI, timeout=15)
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        with patch('bioagents.judge.response_judge.LLM') as mock_llm:
            judge = ResponseJudge(
                model_name="custom-model", 
                timeout=30, 
                temperature=0.5,
                enable_schema_mode=False
            )
            assert judge.model_name == "custom-model"
            assert judge.timeout == 30
            mock_llm.assert_called_once_with(model_name="custom-model", timeout=30)


class TestResponseJudging:
    """Test the main response judging functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_judgment_with_schema(self, response_judge, mock_llm, sample_response, valid_judgment_data):
        """Test successful judgment using JSON schema mode."""
        # Mock LLM response
        mock_llm.achat_completion.return_value = json.dumps(valid_judgment_data)
        
        judgment = await response_judge.judge_response("graph", sample_response, "Test query")
        
        assert isinstance(judgment, AgentJudgment)
        assert judgment.overall_score == 0.81
        acc_text = judgment.justifications.accuracy.lower()
        assert ("accurate" in acc_text) or ("factually correct" in acc_text) or ("correct" in acc_text)
        
        # Verify LLM was called with correct parameters
        assert mock_llm.achat_completion.called
        call_args = mock_llm.achat_completion.call_args
        assert "json_schema" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_successful_judgment_fallback_json_object(self, response_judge, mock_llm, sample_response, valid_judgment_data):
        """Test successful judgment when schema mode fails but json_object works."""
        # First call (schema mode) fails, second call (json_object) succeeds
        mock_llm.achat_completion.side_effect = [
            Exception("Schema mode not supported"),
            json.dumps(valid_judgment_data)
        ]
        
        judgment = await response_judge.judge_response("biomcp", sample_response, "Test query")
        
        assert isinstance(judgment, AgentJudgment)
        assert judgment.overall_score == 0.81
        assert mock_llm.achat_completion.call_count == 2
    
    @pytest.mark.asyncio
    async def test_judgment_with_context(self, response_judge, mock_llm, sample_response, valid_judgment_data):
        """Test judgment with additional context information."""
        mock_llm.achat_completion.return_value = json.dumps(valid_judgment_data)
        
        context = {"domain": "medical", "urgency": "high"}
        judgment = await response_judge.judge_response("web", sample_response, "Test query", context)
        
        assert isinstance(judgment, AgentJudgment)
        
        # Verify context was included in prompt
        call_args = mock_llm.achat_completion.call_args[0][0]
        assert "domain: medical" in call_args
        assert "urgency: high" in call_args
    
    @pytest.mark.asyncio
    async def test_judgment_with_no_citations(self, response_judge, mock_llm, valid_judgment_data):
        """Test judgment of response with no citations."""
        mock_llm.achat_completion.return_value = json.dumps(valid_judgment_data)
        
        response_no_citations = AgentResponse(
            response_str="Response without citations",
            citations=[],
            route=AgentRouteType.REASONING
        )
        
        judgment = await response_judge.judge_response("chitchat", response_no_citations, "Hello")
        
        assert isinstance(judgment, AgentJudgment)
        
        # Verify "None provided" appears in citations section of prompt
        call_args = mock_llm.achat_completion.call_args[0][0]
        assert "Citations**:\nNone provided" in call_args
    
    @pytest.mark.asyncio
    async def test_llm_failure_triggers_fallback(self, response_judge, mock_llm, sample_response):
        """Test that LLM failures trigger fallback judgment."""
        # Mock LLM to fail
        mock_llm.achat_completion.side_effect = Exception("LLM service unavailable")
        
        judgment = await response_judge.judge_response("graph", sample_response, "Test query")
        
        assert isinstance(judgment, AgentJudgment)
        assert ("Heuristic" in judgment.prose_summary) or ("Evaluation" in judgment.prose_summary)
        assert "LLM evaluation failed" in judgment.justifications.accuracy
        # Fallback scores should be reasonable but capped
        assert 0.0 <= judgment.overall_score <= 0.8


class TestFallbackJudgment:
    """Test the fallback judgment functionality."""
    
    def test_fallback_with_citations(self, response_judge):
        """Test fallback judgment for response with citations."""
        citations = [MockCitation() for _ in range(3)]
        response = AgentResponse(
            response_str="A detailed response with multiple citations and good content.",
            citations=citations,
            route=AgentRouteType.REASONING
        )
        
        judgment = response_judge.create_fallback_judgment("biomcp", response, "medical query", "Test fallback")
        
        assert isinstance(judgment, AgentJudgment)
        assert judgment.overall_score > 0.5  # Should be decent with citations
        assert judgment.scores.groundedness >= judgment.scores.accuracy  # Citations boost groundedness
        assert "Test fallback" in judgment.prose_summary
    
    def test_fallback_without_citations(self, response_judge):
        """Test fallback judgment for response without citations."""
        response = AgentResponse(
            response_str="A short response without citations.",
            citations=[],
            route=AgentRouteType.REASONING
        )
        
        judgment = response_judge.create_fallback_judgment("web", response, "news query", "No citations")
        
        assert isinstance(judgment, AgentJudgment)
        assert judgment.scores.groundedness < judgment.scores.accuracy  # No citations hurts groundedness
        assert "No citations" in judgment.prose_summary
    
    def test_fallback_domain_alignment(self, response_judge):
        """Test that domain alignment affects fallback scoring."""
        response = AgentResponse(
            response_str="This shows relationships and interactions between entities in the network.",
            citations=[],
            route=AgentRouteType.REASONING
        )
        
        # Graph capability with graph-related terms should score higher
        judgment_graph = response_judge.create_fallback_judgment("graph", response, "show relationships")
        judgment_web = response_judge.create_fallback_judgment("web", response, "show relationships")
        
        assert judgment_graph.overall_score >= judgment_web.overall_score
    
    def test_fallback_empty_response(self, response_judge):
        """Test fallback judgment for empty or None response."""
        response = AgentResponse(
            response_str="",
            citations=[],
            route=AgentRouteType.REASONING
        )
        
        judgment = response_judge.create_fallback_judgment("chitchat", response, "hello", "Empty response")
        
        assert isinstance(judgment, AgentJudgment)
        assert judgment.overall_score >= 0.3  # Should still get baseline score
        assert "Empty response" in judgment.prose_summary


class TestDataValidation:
    """Test data validation and normalization."""
    
    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, response_judge, mock_llm, sample_response):
        """Test handling of invalid JSON from LLM."""
        mock_llm.achat_completion.return_value = "invalid json content"
        
        # Should fall back to heuristic judgment
        judgment = await response_judge.judge_response("graph", sample_response, "Test query")
        assert isinstance(judgment, AgentJudgment)
        assert ("Heuristic" in judgment.prose_summary) or ("Evaluation" in judgment.prose_summary)
    
    @pytest.mark.asyncio
    async def test_incomplete_judgment_data(self, response_judge, mock_llm, sample_response):
        """Test handling of incomplete judgment data from LLM."""
        incomplete_data = {
            "prose_summary": "Good response",
            "scores": {"accuracy": 0.8, "completeness": 0.7},  # Missing some scores
            # Missing justifications
        }
        mock_llm.achat_completion.return_value = json.dumps(incomplete_data)
        
        judgment = await response_judge.judge_response("biomcp", sample_response, "Test query")
        
        assert isinstance(judgment, AgentJudgment)
        # Should have filled in missing scores and justifications
        assert hasattr(judgment.scores, 'groundedness')
        assert hasattr(judgment.justifications, 'relevance')
    
    @pytest.mark.asyncio
    async def test_score_clamping(self, response_judge, mock_llm, sample_response):
        """Test that scores are properly clamped to [0.0, 1.0] range."""
        invalid_scores = {
            "prose_summary": "Test response",
            "scores": {
                "accuracy": -0.5,  # Below 0
                "completeness": 1.5,  # Above 1
                "groundedness": "invalid",  # Non-numeric
                "professional_tone": 0.5,
                "clarity_coherence": 0.7,
                "relevance": 0.8,
                "usefulness": 0.9
            },
            "justifications": {
                "accuracy": "Test", "completeness": "Test", "groundedness": "Test",
                "professional_tone": "Test", "clarity_coherence": "Test", 
                "relevance": "Test", "usefulness": "Test"
            }
        }
        mock_llm.achat_completion.return_value = json.dumps(invalid_scores)
        
        judgment = await response_judge.judge_response("web", sample_response, "Test query")
        
        assert 0.0 <= judgment.scores.accuracy <= 1.0
        assert 0.0 <= judgment.scores.completeness <= 1.0
        assert 0.0 <= judgment.scores.groundedness <= 1.0
        assert judgment.scores.accuracy == 0.0  # -0.5 clamped to 0.0
        assert judgment.scores.completeness == 1.0  # 1.5 clamped to 1.0


class TestHelperMethods:
    """Test helper methods and utilities."""
    
    def test_format_citations(self, response_judge):
        """Test citation formatting for prompts."""
        citations = [
            MockCitation("Paper 1", "https://example.com/1", "pubmed"),
            MockCitation("Paper 2", "https://example.com/2", "arxiv"),
        ]
        
        formatted = response_judge._format_citations(citations)
        
        assert "1. Paper 1 (pubmed): https://example.com/1" in formatted
        assert "2. Paper 2 (arxiv): https://example.com/2" in formatted
    
    def test_format_citations_empty(self, response_judge):
        """Test citation formatting with empty list."""
        formatted = response_judge._format_citations([])
        assert formatted == "None provided"
        
        formatted_none = response_judge._format_citations(None)
        assert formatted_none == "None provided"
    
    def test_format_context(self, response_judge):
        """Test context formatting for prompts."""
        context = {"domain": "medical", "priority": "high", "source": "clinical"}
        formatted = response_judge._format_context(context)
        
        assert "domain: medical" in formatted
        assert "priority: high" in formatted
        assert "source: clinical" in formatted
    
    def test_format_context_empty(self, response_judge):
        """Test context formatting with empty/None context."""
        assert response_judge._format_context({}) == "None provided"
        assert response_judge._format_context(None) == "None provided"
    
    def test_domain_alignment_scoring(self, response_judge):
        """Test domain alignment scoring logic."""
        # Test graph domain keywords
        score_graph = response_judge._calculate_domain_alignment("graph", "show relationships", "network interactions")
        score_other = response_judge._calculate_domain_alignment("web", "show relationships", "network interactions")
        
        assert score_graph > score_other
        assert 0.0 <= score_graph <= 0.2
    
    def test_clamp_score_utility(self, response_judge):
        """Test the score clamping utility method."""
        assert ResponseJudge._clamp_score(-0.5) == 0.0
        assert ResponseJudge._clamp_score(1.5) == 1.0
        assert ResponseJudge._clamp_score(0.5) == 0.5
        assert ResponseJudge._clamp_score("invalid") == 0.0
        assert ResponseJudge._clamp_score(None) == 0.0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_critical_failure_raises_judgment_error(self, response_judge):
        """Test that critical failures raise JudgmentError."""
        # Mock to simulate critical failure in validation
        with patch.object(response_judge, '_validate_and_normalize_judgment') as mock_validate:
            mock_validate.side_effect = Exception("Critical validation error")
            
            with pytest.raises(JudgmentError) as exc_info:
                await response_judge.judge_response("graph", Mock(), "query")
            
            assert "Critical judgment failure" in str(exc_info.value)
            assert exc_info.value.capability == "graph"
    
    def test_judgment_error_creation(self):
        """Test JudgmentError creation and string representation."""
        original_error = ValueError("Original error")
        error = JudgmentError("Test error", "test_capability", original_error)
        
        assert error.capability == "test_capability"
        assert error.original_error == original_error
        assert "[test_capability]" in str(error)
        assert "Original error" in str(error)
    
    @pytest.mark.asyncio 
    async def test_malformed_response_handling(self, response_judge, mock_llm):
        """Test handling of malformed AgentResponse objects."""
        # Create response with None values
        malformed_response = AgentResponse(
            response_str=None,
            citations=None,
            route=AgentRouteType.REASONING
        )
        
        # Mock LLM to avoid actual calls in this error handling test
        mock_llm.achat_completion.side_effect = Exception("Simulated failure")
        
        # Should not crash, should use fallback
        judgment = await response_judge.judge_response("test", malformed_response, "query")
        assert isinstance(judgment, AgentJudgment)


class TestIntegrationWithModels:
    """Test integration with Pydantic models."""
    
    def test_judgment_scores_methods(self):
        """Test JudgmentScores model methods."""
        scores = JudgmentScores(
            accuracy=0.8, completeness=0.7, groundedness=0.9,
            professional_tone=0.8, clarity_coherence=0.8,
            relevance=0.9, usefulness=0.8
        )
        
        avg_score = scores.get_average_score()
        assert abs(avg_score - 0.81) < 0.01  # Should be about 0.814
        
        weights = {
            'accuracy': 0.3, 'groundedness': 0.3, 'relevance': 0.2,
            'usefulness': 0.1, 'completeness': 0.05, 
            'clarity_coherence': 0.03, 'professional_tone': 0.02
        }
        weighted_score = scores.get_weighted_score(weights)
        assert 0.0 <= weighted_score <= 1.0
    
    def test_agent_judgment_methods(self):
        """Test AgentJudgment model methods."""
        scores = JudgmentScores(
            accuracy=0.9, completeness=0.8, groundedness=0.9,
            professional_tone=0.8, clarity_coherence=0.8,
            relevance=0.9, usefulness=0.8
        )
        justifications = JudgmentJustifications(
            accuracy="High accuracy", completeness="Good coverage",
            groundedness="Strong evidence", professional_tone="Appropriate",
            clarity_coherence="Clear", relevance="Very relevant", 
            usefulness="Highly useful"
        )
        
        judgment = AgentJudgment(
            agent_name="a", response_str="r",
            prose_summary="Excellent response",
            scores=scores,
            overall_score=0.86,
            justifications=justifications
        )
        
        assert judgment.is_high_quality(threshold=0.8) == True
        assert judgment.is_high_quality(threshold=0.9) == False
        
        improvement_areas = judgment.get_improvement_areas(threshold=0.85)
        assert 'completeness' in improvement_areas  # 0.8 < 0.85
        assert 'accuracy' not in improvement_areas  # 0.9 >= 0.85
    
    def test_halo_judgment_summary_methods(self):
        """Test HALOJudgmentSummary model methods."""
        # Create multiple agent judgments
        judgment1 = Mock(overall_score=0.9)
        judgment2 = Mock(overall_score=0.7) 
        judgment3 = Mock(overall_score=0.4)
        
        # Build minimal valid AgentJudgment instances
        from bioagents.judge.models import AgentJudgment as AJ, JudgmentScores as JS, JudgmentJustifications as JJ
        js = JS(accuracy=0.5, completeness=0.5, groundedness=0.5, professional_tone=0.5, clarity_coherence=0.5, relevance=0.5, usefulness=0.5)
        jj = JJ(accuracy="a", completeness="b", groundedness="c", professional_tone="d", clarity_coherence="e", relevance="f", usefulness="g")
        summary = HALOJudgmentSummary(
            individual_judgments={
                "agent1": AJ(prose_summary="s", scores=js, overall_score=0.9, justifications=jj),
                "agent2": AJ(prose_summary="s", scores=js, overall_score=0.7, justifications=jj), 
                "agent3": AJ(prose_summary="s", scores=js, overall_score=0.4, justifications=jj)
            },
            overall_halo_score=0.67,
            synthesis_notes="Mixed performance across agents"
        )
        
        top_performers = summary.get_top_performers(count=2)
        assert len(top_performers) == 2
        assert "agent1" in top_performers
        assert "agent2" in top_performers
        
        low_performers = summary.get_low_performers(threshold=0.6)
        assert len(low_performers) == 1
        assert "agent3" in low_performers
        
        score_dist = summary.get_score_distribution()
        assert score_dist['min'] == 0.4
        assert score_dist['max'] == 0.9
        assert abs(score_dist['mean'] - 0.67) < 0.01
        assert score_dist['median'] == 0.7


@pytest.mark.asyncio
async def test_end_to_end_judgment_workflow():
    """Test complete end-to-end judgment workflow."""
    with patch('bioagents.judge.response_judge.LLM') as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm_class.return_value = mock_llm
        
        # Mock successful LLM response
        judgment_data = {
            "prose_summary": "Comprehensive response with excellent grounding.",
            "scores": {
                "accuracy": 0.9, "completeness": 0.8, "groundedness": 0.95,
                "professional_tone": 0.9, "clarity_coherence": 0.85,
                "relevance": 0.9, "usefulness": 0.85
            },
            "overall_score": 0.88,
            "justifications": {
                "accuracy": "Highly accurate medical information",
                "completeness": "Covers all major aspects", 
                "groundedness": "Excellent citation support",
                "professional_tone": "Very professional",
                "clarity_coherence": "Clear and well-structured",
                "relevance": "Directly addresses query",
                "usefulness": "Very actionable information"
            }
        }
        mock_llm.achat_completion.return_value = json.dumps(judgment_data)
        
        # Create judge and test response
        judge = ResponseJudge(model_name=LLM.GPT_4_1_MINI)
        
        citations = [MockCitation("Clinical Study", "https://pubmed.gov/123", "pubmed")]
        response = AgentResponse(
            response_str="Based on clinical evidence, the recommended treatment approach is...",
            citations=citations,
            route=AgentRouteType.BIOMCP
        )
        
        # Execute judgment
        judgment = await judge.judge_response("biomcp", response, "What is the treatment for condition X?")
        
        # Verify comprehensive results
        assert isinstance(judgment, AgentJudgment)
        assert judgment.overall_score == 0.88
        assert judgment.is_high_quality()
        assert len(judgment.get_improvement_areas()) == 0  # All scores above default threshold
        assert "Comprehensive response" in judgment.prose_summary
        # Loosen assertion to cover general accuracy phrasing
        assert "accurate" in judgment.justifications.accuracy.lower()
