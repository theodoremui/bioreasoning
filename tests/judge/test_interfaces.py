# ------------------------------------------------------------------------------
# tests/judge/test_interfaces.py
#
# Test suite for judgment interfaces and error handling.
# Tests abstract interfaces, custom exceptions, and contract compliance.
#
# Author: Theodore Mui
# Date: 2025-01-31
# ------------------------------------------------------------------------------

import pytest
from abc import ABC
from unittest.mock import Mock

from bioagents.judge.interfaces import ResponseJudgeInterface, JudgmentError
from bioagents.judge import ResponseJudge


class TestResponseJudgeInterface:
    """Test the ResponseJudgeInterface abstract base class."""
    
    def test_interface_is_abstract(self):
        """Test that ResponseJudgeInterface is an abstract base class."""
        assert issubclass(ResponseJudgeInterface, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ResponseJudgeInterface()
    
    def test_interface_methods_are_abstract(self):
        """Test that required methods are abstract."""
        # Create a concrete implementation that doesn't implement all methods
        class IncompleteJudge(ResponseJudgeInterface):
            @property
            def model_name(self):
                return "test"
            
            @property
            def timeout(self):
                return 10
            
            # Missing judge_response and create_fallback_judgment
        
        # Should not be able to instantiate without implementing all abstract methods
        with pytest.raises(TypeError):
            IncompleteJudge()
    
    def test_concrete_implementation_compliance(self):
        """Test that ResponseJudge properly implements the interface."""
        judge = ResponseJudge()
        
        assert isinstance(judge, ResponseJudgeInterface)
        
        # Check all required methods exist and are callable
        assert hasattr(judge, 'judge_response')
        assert callable(getattr(judge, 'judge_response'))
        
        assert hasattr(judge, 'create_fallback_judgment')
        assert callable(getattr(judge, 'create_fallback_judgment'))
        
        # Check properties exist
        assert hasattr(judge, 'model_name')
        assert hasattr(judge, 'timeout')
        
        # Properties should return expected types
        assert isinstance(judge.model_name, str)
        assert isinstance(judge.timeout, int)


class TestJudgmentError:
    """Test the custom JudgmentError exception."""
    
    def test_basic_judgment_error(self):
        """Test basic JudgmentError creation."""
        error = JudgmentError("Test error message")
        
        assert isinstance(error, Exception)
        assert str(error) == "Test error message"
        assert error.capability == ""
        assert error.original_error is None
    
    def test_judgment_error_with_capability(self):
        """Test JudgmentError with capability information."""
        error = JudgmentError("Test error", capability="graph")
        
        assert error.capability == "graph"
        assert "[graph]" in str(error)
        assert "Test error" in str(error)
    
    def test_judgment_error_with_original_error(self):
        """Test JudgmentError with original exception."""
        original = ValueError("Original error")
        error = JudgmentError("Test error", original_error=original)
        
        assert error.original_error == original
        assert "Original error" in str(error)
        assert "caused by" in str(error)
    
    def test_judgment_error_with_all_parameters(self):
        """Test JudgmentError with all parameters."""
        original = RuntimeError("Database connection failed")
        error = JudgmentError(
            "Judgment system unavailable",
            capability="biomcp",
            original_error=original
        )
        
        assert error.capability == "biomcp"
        assert error.original_error == original
        
        error_str = str(error)
        assert "[biomcp]" in error_str
        assert "Judgment system unavailable" in error_str
        assert "Database connection failed" in error_str
        assert "caused by" in error_str
    
    def test_judgment_error_inheritance(self):
        """Test that JudgmentError properly inherits from Exception."""
        error = JudgmentError("Test")
        
        assert isinstance(error, Exception)
        
        # Should be catchable as Exception
        try:
            raise error
        except Exception as e:
            assert e == error
    
    def test_judgment_error_in_context(self):
        """Test JudgmentError in realistic usage context."""
        def failing_judgment_function():
            try:
                # Simulate some internal failure
                raise ConnectionError("Network timeout")
            except ConnectionError as e:
                raise JudgmentError(
                    "Failed to judge response due to network issues",
                    capability="web",
                    original_error=e
                )
        
        with pytest.raises(JudgmentError) as exc_info:
            failing_judgment_function()
        
        error = exc_info.value
        assert error.capability == "web"
        assert isinstance(error.original_error, ConnectionError)
        assert "network issues" in str(error)
        assert "Network timeout" in str(error)


class TestInterfaceContractCompliance:
    """Test that concrete implementations properly follow interface contracts."""
    
    @pytest.mark.asyncio
    async def test_judge_response_signature_compliance(self):
        """Test that judge_response method signature matches interface."""
        judge = ResponseJudge()
        
        # Mock the dependencies to avoid actual LLM calls
        from unittest import mock
        with mock.patch.object(judge, '_llm') as mock_llm:
            mock_llm.achat_completion.side_effect = Exception("Simulated failure")
            
            # Mock response object
            mock_response = Mock()
            mock_response.response_str = "Test response"
            mock_response.citations = []
            
            # Should be able to call with required parameters
            result = await judge.judge_response("test", mock_response, "test query")
            
            # Should return an AgentJudgment (even if via fallback)
            assert hasattr(result, 'overall_score')
            assert hasattr(result, 'prose_summary')
            assert hasattr(result, 'scores')
            assert hasattr(result, 'justifications')
    
    @pytest.mark.asyncio
    async def test_judge_response_with_context(self):
        """Test that judge_response handles optional context parameter."""
        judge = ResponseJudge()
        
        from unittest import mock
        with mock.patch.object(judge, '_llm') as mock_llm:
            mock_llm.achat_completion.side_effect = Exception("Simulated failure")
            
            mock_response = Mock()
            mock_response.response_str = "Test response"
            mock_response.citations = []
            
            # Should work with context parameter
            context = {"domain": "medical", "urgency": "high"}
            result = await judge.judge_response("test", mock_response, "test query", context)
            
            assert hasattr(result, 'overall_score')
            assert isinstance(result.overall_score, float)
            assert 0.0 <= result.overall_score <= 1.0
    
    def test_create_fallback_judgment_signature_compliance(self):
        """Test that create_fallback_judgment matches interface signature."""
        judge = ResponseJudge()
        
        mock_response = Mock()
        mock_response.response_str = "Test response"
        mock_response.citations = []
        
        # Test with required parameters
        result = judge.create_fallback_judgment("test", mock_response, "test query")
        assert hasattr(result, 'overall_score')
        
        # Test with optional reason parameter
        result_with_reason = judge.create_fallback_judgment(
            "test", mock_response, "test query", "Custom reason"
        )
        assert hasattr(result_with_reason, 'overall_score')
        assert "Custom reason" in result_with_reason.prose_summary
    
    def test_properties_return_correct_types(self):
        """Test that properties return values of expected types."""
        judge = ResponseJudge(model_name="test-model", timeout=25)
        
        # model_name should return string
        model_name = judge.model_name
        assert isinstance(model_name, str)
        assert model_name == "test-model"
        
        # timeout should return int
        timeout = judge.timeout
        assert isinstance(timeout, int)
        assert timeout == 25


class MockJudge(ResponseJudgeInterface):
    """Mock implementation of ResponseJudgeInterface for testing."""
    
    def __init__(self, model_name="mock", timeout=10):
        self._model_name = model_name
        self._timeout = timeout
    
    async def judge_response(self, capability, response, query, context=None):
        # Mock implementation
        from bioagents.judge.models import AgentJudgment, JudgmentScores, JudgmentJustifications
        
        scores = JudgmentScores(
            accuracy=0.5, completeness=0.5, groundedness=0.5,
            professional_tone=0.5, clarity_coherence=0.5,
            relevance=0.5, usefulness=0.5
        )
        justifications = JudgmentJustifications(
            accuracy="Mock", completeness="Mock", groundedness="Mock",
            professional_tone="Mock", clarity_coherence="Mock",
            relevance="Mock", usefulness="Mock"
        )
        
        return AgentJudgment(
            agent_name=capability,
            response_str=response.response_str,
            prose_summary="Mock judgment",
            scores=scores,
            overall_score=0.5,
            justifications=justifications
        )
    
    def create_fallback_judgment(self, capability, response, query, reason="Mock fallback"):
        from bioagents.judge.models import AgentJudgment, JudgmentScores, JudgmentJustifications
        
        scores = JudgmentScores(
            accuracy=0.3, completeness=0.3, groundedness=0.3,
            professional_tone=0.3, clarity_coherence=0.3,
            relevance=0.3, usefulness=0.3
        )
        justifications = JudgmentJustifications(
            accuracy="Mock fallback", completeness="Mock fallback", 
            groundedness="Mock fallback", professional_tone="Mock fallback",
            clarity_coherence="Mock fallback", relevance="Mock fallback",
            usefulness="Mock fallback"
        )
        
        return AgentJudgment(
            prose_summary=f"Mock fallback: {reason}",
            scores=scores,
            overall_score=0.3,
            justifications=justifications
        )
    
    @property
    def model_name(self):
        return self._model_name
    
    @property
    def timeout(self):
        return self._timeout


class TestMockImplementation:
    """Test that mock implementation properly follows interface."""
    
    @pytest.mark.asyncio
    async def test_mock_judge_implementation(self):
        """Test that MockJudge properly implements interface."""
        judge = MockJudge("test-model", 20)
        
        assert isinstance(judge, ResponseJudgeInterface)
        assert judge.model_name == "test-model"
        assert judge.timeout == 20
        
        # Test judge_response
        mock_response = Mock()
        mock_response.response_str = "Test"
        mock_response.citations = []
        
        result = await judge.judge_response("test", mock_response, "query")
        assert result.overall_score == 0.5
        assert "Mock judgment" in result.prose_summary
        
        # Test create_fallback_judgment
        fallback = judge.create_fallback_judgment("test", mock_response, "query", "Test reason")
        assert fallback.overall_score == 0.3
        assert "Test reason" in fallback.prose_summary
