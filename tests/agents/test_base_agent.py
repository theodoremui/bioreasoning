"""
Tests for bioagents.agents.base_agent module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM


class ConcreteBaseAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(self, name: str, model_name: str = LLM.GPT_4_1_MINI):
        super().__init__(name, model_name)
        # Create a mock agent for testing
        self._agent = MagicMock()


class TestBaseAgent:
    """Test BaseAgent abstract base class."""

    def test_base_agent_creation_default(self):
        """Test creating BaseAgent with default parameters."""
        agent = ConcreteBaseAgent(name="TestAgent")

        assert agent.name == "TestAgent"
        assert agent.model_name == LLM.GPT_4_1_MINI
        assert agent.instructions == "You are a reasoning conversational agent."
        assert agent.timeout == 60

    def test_base_agent_creation_custom(self):
        """Test creating BaseAgent with custom parameters."""
        custom_instructions = "You are a custom agent."
        agent = ConcreteBaseAgent(
            name="CustomAgent",
            model_name=LLM.GPT_4O,
        )
        agent.instructions = custom_instructions
        agent.timeout = 120

        assert agent.name == "CustomAgent"
        assert agent.model_name == LLM.GPT_4O
        assert agent.instructions == custom_instructions
        assert agent.timeout == 120

    def test_construct_response_minimal(self):
        """Test _construct_response with minimal RunResult."""
        agent = ConcreteBaseAgent(name="TestAgent")

        # Create mock RunResult
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response"
        mock_run_result.new_items = []

        response = agent._construct_response(mock_run_result)

        assert isinstance(response, AgentResponse)
        assert response.response_str == "Test response"
        assert response.citations == []
        assert response.judgement == ""
        assert response.route == AgentRouteType.REASONING

    def test_construct_response_with_citations(self):
        """Test _construct_response with web citations."""
        agent = ConcreteBaseAgent(name="TestAgent")

        # Create mock RunResult with citations
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response with citations"

        # Mock citation annotation
        mock_annotation = MagicMock()
        mock_annotation.type = "url_citation"
        mock_annotation.url = "https://example.com"
        mock_annotation.title = "Example Article"
        mock_annotation.start_index = 0
        mock_annotation.end_index = 10

        mock_content = MagicMock()
        mock_content.annotations = [mock_annotation]
        mock_content.text = "Test snippet from citation"

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item.content = [mock_content]

        mock_run_result.new_items = [mock_item]

        response = agent._construct_response(
            mock_run_result, "judge", AgentRouteType.WEBSEARCH
        )

        assert response.response_str == "Test response with citations"
        assert response.judgement == "judge"
        assert response.route == AgentRouteType.WEBSEARCH
        assert len(response.citations) == 1
        assert response.citations[0].url == "https://example.com"
        assert response.citations[0].title == "Example Article"
        assert response.citations[0].source == "web"

    def test_construct_response_no_citations(self):
        """Test _construct_response with items but no citations."""
        agent = ConcreteBaseAgent(name="TestAgent")

        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response"

        # Mock item without annotations
        mock_content = MagicMock()
        mock_content.annotations = []

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item.content = [mock_content]

        mock_run_result.new_items = [mock_item]

        response = agent._construct_response(mock_run_result)

        assert response.citations == []

    @patch("agents.Runner.run")
    @patch("bioagents.agents.base_agent.gen_trace_id")
    @patch("bioagents.agents.base_agent.trace")
    @pytest.mark.asyncio
    async def test_achat_success(self, mock_trace, mock_gen_trace_id, mock_runner_run):
        """Test successful achat call."""
        # Setup mocks
        mock_gen_trace_id.return_value = "test-trace-id"
        mock_trace.return_value.__enter__ = MagicMock()
        mock_trace.return_value.__exit__ = MagicMock()

        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response"
        mock_run_result.new_items = []

        mock_runner_run.return_value = mock_run_result

        # Create agent and test
        agent = ConcreteBaseAgent(name="TestAgent")

        result = await agent.achat("Test query")

        assert isinstance(result, AgentResponse)
        assert result.response_str == "Test response"

        # Verify mocks were called correctly
        mock_runner_run.assert_called_once()
        mock_gen_trace_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_achat_no_agent_error(self):
        """Test achat raises error when agent is not initialized."""
        agent = ConcreteBaseAgent(name="TestAgent")
        agent._agent = None

        with pytest.raises(ValueError, match="Agent not initialized"):
            await agent.achat("Test query")

    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_timeout_error(self, mock_runner_run):
        """Test achat handles timeout error."""
        import asyncio

        # Setup mock to raise timeout
        mock_runner_run.side_effect = asyncio.TimeoutError()

        agent = ConcreteBaseAgent(name="TestAgent")
        agent.timeout = 1  # Short timeout for testing

        with pytest.raises(asyncio.TimeoutError):
            await agent.achat("Test query")

    @patch("agents.Runner.run")
    @patch("bioagents.agents.base_agent.logger")
    @pytest.mark.asyncio
    async def test_achat_general_error(self, mock_logger, mock_runner_run):
        """Test achat handles general errors."""
        # Setup mock to raise general error
        test_error = Exception("Test error")
        mock_runner_run.side_effect = test_error

        agent = ConcreteBaseAgent(name="TestAgent")

        with pytest.raises(Exception, match="Test error"):
            await agent.achat("Test query")

        # Verify error was logged
        mock_logger.error.assert_called()

    def test_reasoning_agent_different_models(self):
        """Test BaseAgent with different model types."""
        models = [LLM.GPT_4_1, LLM.GPT_4_1_MINI, LLM.GPT_4_1_NANO, LLM.GPT_4O]

        for model in models:
            agent = ConcreteBaseAgent(name=f"Agent-{model}", model_name=model)
            assert agent.model_name == model

    def test_reasoning_agent_custom_instructions(self):
        """Test BaseAgent with custom instructions."""
        custom_instructions = "You are a specialized test agent with custom behavior."
        agent = ConcreteBaseAgent(name="CustomAgent")
        agent.instructions = custom_instructions

        assert agent.instructions == custom_instructions

    def test_reasoning_agent_timeout_values(self):
        """Test BaseAgent with different timeout values."""
        timeouts = [30, 60, 120, 300]

        for timeout in timeouts:
            agent = ConcreteBaseAgent(name="TestAgent")
            agent.timeout = timeout
            assert agent.timeout == timeout

    def test_construct_response_multiple_items(self):
        """Test _construct_response with multiple items containing citations."""
        agent = ConcreteBaseAgent(name="TestAgent")

        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response"

        # Create multiple mock items with different citation types
        items = []
        for i in range(3):
            mock_annotation = MagicMock()
            mock_annotation.type = "url_citation"
            mock_annotation.url = f"https://example{i}.com"
            mock_annotation.title = f"Article {i}"
            mock_annotation.start_index = 0
            mock_annotation.end_index = 5

            mock_content = MagicMock()
            mock_content.annotations = [mock_annotation]
            mock_content.text = f"Text {i}"

            mock_item = MagicMock()
            mock_item.type = "message_output_item"
            mock_item.raw_item.content = [mock_content]

            items.append(mock_item)

        mock_run_result.new_items = items

        response = agent._construct_response(mock_run_result)

        assert len(response.citations) == 3
        for i, citation in enumerate(response.citations):
            assert citation.url == f"https://example{i}.com"
            assert citation.title == f"Article {i}"

    @pytest.mark.asyncio
    async def test_simple_achat_success(self):
        """Test simple_achat method with successful response."""
        agent = ConcreteBaseAgent(name="TestAgent")
        
        # Mock the achat method to return a response with judgement
        mock_response = AgentResponse(
            response_str="Test response",
            citations=[],
            judgement="**Score**: 0.85\n**Assessment**: Good response",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent, 'achat', return_value=mock_response):
            result = await agent.simple_achat("Test query")
            
            expected = "Test response\tScore: 0.85"
            assert result == expected

    @pytest.mark.asyncio
    async def test_simple_achat_no_score(self):
        """Test simple_achat method when judgement has no score."""
        agent = ConcreteBaseAgent(name="TestAgent")
        
        # Mock the achat method to return a response without score
        mock_response = AgentResponse(
            response_str="Test response",
            citations=[],
            judgement="No score in this judgement",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent, 'achat', return_value=mock_response):
            result = await agent.simple_achat("Test query")
            
            expected = "Test response\tScore: "
            assert result == expected

    @pytest.mark.asyncio
    async def test_simple_achat_different_score_formats(self):
        """Test simple_achat method with different score formats."""
        agent = ConcreteBaseAgent(name="TestAgent")
        
        test_cases = [
            ("**Score**: 0.75", "0.75"),
            ("Score: 0.90", "0.90"),
            ("**Score**: 1.0", "1.0"),
            ("Score: 0", "0"),
            ("**Score**: 0.123", "0.123"),
        ]
        
        for judgement, expected_score in test_cases:
            mock_response = AgentResponse(
                response_str="Test response",
                citations=[],
                judgement=judgement,
                route=AgentRouteType.REASONING
            )
            
            with patch.object(agent, 'achat', return_value=mock_response):
                result = await agent.simple_achat("Test query")
                
                expected = f"Test response\tScore: {expected_score}"
                assert result == expected

    @pytest.mark.asyncio
    async def test_simple_achat_exception_handling(self):
        """Test simple_achat method handles exceptions gracefully."""
        agent = ConcreteBaseAgent(name="TestAgent")
        
        # Mock achat to raise an exception
        with patch.object(agent, 'achat', side_effect=Exception("Test error")):
            with patch('bioagents.agents.base_agent.logger') as mock_logger:
                result = await agent.simple_achat("Test query")
                
                # Should return empty string when exception occurs
                assert result == ""
                mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_simple_achat_with_undefined_response(self):
        """Test simple_achat method when response is undefined after exception."""
        agent = ConcreteBaseAgent(name="TestAgent")
        
        # Mock achat to raise an exception and response to be undefined
        with patch.object(agent, 'achat', side_effect=Exception("Test error")):
            with patch('bioagents.agents.base_agent.logger') as mock_logger:
                # This should not cause an error even if response is undefined
                result = await agent.simple_achat("Test query")
                
                assert result == ""
                mock_logger.error.assert_called_once()

    def test_construct_response_duplicate_urls(self):
        """Test _construct_response filters duplicate URLs."""
        agent = ConcreteBaseAgent(name="TestAgent")

        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response"

        # Create mock items with duplicate URLs
        mock_annotation1 = MagicMock()
        mock_annotation1.type = "url_citation"
        mock_annotation1.url = "https://example.com"
        mock_annotation1.title = "Article 1"
        mock_annotation1.start_index = 0
        mock_annotation1.end_index = 5

        mock_annotation2 = MagicMock()
        mock_annotation2.type = "url_citation"
        mock_annotation2.url = "https://example.com"  # Duplicate URL
        mock_annotation2.title = "Article 2"
        mock_annotation2.start_index = 0
        mock_annotation2.end_index = 5

        mock_content1 = MagicMock()
        mock_content1.annotations = [mock_annotation1]
        mock_content1.text = "Text 1"

        mock_content2 = MagicMock()
        mock_content2.annotations = [mock_annotation2]
        mock_content2.text = "Text 2"

        mock_item1 = MagicMock()
        mock_item1.type = "message_output_item"
        mock_item1.raw_item.content = [mock_content1]

        mock_item2 = MagicMock()
        mock_item2.type = "message_output_item"
        mock_item2.raw_item.content = [mock_content2]

        mock_run_result.new_items = [mock_item1, mock_item2]

        response = agent._construct_response(mock_run_result)

        # Should only have one citation despite duplicate URLs
        assert len(response.citations) == 1
        assert response.citations[0].url == "https://example.com"
        assert response.citations[0].title == "Article 1"  # First occurrence

    def test_construct_response_non_citation_annotations(self):
        """Test _construct_response ignores non-citation annotations."""
        agent = ConcreteBaseAgent(name="TestAgent")

        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response"

        # Create mock annotation that's not a URL citation
        mock_annotation = MagicMock()
        mock_annotation.type = "other_annotation"
        mock_annotation.url = "https://example.com"
        mock_annotation.title = "Article"
        mock_annotation.start_index = 0
        mock_annotation.end_index = 5

        mock_content = MagicMock()
        mock_content.annotations = [mock_annotation]
        mock_content.text = "Text"

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item.content = [mock_content]

        mock_run_result.new_items = [mock_item]

        response = agent._construct_response(mock_run_result)

        # Should have no citations since annotation type is not url_citation
        assert len(response.citations) == 0

    def test_construct_response_missing_annotation_attributes(self):
        """Test _construct_response handles missing annotation attributes."""
        agent = ConcreteBaseAgent(name="TestAgent")

        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response"

        # Create mock annotation with missing URL
        mock_annotation = MagicMock()
        mock_annotation.type = "url_citation"
        mock_annotation.url = None  # Missing URL
        mock_annotation.title = "Article"
        mock_annotation.start_index = 0
        mock_annotation.end_index = 5

        mock_content = MagicMock()
        mock_content.annotations = [mock_annotation]
        mock_content.text = "Text"

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item.content = [mock_content]

        mock_run_result.new_items = [mock_item]

        response = agent._construct_response(mock_run_result)

        # Should have no citations since URL is missing
        assert len(response.citations) == 0
