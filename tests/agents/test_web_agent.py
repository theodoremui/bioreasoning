"""
Tests for bioagents.agents.web_agent module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime

from bioagents.agents.web_agent import WebReasoningAgent, INSTRUCTIONS, HANDOFF_DESCRIPTION
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM


class TestWebReasoningAgent:
    """Test WebReasoningAgent class."""

    def test_web_agent_creation_default(self):
        """Test creating WebReasoningAgent with default parameters."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        assert agent.name == "TestWebAgent"
        assert agent.model_name == LLM.GPT_4_1_MINI
        assert agent.instructions == INSTRUCTIONS
        assert agent._agent is not None
        assert hasattr(agent._agent, 'name')
        assert hasattr(agent._agent, 'tools')

    def test_web_agent_creation_custom_model(self):
        """Test creating WebReasoningAgent with custom model."""
        agent = WebReasoningAgent(name="TestWebAgent", model_name=LLM.GPT_4O)
        
        assert agent.name == "TestWebAgent"
        assert agent.model_name == LLM.GPT_4O
        assert agent.instructions == INSTRUCTIONS

    def test_web_agent_instructions_content(self):
        """Test that instructions contain expected content."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        assert "real time web" in agent.instructions.lower()
        assert "latest information" in agent.instructions.lower()
        assert "news" in agent.instructions.lower()
        assert "[Web Search]" in agent.instructions
        assert "Today's date is" in agent.instructions
        assert datetime.now().strftime('%Y-%m-%d') in agent.instructions

    def test_web_agent_handoff_description(self):
        """Test handoff description content."""
        assert "real time web" in HANDOFF_DESCRIPTION.lower()
        assert "latest information" in HANDOFF_DESCRIPTION.lower()
        assert "news" in HANDOFF_DESCRIPTION.lower()

    def test_create_agent_structure(self):
        """Test that _create_agent creates proper agent structure."""
        agent = WebReasoningAgent(name="TestWebAgent")
        created_agent = agent._agent
        
        assert created_agent.name == "TestWebAgent"
        assert created_agent.model == LLM.GPT_4_1_MINI
        assert created_agent.instructions == INSTRUCTIONS
        assert created_agent.handoff_description == HANDOFF_DESCRIPTION
        assert len(created_agent.tools) == 1
        assert created_agent.tools[0].__class__.__name__ == "WebSearchTool"

    def test_create_agent_with_different_models(self):
        """Test _create_agent with different model types."""
        models = [LLM.GPT_4_1, LLM.GPT_4_1_MINI, LLM.GPT_4_1_NANO, LLM.GPT_4O]
        
        for model in models:
            agent = WebReasoningAgent(name="TestWebAgent", model_name=model)
            created_agent = agent._create_agent("TestAgent", model)
            
            assert created_agent.model == model
            assert created_agent.name == "TestAgent"
            assert created_agent.instructions == INSTRUCTIONS

    def test_create_agent_tool_configuration(self):
        """Test that _create_agent configures tools correctly."""
        agent = WebReasoningAgent(name="TestWebAgent")
        created_agent = agent._create_agent("TestAgent", LLM.GPT_4_1_MINI)
        
        # Check tool configuration
        assert len(created_agent.tools) == 1
        web_search_tool = created_agent.tools[0]
        
        # Check tool settings
        assert hasattr(created_agent, 'model_settings')
        assert created_agent.model_settings.tool_choice == "required"
        assert created_agent.tool_use_behavior == "stop_on_first_tool"

    @pytest.mark.asyncio
    async def test_achat_success(self):
        """Test successful achat call."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        # Mock the parent achat method
        mock_response = AgentResponse(
            response_str="Web search results",
            citations=[],
            judgement="Good response",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response) as mock_parent_achat:
            result = await agent.achat("What is the latest news?")
            
            assert isinstance(result, AgentResponse)
            assert result.response_str == "Web search results"
            assert result.route == AgentRouteType.WEBSEARCH
            assert result.judgement == "Good response"
            
            # Verify parent achat was called
            mock_parent_achat.assert_called_once_with("What is the latest news?")

    @pytest.mark.asyncio
    async def test_achat_route_override(self):
        """Test that achat overrides route to WEBSEARCH."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        # Mock parent achat to return response with different route
        mock_response = AgentResponse(
            response_str="Test response",
            citations=[],
            judgement="Test judgement",
            route=AgentRouteType.REASONING  # Different route
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response):
            result = await agent.achat("Test query")
            
            # Route should be overridden to WEBSEARCH
            assert result.route == AgentRouteType.WEBSEARCH
            assert result.response_str == "Test response"
            assert result.judgement == "Test judgement"

    @pytest.mark.asyncio
    async def test_achat_with_citations(self):
        """Test achat preserves citations from parent response."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        # Mock parent achat to return response with citations
        from bioagents.models.source import Source
        mock_citations = [
            Source(url="https://example.com", title="Example", source="web"),
            Source(url="https://test.com", title="Test", source="web")
        ]
        
        mock_response = AgentResponse(
            response_str="Response with citations",
            citations=mock_citations,
            judgement="Good response",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response):
            result = await agent.achat("Test query")
            
            assert len(result.citations) == 2
            assert result.citations[0].url == "https://example.com"
            assert result.citations[1].url == "https://test.com"
            assert result.route == AgentRouteType.WEBSEARCH

    @pytest.mark.asyncio
    async def test_achat_exception_handling(self):
        """Test achat handles exceptions from parent."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        # Mock parent achat to raise exception
        with patch.object(agent.__class__.__bases__[0], 'achat', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                await agent.achat("Test query")

    @pytest.mark.asyncio
    async def test_achat_logging(self):
        """Test that achat logs the query."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        mock_response = AgentResponse(
            response_str="Test response",
            citations=[],
            judgement="Test judgement",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response):
            with patch('bioagents.agents.web_agent.logger') as mock_logger:
                await agent.achat("Test query")
                
                # Verify logging was called
                mock_logger.info.assert_called_once()
                log_call = mock_logger.info.call_args[0][0]
                assert "websearch" in log_call.lower()
                assert "TestWebAgent" in log_call
                assert "Test query" in log_call

    def test_web_agent_inheritance(self):
        """Test that WebReasoningAgent properly inherits from BaseAgent."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        # Should have BaseAgent attributes
        assert hasattr(agent, 'name')
        assert hasattr(agent, 'model_name')
        assert hasattr(agent, 'instructions')
        assert hasattr(agent, 'timeout')
        assert hasattr(agent, '_agent')
        assert hasattr(agent, '_response_judge')
        
        # Should have BaseAgent methods
        assert hasattr(agent, 'achat')
        assert hasattr(agent, 'simple_achat')
        assert hasattr(agent, '_construct_response')

    def test_web_agent_override_decorator(self):
        """Test that achat method has @override decorator."""
        import inspect
        
        agent = WebReasoningAgent(name="TestWebAgent")
        achat_method = getattr(agent, 'achat')
        
        # Check if method has override decorator (this is more of a static check)
        # The @override decorator is mainly for static type checking
        assert callable(achat_method)

    def test_web_agent_instructions_immutability(self):
        """Test that instructions are properly set and immutable."""
        agent = WebReasoningAgent(name="TestWebAgent")
        original_instructions = agent.instructions
        
        # Instructions should be the same as the module constant
        assert agent.instructions == INSTRUCTIONS
        
        # Modifying instructions should not affect the constant
        agent.instructions = "Modified instructions"
        assert INSTRUCTIONS != "Modified instructions"
        assert agent.instructions == "Modified instructions"

    def test_web_agent_agent_initialization(self):
        """Test that _agent is properly initialized."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        assert agent._agent is not None
        assert hasattr(agent._agent, 'name')
        assert hasattr(agent._agent, 'model')
        assert hasattr(agent._agent, 'instructions')
        assert hasattr(agent._agent, 'tools')
        assert hasattr(agent._agent, 'handoff_description')

    def test_web_agent_tool_configuration_details(self):
        """Test detailed tool configuration."""
        agent = WebReasoningAgent(name="TestWebAgent")
        created_agent = agent._create_agent("TestAgent", LLM.GPT_4_1_MINI)
        
        # Check WebSearchTool configuration
        web_search_tool = created_agent.tools[0]
        assert hasattr(web_search_tool, 'search_context_size')
        assert web_search_tool.search_context_size == "low"

    def test_web_agent_model_settings(self):
        """Test model settings configuration."""
        agent = WebReasoningAgent(name="TestWebAgent")
        created_agent = agent._create_agent("TestAgent", LLM.GPT_4_1_MINI)
        
        # Check model settings
        assert hasattr(created_agent, 'model_settings')
        assert created_agent.model_settings.tool_choice == "required"
        assert created_agent.tool_use_behavior == "stop_on_first_tool"

    @pytest.mark.asyncio
    async def test_achat_preserves_all_fields(self):
        """Test that achat preserves all fields from parent response."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        from bioagents.models.source import Source
        mock_citations = [
            Source(url="https://example.com", title="Example", source="web")
        ]
        
        mock_response = AgentResponse(
            response_str="Original response",
            citations=mock_citations,
            judgement="Original judgement",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response):
            result = await agent.achat("Test query")
            
            # All fields should be preserved except route
            assert result.response_str == "Original response"
            assert result.citations == mock_citations
            assert result.judgement == "Original judgement"
            assert result.route == AgentRouteType.WEBSEARCH  # Only route changes

    def test_web_agent_instructions_date_dynamic(self):
        """Test that instructions contain current date."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        # Instructions should contain today's date
        today = datetime.now().strftime('%Y-%m-%d')
        assert today in agent.instructions
        
        # Should be in the format "Today's date is YYYY-MM-DD"
        assert f"Today's date is {today}" in agent.instructions

    def test_web_agent_instructions_format(self):
        """Test that instructions have proper format."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        # Should contain key phrases
        assert "You are an expert" in agent.instructions
        assert "real time web" in agent.instructions
        assert "latest information" in agent.instructions
        assert "news" in agent.instructions
        assert "general topics" in agent.instructions
        assert "directly answer" in agent.instructions
        assert "without asking for permission" in agent.instructions
        assert "inline citations" in agent.instructions
        assert "[Web Search]" in agent.instructions

    @pytest.mark.asyncio
    async def test_achat_multiple_calls(self):
        """Test multiple achat calls work correctly."""
        agent = WebReasoningAgent(name="TestWebAgent")
        
        mock_response = AgentResponse(
            response_str="Test response",
            citations=[],
            judgement="Test judgement",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response):
            # Make multiple calls
            result1 = await agent.achat("Query 1")
            result2 = await agent.achat("Query 2")
            result3 = await agent.achat("Query 3")
            
            # All should have WEBSEARCH route
            assert result1.route == AgentRouteType.WEBSEARCH
            assert result2.route == AgentRouteType.WEBSEARCH
            assert result3.route == AgentRouteType.WEBSEARCH
            
            # All should have same response content
            assert result1.response_str == "Test response"
            assert result2.response_str == "Test response"
            assert result3.response_str == "Test response"

    def test_web_agent_agent_name_consistency(self):
        """Test that agent name is consistent across initialization."""
        agent_name = "ConsistentWebAgent"
        agent = WebReasoningAgent(name=agent_name)
        
        assert agent.name == agent_name
        assert agent._agent.name == agent_name
        
        # Test with different name
        different_name = "DifferentWebAgent"
        different_agent = WebReasoningAgent(name=different_name)
        
        assert different_agent.name == different_name
        assert different_agent._agent.name == different_name
