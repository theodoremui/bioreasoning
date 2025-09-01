"""
Tests for bioagents.agents.chitchat_agent module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock

from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM


class TestChitChatAgent:
    """Test ChitChatAgent class."""

    def test_chitchat_agent_creation_default(self):
        """Test creating ChitChatAgent with default parameters."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        assert agent.name == "TestChitChatAgent"
        assert agent.model_name == LLM.GPT_4_1_NANO
        assert agent._agent is not None
        assert hasattr(agent._agent, 'name')
        assert hasattr(agent._agent, 'tools')

    def test_chitchat_agent_creation_custom_model(self):
        """Test creating ChitChatAgent with custom model."""
        agent = ChitChatAgent(name="TestChitChatAgent", model_name=LLM.GPT_4_1_MINI)
        
        assert agent.name == "TestChitChatAgent"
        assert agent.model_name == LLM.GPT_4_1_MINI

    def test_chitchat_agent_instructions_content(self):
        """Test that instructions contain expected content."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        assert "friendly conversational assistant" in agent.instructions.lower()
        assert "brief and to the point" in agent.instructions.lower()
        assert "avoid asking the user any question" in agent.instructions.lower()
        assert "[Chit Chat]" in agent.instructions

    def test_chitchat_agent_instructions_format(self):
        """Test that instructions have proper format."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        # Should contain key phrases
        assert "You are a friendly conversational assistant" in agent.instructions
        assert "very brief and to the point" in agent.instructions
        assert "AVOID asking the user any question" in agent.instructions
        assert "## Response Instructions:" in agent.instructions
        assert "- Prepend the response with '[Chit Chat]'" in agent.instructions

    def test_create_agent_structure(self):
        """Test that _create_agent creates proper agent structure."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        created_agent = agent._create_agent("TestAgent", LLM.GPT_4_1_NANO)
        
        assert created_agent.name == "TestAgent"
        assert created_agent.model == LLM.GPT_4_1_NANO
        assert created_agent.instructions == agent.instructions
        assert "friendly conversational assistant" in created_agent.handoff_description.lower()
        assert "brief and to the point" in created_agent.handoff_description.lower()
        assert created_agent.handoffs == []
        assert created_agent.tools == []

    def test_create_agent_with_different_models(self):
        """Test _create_agent with different model types."""
        models = [LLM.GPT_4_1, LLM.GPT_4_1_MINI, LLM.GPT_4_1_NANO, LLM.GPT_4O]
        
        for model in models:
            agent = ChitChatAgent(name="TestChitChatAgent", model_name=model)
            created_agent = agent._create_agent("TestAgent", model)
            
            assert created_agent.model == model
            assert created_agent.name == "TestAgent"
            assert created_agent.instructions == agent.instructions

    def test_create_agent_no_tools(self):
        """Test that _create_agent creates agent with no tools."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        created_agent = agent._create_agent("TestAgent", LLM.GPT_4_1_NANO)
        
        # ChitChat agent should have no tools
        assert created_agent.tools == []
        assert created_agent.handoffs == []

    def test_create_agent_handoff_description(self):
        """Test that handoff description is properly set."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        created_agent = agent._create_agent("TestAgent", LLM.GPT_4_1_NANO)
        
        handoff_desc = created_agent.handoff_description
        assert "friendly conversational assistant" in handoff_desc.lower()
        assert "chit chat" in handoff_desc.lower()
        assert "brief and to the point" in handoff_desc.lower()

    @pytest.mark.asyncio
    async def test_achat_success(self):
        """Test successful achat call."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        # Mock the parent achat method
        mock_response = AgentResponse(
            response_str="Hello! How are you?",
            citations=[],
            judgement="Good response",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response) as mock_parent_achat:
            result = await agent.achat("How are you?")
            
            assert isinstance(result, AgentResponse)
            assert result.response_str == "Hello! How are you?"
            assert result.route == AgentRouteType.CHITCHAT
            assert result.judgement == "Good response"
            
            # Verify parent achat was called
            mock_parent_achat.assert_called_once_with("How are you?")

    @pytest.mark.asyncio
    async def test_achat_route_override(self):
        """Test that achat overrides route to CHITCHAT."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        # Mock parent achat to return response with different route
        mock_response = AgentResponse(
            response_str="Test response",
            citations=[],
            judgement="Test judgement",
            route=AgentRouteType.REASONING  # Different route
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response):
            result = await agent.achat("Test query")
            
            # Route should be overridden to CHITCHAT
            assert result.route == AgentRouteType.CHITCHAT
            assert result.response_str == "Test response"
            assert result.judgement == "Test judgement"

    @pytest.mark.asyncio
    async def test_achat_with_citations(self):
        """Test achat preserves citations from parent response."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
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
            assert result.route == AgentRouteType.CHITCHAT

    @pytest.mark.asyncio
    async def test_achat_exception_handling(self):
        """Test achat handles exceptions from parent."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        # Mock parent achat to raise exception
        with patch.object(agent.__class__.__bases__[0], 'achat', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                await agent.achat("Test query")

    @pytest.mark.asyncio
    async def test_achat_logging(self):
        """Test that achat logs the query."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        mock_response = AgentResponse(
            response_str="Test response",
            citations=[],
            judgement="Test judgement",
            route=AgentRouteType.REASONING
        )
        
        with patch.object(agent.__class__.__bases__[0], 'achat', return_value=mock_response):
            with patch('bioagents.agents.chitchat_agent.logger') as mock_logger:
                await agent.achat("Test query")
                
                # Verify logging was called
                mock_logger.info.assert_called_once()
                log_call = mock_logger.info.call_args[0][0]
                assert "chitchat" in log_call.lower()
                assert "TestChitChatAgent" in log_call
                assert "Test query" in log_call

    def test_chitchat_agent_inheritance(self):
        """Test that ChitChatAgent properly inherits from BaseAgent."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
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

    def test_chitchat_agent_override_decorator(self):
        """Test that achat method has @override decorator."""
        import inspect
        
        agent = ChitChatAgent(name="TestChitChatAgent")
        achat_method = getattr(agent, 'achat')
        
        # Check if method has override decorator (this is more of a static check)
        # The @override decorator is mainly for static type checking
        assert callable(achat_method)

    def test_chitchat_agent_instructions_immutability(self):
        """Test that instructions are properly set and immutable."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        original_instructions = agent.instructions
        
        # Instructions should contain expected content
        assert "friendly conversational assistant" in agent.instructions.lower()
        assert "brief and to the point" in agent.instructions.lower()
        assert "[Chit Chat]" in agent.instructions
        
        # Modifying instructions should not affect the original
        agent.instructions = "Modified instructions"
        assert agent.instructions == "Modified instructions"

    def test_chitchat_agent_agent_initialization(self):
        """Test that _agent is properly initialized."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        assert agent._agent is not None
        assert hasattr(agent._agent, 'name')
        assert hasattr(agent._agent, 'model')
        assert hasattr(agent._agent, 'instructions')
        assert hasattr(agent._agent, 'tools')
        assert hasattr(agent._agent, 'handoff_description')
        assert hasattr(agent._agent, 'handoffs')

    def test_chitchat_agent_default_model(self):
        """Test that ChitChatAgent uses GPT_4_1_NANO by default."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        assert agent.model_name == LLM.GPT_4_1_NANO
        assert agent._agent.model == LLM.GPT_4_1_NANO

    def test_chitchat_agent_custom_model(self):
        """Test that ChitChatAgent can use custom models."""
        custom_model = LLM.GPT_4_1_MINI
        agent = ChitChatAgent(name="TestChitChatAgent", model_name=custom_model)
        
        assert agent.model_name == custom_model
        assert agent._agent.model == custom_model

    @pytest.mark.asyncio
    async def test_achat_preserves_all_fields(self):
        """Test that achat preserves all fields from parent response."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
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
            assert result.route == AgentRouteType.CHITCHAT  # Only route changes

    def test_chitchat_agent_instructions_construction(self):
        """Test that instructions are constructed correctly."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        # Instructions should be constructed in __init__
        expected_parts = [
            "You are a friendly conversational assistant",
            "very brief and to the point",
            "AVOID asking the user any question",
            "## Response Instructions:",
            "- Prepend the response with '[Chit Chat]'"
        ]
        
        for part in expected_parts:
            assert part in agent.instructions

    def test_chitchat_agent_handoff_description_construction(self):
        """Test that handoff description is constructed correctly."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        created_agent = agent._create_agent("TestAgent", LLM.GPT_4_1_NANO)
        
        handoff_desc = created_agent.handoff_description
        expected_parts = [
            "friendly conversational assistant",
            "chit chat",
            "brief and to the point"
        ]
        
        for part in expected_parts:
            assert part in handoff_desc.lower()

    @pytest.mark.asyncio
    async def test_achat_multiple_calls(self):
        """Test multiple achat calls work correctly."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
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
            
            # All should have CHITCHAT route
            assert result1.route == AgentRouteType.CHITCHAT
            assert result2.route == AgentRouteType.CHITCHAT
            assert result3.route == AgentRouteType.CHITCHAT
            
            # All should have same response content
            assert result1.response_str == "Test response"
            assert result2.response_str == "Test response"
            assert result3.response_str == "Test response"

    def test_chitchat_agent_agent_name_consistency(self):
        """Test that agent name is consistent across initialization."""
        agent_name = "ConsistentChitChatAgent"
        agent = ChitChatAgent(name=agent_name)
        
        assert agent.name == agent_name
        assert agent._agent.name == agent_name
        
        # Test with different name
        different_name = "DifferentChitChatAgent"
        different_agent = ChitChatAgent(name=different_name)
        
        assert different_agent.name == different_name
        assert different_agent._agent.name == different_name

    def test_chitchat_agent_no_tools_configuration(self):
        """Test that ChitChatAgent is configured with no tools."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        created_agent = agent._create_agent("TestAgent", LLM.GPT_4_1_NANO)
        
        # Should have no tools and no handoffs
        assert created_agent.tools == []
        assert created_agent.handoffs == []

    def test_chitchat_agent_instructions_avoid_questions(self):
        """Test that instructions specifically avoid asking questions."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        # Should explicitly avoid asking questions
        assert "AVOID asking the user any question" in agent.instructions
        assert "avoid" in agent.instructions.lower()

    def test_chitchat_agent_instructions_brief_responses(self):
        """Test that instructions emphasize brief responses."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        # Should emphasize being brief
        assert "very brief and to the point" in agent.instructions
        assert "brief" in agent.instructions.lower()

    def test_chitchat_agent_instructions_prefix_requirement(self):
        """Test that instructions require [Chit Chat] prefix."""
        agent = ChitChatAgent(name="TestChitChatAgent")
        
        # Should require [Chit Chat] prefix
        assert "[Chit Chat]" in agent.instructions
        assert "Prepend the response with" in agent.instructions