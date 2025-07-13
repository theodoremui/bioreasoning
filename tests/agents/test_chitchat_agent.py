"""
Tests for bioagents.agents.chitchat_agent module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.common import AgentResponse, AgentRoute
from bioagents.models.llms import LLM


class TestChitChatAgent:
    """Test ChitChatAgent functionality."""
    
    def test_chitchat_agent_creation_default(self):
        """Test creating ChitChatAgent with default parameters."""
        agent = ChitChatAgent(name="TestChitChat")
        
        assert agent.name == "TestChitChat"
        assert agent.model_name == LLM.GPT_4_1_NANO
        assert "friendly conversational assistant" in agent.instructions
        assert "brief and to the point" in agent.instructions
        assert agent._agent is not None
    
    def test_chitchat_agent_creation_custom_model(self):
        """Test creating ChitChatAgent with custom model."""
        agent = ChitChatAgent(name="CustomChitChat", model_name=LLM.GPT_4O)
        
        assert agent.model_name == LLM.GPT_4O
    
    def test_chitchat_agent_instructions(self):
        """Test that ChitChatAgent has appropriate instructions."""
        agent = ChitChatAgent(name="TestChitChat")
        
        # Check key instruction elements
        assert "friendly" in agent.instructions
        assert "brief" in agent.instructions
        assert "AVOID asking the user any question" in agent.instructions
    
    def test_create_agent_structure(self):
        """Test that _create_agent creates proper agent structure."""
        agent = ChitChatAgent(name="TestChitChat")
        
        # Check agent properties
        assert agent._agent.name == "TestChitChat"
        assert agent._agent.model == LLM.GPT_4_1_NANO
        assert agent._agent.handoffs == []  # No handoffs for chitchat
        assert agent._agent.tools == []  # No tools for chitchat
        
        # Check handoff description
        handoff_desc = agent._agent.handoff_description
        assert "friendly conversational assistant" in str(handoff_desc)
        assert "chit chat" in str(handoff_desc)
    
    @patch('bioagents.agents.chitchat_agent.Runner')
    @pytest.mark.asyncio
    async def test_achat_success(self, mock_runner):
        """Test successful achat call."""
        # Setup mock
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Hello! Nice to meet you."
        mock_run_result.new_items = []
        
        mock_runner.run = AsyncMock(return_value=mock_run_result)
        
        # Create agent and test
        agent = ChitChatAgent(name="TestChitChat")
        
        result = await agent.achat("Hello")
        
        assert isinstance(result, AgentResponse)
        assert result.response_str == "Hello! Nice to meet you."
        assert result.route == AgentRoute.CHIT_CHAT
        assert result.citations == []
        
        # Verify runner was called with max_turns=1
        mock_runner.run.assert_called_once()
        call_args = mock_runner.run.call_args
        assert call_args.kwargs['max_turns'] == 1
    
    @patch('bioagents.agents.chitchat_agent.Runner')
    @pytest.mark.asyncio
    async def test_achat_with_different_greetings(self, mock_runner):
        """Test achat with different greeting inputs."""
        greetings = [
            ("Hello", "Hello! How can I help you today?"),
            ("Hi there", "Hi! Nice to see you."),
            ("Good morning", "Good morning! Hope you're having a great day."),
            ("How are you?", "I'm doing well, thank you for asking!"),
        ]
        
        agent = ChitChatAgent(name="TestChitChat")
        
        for input_text, expected_response in greetings:
            mock_run_result = MagicMock()
            mock_run_result.final_output = expected_response
            mock_run_result.new_items = []
            
            mock_runner.run = AsyncMock(return_value=mock_run_result)
            
            result = await agent.achat(input_text)
            
            assert result.response_str == expected_response
            assert result.route == AgentRoute.CHIT_CHAT
    
    @patch('bioagents.agents.chitchat_agent.Runner')
    @pytest.mark.asyncio
    async def test_achat_casual_conversation(self, mock_runner):
        """Test achat with casual conversation topics."""
        casual_topics = [
            "How's the weather?",
            "What's your favorite color?",
            "Tell me about yourself",
            "Do you like music?",
            "What's your opinion on coffee?",
        ]
        
        agent = ChitChatAgent(name="TestChitChat")
        
        for topic in casual_topics:
            mock_run_result = MagicMock()
            mock_run_result.final_output = f"That's an interesting question about {topic.lower()}!"
            mock_run_result.new_items = []
            
            mock_runner.run = AsyncMock(return_value=mock_run_result)
            
            result = await agent.achat(topic)
            
            assert isinstance(result, AgentResponse)
            assert result.route == AgentRoute.CHIT_CHAT
            assert result.citations == []
    
    @patch('bioagents.agents.chitchat_agent.Runner')
    @patch('bioagents.agents.chitchat_agent.logger')
    @pytest.mark.asyncio
    async def test_achat_logging(self, mock_logger, mock_runner):
        """Test that achat logs the interaction."""
        # Setup mock
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Test response"
        mock_run_result.new_items = []
        
        mock_runner.run = AsyncMock(return_value=mock_run_result)
        
        # Create agent and test
        agent = ChitChatAgent(name="TestChitChat")
        
        await agent.achat("Test query")
        
        # Verify logging was called
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "TestChitChat" in call_args
        assert "Test query" in call_args
    
    @patch('bioagents.agents.chitchat_agent.Runner')
    @pytest.mark.asyncio
    async def test_achat_error_handling(self, mock_runner):
        """Test achat error handling."""
        # Setup mock to raise error
        mock_runner.run = AsyncMock(side_effect=Exception("Test error"))
        
        agent = ChitChatAgent(name="TestChitChat")
        
        with pytest.raises(Exception, match="Test error"):
            await agent.achat("Test query")
    
    def test_chitchat_agent_inheritance(self):
        """Test that ChitChatAgent properly inherits from ReasoningAgent."""
        agent = ChitChatAgent(name="TestChitChat")
        
        # Should have inherited attributes
        assert hasattr(agent, 'name')
        assert hasattr(agent, 'model_name')
        assert hasattr(agent, 'instructions')
        assert hasattr(agent, 'timeout')
        assert hasattr(agent, '_construct_response')
    
    def test_chitchat_agent_different_models(self):
        """Test ChitChatAgent with different model types."""
        models = [LLM.GPT_4_1, LLM.GPT_4_1_MINI, LLM.GPT_4_1_NANO, LLM.GPT_4O]
        
        for model in models:
            agent = ChitChatAgent(name=f"ChitChat-{model}", model_name=model)
            assert agent.model_name == model
            assert agent._agent.model == model
    
    def test_chitchat_agent_timeout_default(self):
        """Test that ChitChatAgent has reasonable timeout."""
        agent = ChitChatAgent(name="TestChitChat")
        
        # Should have inherited default timeout
        assert agent.timeout == 60
    
    @patch('bioagents.agents.chitchat_agent.Runner')
    @pytest.mark.asyncio
    async def test_achat_empty_response(self, mock_runner):
        """Test achat with empty response."""
        # Setup mock with empty response
        mock_run_result = MagicMock()
        mock_run_result.final_output = ""
        mock_run_result.new_items = []
        
        mock_runner.run = AsyncMock(return_value=mock_run_result)
        
        agent = ChitChatAgent(name="TestChitChat")
        
        result = await agent.achat("Test query")
        
        assert result.response_str == ""
        assert result.route == AgentRoute.CHIT_CHAT
    
    @patch('bioagents.agents.chitchat_agent.Runner')
    @pytest.mark.asyncio
    async def test_achat_long_input(self, mock_runner):
        """Test achat with long input text."""
        # Setup mock
        mock_run_result = MagicMock()
        mock_run_result.final_output = "I understand you have a lot to say!"
        mock_run_result.new_items = []
        
        mock_runner.run = AsyncMock(return_value=mock_run_result)
        
        agent = ChitChatAgent(name="TestChitChat")
        
        # Create long input
        long_input = "This is a very long input text. " * 100
        
        result = await agent.achat(long_input)
        
        assert result.response_str == "I understand you have a lot to say!"
        assert result.route == AgentRoute.CHIT_CHAT
    
    def test_chitchat_agent_instructions_brevity(self):
        """Test that instructions emphasize brevity."""
        agent = ChitChatAgent(name="TestChitChat")
        
        # Should emphasize being brief
        instructions_lower = agent.instructions.lower()
        assert "brief" in instructions_lower or "concise" in instructions_lower
    
    def test_chitchat_agent_no_questions_instruction(self):
        """Test that instructions include not asking questions."""
        agent = ChitChatAgent(name="TestChitChat")
        
        # Should avoid asking questions
        assert "AVOID asking the user any question" in agent.instructions
    
    @patch('bioagents.agents.chitchat_agent.Runner')
    @pytest.mark.asyncio
    async def test_multiple_achat_calls(self, mock_runner):
        """Test multiple sequential achat calls."""
        agent = ChitChatAgent(name="TestChitChat")
        
        responses = [
            "First response",
            "Second response", 
            "Third response"
        ]
        
        for i, expected_response in enumerate(responses):
            mock_run_result = MagicMock()
            mock_run_result.final_output = expected_response
            mock_run_result.new_items = []
            
            mock_runner.run = AsyncMock(return_value=mock_run_result)
            
            result = await agent.achat(f"Query {i+1}")
            
            assert result.response_str == expected_response
            assert result.route == AgentRoute.CHIT_CHAT 