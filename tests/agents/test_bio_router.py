"""
Tests for the router-based agent in bioagents.agents.bio_router module.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bioagents.agents.bio_router import BioRouterAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM

# Load environment variables from .env for integration tests
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
except ImportError:
    pass


class TestBioRouterAgent:
    """Test BioRouterAgent (router-based) functionality."""

    def test_bio_router_creation_default(self):
        """Test creating BioRouterAgent with default parameters."""
        agent = BioRouterAgent(name="TestConcierge")

        assert agent.name == "TestConcierge"
        assert agent.model_name == LLM.GPT_4_1_MINI
        assert "bio-reasoning agent" in agent.instructions
        asyncio.run(agent.start())
        assert agent._agent is not None
        asyncio.run(agent.stop())

    def test_bio_router_creation_custom_model(self):
        """Test creating BioRouterAgent with custom model."""
        agent = BioRouterAgent(name="CustomConcierge", model_name=LLM.GPT_4_1_MINI)

        assert agent.model_name == LLM.GPT_4_1_MINI

    def test_create_agent_structure(self):
        """Test that _create_agent creates proper agent structure."""
        agent = BioRouterAgent(name="TestConcierge")

        # Check that the agent was created
        asyncio.run(agent.start())
        assert agent._agent is not None
        assert agent._agent.name == "Bio Concierge"
        assert agent._agent.model == LLM.GPT_4_1_MINI
        assert len(agent._agent.handoffs) == 4  # Four sub-agents (excluding graph and llama agents)
        asyncio.run(agent.stop())

    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_bio_router_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Bio Concierge Agent: Test query received!"
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("Test query")
        assert isinstance(result, AgentResponse)
        assert "Bio Concierge Agent:" in result.response_str

    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_chit_chat_agent_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = (
            "Chit Chat Agent: Hello! How can I help you today?"
        )
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("Hello")
        assert isinstance(result, AgentResponse)
        assert (
            "Chit Chat Agent:" in result.response_str
            or "Bio Concierge Agent:" in result.response_str
        )

    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_web_agent_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = (
            "Web Reasoning Agent: Recent developments in genetics..."
        )
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("What is the latest news in genetics?")
        assert isinstance(result, AgentResponse)
        assert (
            "Web Reasoning Agent:" in result.response_str
            or "Bio Concierge Agent:" in result.response_str
        )

    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_bio_mcp_agent_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Bio MCP Agent: The variant rs113488022..."
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("Tell me about variant rs113488022")
        assert isinstance(result, AgentResponse)
        assert (
            "Bio MCP Agent:" in result.response_str
            or "Bio Concierge Agent:" in result.response_str
        )

    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_unknown_agent_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Bio Concierge Agent: The airspeed velocity..."
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat(
            "What is the airspeed velocity of an unladen swallow?"
        )
        assert isinstance(result, AgentResponse)
        assert "Bio Concierge Agent:" in result.response_str

    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_no_source_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = (
            "Bio Concierge Agent: Test query with no clear source"
        )
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("Test query with no clear source")
        assert isinstance(result, AgentResponse)
        assert "Bio Concierge Agent:" in result.response_str

    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_already_prefixed_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Chit Chat Agent: Already prefixed response"
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("Chit Chat Agent: Already prefixed response")
        assert isinstance(result, AgentResponse)
        assert (
            "Chit Chat Agent:" in result.response_str
            or "Bio Concierge Agent:" in result.response_str
        )

    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_with_citations(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        # Setup mock response with citations
        from bioagents.models.source import Source

        mock_run_result = MagicMock()
        mock_run_result.final_output = "Web Reasoning Agent: A recent PubMed article..."
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("Find a recent PubMed article on CRISPR.")
        assert isinstance(result, AgentResponse)
        assert (
            "Web Reasoning Agent:" in result.response_str
            or "Bio Concierge Agent:" in result.response_str
        )
        # Citations may or may not be present, but should be a list
        assert isinstance(result.citations, list)

    @pytest.mark.asyncio
    async def test_achat_lazy_initializes_agent(self):
        """achat() should lazily initialize concierge handoffs if _agent is None."""
        agent = BioRouterAgent(name="TestConcierge")
        agent._agent = None  # simulate uninitialized
        result = await agent.achat("Test query")
        assert isinstance(result, AgentResponse)

    # Keep the timeout test as a controlled (mocked) test
    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_timeout_error(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp
    ):
        import asyncio

        mock_runner_run.side_effect = asyncio.TimeoutError()
        agent = BioRouterAgent(name="TestConcierge")
        agent.timeout = 1
        with pytest.raises(asyncio.TimeoutError):
            await agent.achat("Test query")

    def test_construct_response_with_agent_info(self):
        """Test _construct_response_with_agent_info method."""
        agent = BioRouterAgent(name="TestConcierge")

        # Create mock RunResult
        mock_run_result = MagicMock()
        mock_run_result.new_items = []

        response = agent._construct_response_with_agent_info(
            mock_run_result, "Test response", AgentRouteType.WEBSEARCH
        )

        assert isinstance(response, AgentResponse)
        assert response.response_str == "Test response"
        assert response.route == AgentRouteType.WEBSEARCH
        assert response.citations == []

    def test_multiple_message_items_source_detection(self):
        """Test source detection with multiple message items."""
        agent = BioRouterAgent(name="TestConcierge")

        # Create multiple mock items, last one should be used for source
        mock_item1 = MagicMock()
        mock_item1.type = "message_output_item"
        mock_item1.raw_item.source = "Bio Concierge"

        mock_item2 = MagicMock()
        mock_item2.type = "message_output_item"
        mock_item2.raw_item.source = "Chit Chat Agent"

        mock_items = [mock_item1, mock_item2]

        # Test that it finds the last source (when reversed)
        responding_agent_name = "Bio Concierge"  # Default
        for item in reversed(mock_items):
            if (
                item.type == "message_output_item"
                and hasattr(item.raw_item, "source")
                and item.raw_item.source
            ):
                responding_agent_name = item.raw_item.source
                break

        assert responding_agent_name == "Chit Chat Agent"

    def test_response_prefixing_and_mapping(self):
        """Unit test for response prefixing and mapping logic in _construct_response_with_agent_info."""
        from bioagents.agents.bio_router import BioRouterAgent

        agent = BioRouterAgent(name="TestConcierge")

        # Simulate a RunResult with a message from Chit Chat Agent
        class RawItem:
            def __init__(self, source, content=None):
                self.source = source
                self.content = content or []

        class MockRunResult:
            def __init__(self, final_output, new_items):
                self.final_output = final_output
                self.new_items = new_items

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item = RawItem("Chit Chat Agent")
        run_result = MockRunResult("Hello! How are you today?", [mock_item])
        response = agent._construct_response_with_agent_info(
            run_result,
            "Chit Chat Agent: Hello! How are you today?",
            AgentRouteType.CHITCHAT,
        )
        assert response.response_str == "Chit Chat Agent: Hello! How are you today?"
        assert response.route == AgentRouteType.CHITCHAT

    def test_response_prefixing_web_agent(self):
        from bioagents.agents.bio_router import BioRouterAgent

        agent = BioRouterAgent(name="TestConcierge")

        class RawItem:
            def __init__(self, source, content=None):
                self.source = source
                self.content = content or []

        class MockRunResult:
            def __init__(self, final_output, new_items):
                self.final_output = final_output
                self.new_items = new_items

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item = RawItem("Web Reasoning Agent")
        run_result = MockRunResult("Web result here", [mock_item])
        response = agent._construct_response_with_agent_info(
            run_result, "Web Reasoning Agent: Web result here", AgentRouteType.WEBSEARCH
        )
        assert response.response_str == "Web Reasoning Agent: Web result here"
        assert response.route == AgentRouteType.WEBSEARCH

    def test_response_prefixing_bio_mcp_agent(self):
        from bioagents.agents.bio_router import BioRouterAgent

        agent = BioRouterAgent(name="TestConcierge")

        class RawItem:
            def __init__(self, source, content=None):
                self.source = source
                self.content = content or []

        class MockRunResult:
            def __init__(self, final_output, new_items):
                self.final_output = final_output
                self.new_items = new_items

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item = RawItem("BiomedicalAssistant")
        run_result = MockRunResult("BioMCP result", [mock_item])
        response = agent._construct_response_with_agent_info(
            run_result, "Bio MCP Agent: BioMCP result", AgentRouteType.BIOMCP
        )
        assert response.response_str == "Bio MCP Agent: BioMCP result"
        assert response.route == AgentRouteType.BIOMCP

    def test_response_prefixing_unknown_agent(self):
        from bioagents.agents.bio_router import BioRouterAgent
        from bioagents.agents.common import AgentRouteType

        agent = BioRouterAgent(name="TestConcierge")

        class RawItem:
            def __init__(self, source, content=None):
                self.source = source
                self.content = content or []

        class MockRunResult:
            def __init__(self, final_output, new_items):
                self.final_output = final_output
                self.new_items = new_items

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item = RawItem("Unknown Agent")
        run_result = MockRunResult("Some response", [mock_item])
        response = agent._construct_response_with_agent_info(
            run_result, "Bio Concierge Agent: Some response", AgentRouteType.REASONING
        )
        assert response.response_str == "Bio Concierge Agent: Some response"
        assert response.route == AgentRouteType.REASONING

    def test_response_prefixing_already_prefixed(self):
        from bioagents.agents.bio_router import BioRouterAgent

        agent = BioRouterAgent(name="TestConcierge")

        class RawItem:
            def __init__(self, source, content=None):
                self.source = source
                self.content = content or []

        class MockRunResult:
            def __init__(self, final_output, new_items):
                self.final_output = final_output
                self.new_items = new_items

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item = RawItem("Chit Chat Agent")
        run_result = MockRunResult(
            "Chit Chat Agent: Already prefixed response", [mock_item]
        )
        response = agent._construct_response_with_agent_info(
            run_result,
            "Chit Chat Agent: Already prefixed response",
            AgentRouteType.CHITCHAT,
        )
        assert response.response_str == "Chit Chat Agent: Already prefixed response"
        assert response.route == AgentRouteType.CHITCHAT

    @patch("bioagents.agents.bio_router.GraphAgent")
    @patch("bioagents.agents.bio_router.LlamaRAGAgent")
    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_graph_agent_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp, mock_llamarag, mock_graph
    ):
        """Test routing to GraphAgent for complex NCCN questions."""
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = "Graph Agent: Complex treatment relationships..."
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("How do HER2 status and hormone receptor status interact for treatment?")
        assert isinstance(result, AgentResponse)
        assert (
            "Graph Agent:" in result.response_str
            or "Bio Concierge Agent:" in result.response_str
        )

    @patch("bioagents.agents.bio_router.GraphAgent")
    @patch("bioagents.agents.bio_router.LlamaRAGAgent")
    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @patch("agents.Runner.run")
    @pytest.mark.asyncio
    async def test_achat_llamarag_agent_response(
        self, mock_runner_run, mock_chitchat, mock_web, mock_biomcp, mock_llamarag, mock_graph
    ):
        """Test routing to LlamaRAGAgent for simple NCCN questions."""
        # Setup mock response
        mock_run_result = MagicMock()
        mock_run_result.final_output = "LlamaCloud RAG Agent: NCCN recommendations are..."
        mock_run_result.new_items = []
        mock_runner_run.return_value = mock_run_result

        agent = BioRouterAgent(name="TestConcierge")
        result = await agent.achat("What are the NCCN recommendations for HER2+ breast cancer?")
        assert isinstance(result, AgentResponse)
        assert (
            "LlamaCloud RAG Agent:" in result.response_str
            or "Bio Concierge Agent:" in result.response_str
        )

    @patch("bioagents.agents.bio_router.GraphAgent")
    @patch("bioagents.agents.bio_router.LlamaRAGAgent")
    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @pytest.mark.asyncio
    async def test_graph_agent_routing_logic(
        self, mock_chitchat, mock_web, mock_biomcp, mock_llamarag, mock_graph
    ):
        """Test that GraphAgent is properly instantiated and routing works."""
        agent = BioRouterAgent(name="TestConcierge")
        await agent.start()
        
        # Verify GraphAgent was instantiated
        assert hasattr(agent, '_graph_agent')
        assert agent._graph_agent is not None
        mock_graph.assert_called_once_with(name="Graph Agent")
        
        await agent.stop()

    @patch("bioagents.agents.bio_router.GraphAgent")
    @patch("bioagents.agents.bio_router.LlamaRAGAgent")
    @patch("bioagents.agents.bio_router.BioMCPAgent")
    @patch("bioagents.agents.bio_router.WebReasoningAgent")
    @patch("bioagents.agents.bio_router.ChitChatAgent")
    @pytest.mark.asyncio
    async def test_graph_agent_delegation(
        self, mock_chitchat, mock_web, mock_biomcp, mock_llamarag, mock_graph
    ):
        """Test that queries are properly delegated to GraphAgent."""
        # Setup mock graph agent
        mock_graph_instance = AsyncMock()
        mock_graph_response = AgentResponse(
            response_str="[Graph] Complex treatment analysis...",
            route=AgentRouteType.GRAPH
        )
        mock_graph_instance.achat.return_value = mock_graph_response
        mock_graph.return_value = mock_graph_instance

        agent = BioRouterAgent(name="TestConcierge")
        
        # Mock the router to return "graph"
        with patch("agents.Runner.run") as mock_runner:
            mock_run_result = MagicMock()
            mock_run_result.final_output = "graph"
            mock_runner.return_value = mock_run_result
            
            result = await agent.achat("Complex NCCN question")
            
            # Verify GraphAgent was called
            mock_graph_instance.achat.assert_called_once_with("Complex NCCN question")
            assert result.route == AgentRouteType.GRAPH
            assert "[Graph]" in result.response_str

    def test_response_prefixing_graph_agent(self):
        """Test response prefixing for GraphAgent."""
        from bioagents.agents.bio_router import BioRouterAgent

        agent = BioRouterAgent(name="TestConcierge")

        class RawItem:
            def __init__(self, source, content=None):
                self.source = source
                self.content = content or []

        class MockRunResult:
            def __init__(self, final_output, new_items):
                self.final_output = final_output
                self.new_items = new_items

        mock_item = MagicMock()
        mock_item.type = "message_output_item"
        mock_item.raw_item = RawItem("Graph Agent")
        run_result = MockRunResult("Graph analysis result", [mock_item])
        response = agent._construct_response_with_agent_info(
            run_result, "Graph Agent: Graph analysis result", AgentRouteType.GRAPH
        )
        assert response.response_str == "Graph Agent: Graph analysis result"
        assert response.route == AgentRouteType.GRAPH

    def test_router_tools_include_graph(self):
        """Test that router tools include the graph routing tool."""
        agent = BioRouterAgent(name="TestConcierge")
        asyncio.run(agent.start())
        
        # Check that the router agent has the expected number of tools
        assert len(agent._router_agent.tools) == 5  # graph, llamarag, biomcp, websearch, chitchat
        
        # Check that tool descriptions include graph-related routing
        tool_descriptions = [tool.description for tool in agent._router_agent.tools if hasattr(tool, 'description')]
        assert any("graph" in desc.lower() for desc in tool_descriptions)
        
        asyncio.run(agent.stop())
