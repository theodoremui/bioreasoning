"""
Comprehensive tests for bioagents.agents.graph_agent module.

This test suite covers all functionality of the GraphAgent class including:
- Initialization with dependency injection
- Component lifecycle management
- Graph query tool functionality
- Error handling and edge cases
- Configuration management
- Integration with GraphRAG components

Author: Theodore Mui
Date: 2025-08-16
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

from bioagents.agents.graph_agent import GraphAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.graph import GraphConfig, GraphRAGStore, GraphRAGQueryEngine
from bioagents.models.llms import LLM
from bioagents.models.source import Source


class TestGraphAgent:
    """Test suite for GraphAgent class."""

    def test_graph_agent_creation_default(self):
        """Test creating GraphAgent with default parameters."""
        agent = GraphAgent(name="TestGraphAgent", auto_initialize=False)

        assert agent.name == "TestGraphAgent"
        assert agent.model_name == LLM.GPT_4_1_NANO
        assert agent.instructions.startswith("You are an expert that can answer questions")
        assert agent.handoff_description.startswith("You are an expert that can answer questions")
        assert agent.pdf_path == "data/nccn_breast_cancer.pdf"
        assert agent.auto_initialize is False
        assert not agent._started

    def test_graph_agent_creation_custom(self):
        """Test creating GraphAgent with custom parameters."""
        config = GraphConfig.from_environment()
        config.query.similarity_top_k = 15

        agent = GraphAgent(
            name="CustomGraphAgent",
            model_name=LLM.GPT_4_1_MINI,
            config=config,
            pdf_path="custom/path.pdf",
            auto_initialize=False,
        )

        assert agent.name == "CustomGraphAgent"
        assert agent.model_name == LLM.GPT_4_1_MINI
        assert agent.config.query.similarity_top_k == 15
        assert agent.pdf_path == "custom/path.pdf"

    def test_graph_agent_dependency_injection(self):
        """Test GraphAgent with dependency injection."""
        # Create mock dependencies
        mock_config = MagicMock(spec=GraphConfig)
        mock_store = MagicMock(spec=GraphRAGStore)
        mock_index = MagicMock()
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)

        agent = GraphAgent(
            name="DIAgent",
            config=mock_config,
            graph_store=mock_store,
            index=mock_index,
            query_engine=mock_engine,
            auto_initialize=False,
        )

        assert agent.config is mock_config
        assert agent._graph_store is mock_store
        assert agent._index is mock_index
        assert agent._query_engine is mock_engine

    def test_graph_agent_properties(self):
        """Test GraphAgent property accessors."""
        config = GraphConfig.from_environment()
        mock_store = MagicMock()
        mock_index = MagicMock()
        mock_engine = MagicMock()

        agent = GraphAgent(
            name="PropAgent",
            config=config,
            graph_store=mock_store,
            index=mock_index,
            query_engine=mock_engine,
            auto_initialize=False,
        )

        assert agent.config is config
        assert agent.graph_store is mock_store
        assert agent.index is mock_index
        assert agent.query_engine is mock_engine

        # Test config setter
        new_config = GraphConfig.from_environment()
        new_config.query.similarity_top_k = 25
        agent.config = new_config
        assert agent.config.query.similarity_top_k == 25

    @patch('bioagents.agents.graph_agent.OpenAI')
    @patch('bioagents.agents.graph_agent.GraphRAGStore')
    @patch('bioagents.agents.graph_agent.GraphRAGQueryEngine')
    @patch('bioagents.agents.graph_agent.PropertyGraphIndex')
    @patch('bioagents.agents.graph_agent.Spinner')
    @pytest.mark.asyncio
    async def test_initialize_components_success(
        self, mock_spinner, mock_index_class, mock_engine_class, mock_store_class, mock_openai
    ):
        """Test successful component initialization."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm

        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        mock_index = MagicMock()
        mock_index_class.return_value = mock_index

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        # Mock spinner context manager
        mock_spinner_instance = MagicMock()
        mock_spinner_instance.__enter__ = MagicMock(return_value=mock_spinner_instance)
        mock_spinner_instance.__exit__ = MagicMock(return_value=None)
        mock_spinner.return_value = mock_spinner_instance

        agent = GraphAgent(name="TestAgent", auto_initialize=False)

        # Mock the _load_or_create_index method
        with patch.object(agent, '_load_or_create_index', return_value=mock_index):
            await agent._initialize_components()

        # Components should be initialized
        assert agent._llm is not None
        assert agent._graph_store is not None
        assert agent._index is not None
        assert agent._query_engine is not None
        assert agent._llm is mock_llm
        assert agent._graph_store is mock_store
        assert agent._index is mock_index
        assert agent._query_engine is mock_engine

    @pytest.mark.asyncio
    async def test_initialize_components_idempotent(self):
        """Test that _initialize_components can be called multiple times safely."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        
        # Mock the components to avoid actual initialization
        with patch('bioagents.agents.graph_agent.OpenAI') as mock_openai, \
             patch('bioagents.agents.graph_agent.GraphRAGStore') as mock_store, \
             patch.object(agent, '_load_or_create_index') as mock_index, \
             patch('bioagents.agents.graph_agent.GraphRAGQueryEngine') as mock_engine:
            
            mock_index.return_value = MagicMock()
            
            # Call twice - should not cause issues
            await agent._initialize_components()
            await agent._initialize_components()
            
            # Components have their own guards (if not self._llm), so they won't be re-created
            # But the method should complete successfully both times
            assert mock_openai.call_count == 1  # Only called once due to internal guards
            assert agent._llm is not None

    @patch('bioagents.agents.graph_agent.OpenAI')
    @pytest.mark.asyncio
    async def test_initialize_components_failure(self, mock_openai):
        """Test component initialization failure handling."""
        mock_openai.side_effect = Exception("OpenAI initialization failed")

        agent = GraphAgent(name="TestAgent", auto_initialize=False)

        with pytest.raises(Exception, match="OpenAI initialization failed"):
            await agent._initialize_components()

        # Components should not be initialized after failure
        assert agent._llm is None

    @patch('os.path.exists')
    @patch('bioagents.agents.graph_agent.PropertyGraphIndex')
    @patch('bioagents.agents.graph_agent.Spinner')
    @pytest.mark.asyncio
    async def test_load_or_create_index_existing(self, mock_spinner, mock_index_class, mock_exists):
        """Test loading existing index and communities."""
        mock_exists.return_value = True
        mock_index = MagicMock()
        mock_index_class.return_value = mock_index

        mock_store = MagicMock()
        mock_store.ensure_communities = MagicMock()

        # Mock spinner
        mock_spinner_instance = MagicMock()
        mock_spinner_instance.__enter__ = MagicMock()
        mock_spinner_instance.__exit__ = MagicMock()
        mock_spinner.return_value = mock_spinner_instance

        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._graph_store = mock_store

        result = await agent._load_or_create_index()

        assert result is mock_index
        mock_store.ensure_communities.assert_called_once()

    @patch('os.path.exists')
    @pytest.mark.asyncio
    async def test_load_or_create_index_no_existing(self, mock_exists):
        """Test creating new index when no existing communities."""
        mock_exists.return_value = False

        agent = GraphAgent(name="TestAgent", auto_initialize=False)

        # Mock the _process_pdf_and_build_index method
        mock_index = MagicMock()
        with patch.object(agent, '_process_pdf_and_build_index', return_value=mock_index):
            result = await agent._load_or_create_index()

        assert result is mock_index

    @patch('bioagents.agents.graph_agent.DocumentProcessor')
    @patch('bioagents.agents.graph_agent.GraphRAGExtractor')
    @patch('bioagents.agents.graph_agent.PropertyGraphIndex')
    @patch('bioagents.agents.graph_agent.Spinner')
    @pytest.mark.asyncio
    async def test_process_pdf_and_build_index_success(
        self, mock_spinner, mock_index_class, mock_extractor_class, mock_processor_class
    ):
        """Test successful PDF processing and index building."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_processor.process_document = AsyncMock(return_value=["node1", "node2"])
        mock_processor_class.return_value = mock_processor

        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor

        mock_index = MagicMock()
        mock_index_class.return_value = mock_index

        mock_store = MagicMock()
        mock_store.ensure_communities = MagicMock()

        # Mock spinner
        mock_spinner_instance = MagicMock()
        mock_spinner_instance.__enter__ = MagicMock()
        mock_spinner_instance.__exit__ = MagicMock()
        mock_spinner.return_value = mock_spinner_instance

        # Setup agent with required components
        config = GraphConfig.from_environment()
        config.api.llamacloud_api_key = "test-key"
        
        agent = GraphAgent(name="TestAgent", config=config, auto_initialize=False)
        agent._graph_store = mock_store
        agent._llm = MagicMock()

        result = await agent._process_pdf_and_build_index()

        assert result is mock_index
        mock_processor.process_document.assert_called_once_with(agent.pdf_path)
        mock_store.ensure_communities.assert_called_once()

    @patch('bioagents.agents.graph_agent.DocumentProcessor')
    @pytest.mark.asyncio
    async def test_process_pdf_and_build_index_no_api_key(self, mock_processor_class):
        """Test PDF processing failure when no API key."""
        config = GraphConfig.from_environment()
        config.api.llamacloud_api_key = None

        agent = GraphAgent(name="TestAgent", config=config, auto_initialize=False)

        with pytest.raises(ValueError, match="LLAMACLOUD_API_KEY is required"):
            await agent._process_pdf_and_build_index()

    @patch('bioagents.agents.graph_agent.DocumentProcessor')
    @patch('bioagents.agents.graph_agent.Spinner')
    @pytest.mark.asyncio
    async def test_process_pdf_and_build_index_no_nodes(self, mock_spinner, mock_processor_class):
        """Test PDF processing failure when no nodes extracted."""
        mock_processor = MagicMock()
        mock_processor.process_document = AsyncMock(return_value=[])
        mock_processor_class.return_value = mock_processor

        # Mock spinner
        mock_spinner_instance = MagicMock()
        mock_spinner_instance.__enter__ = MagicMock()
        mock_spinner_instance.__exit__ = MagicMock()
        mock_spinner.return_value = mock_spinner_instance

        config = GraphConfig.from_environment()
        config.api.llamacloud_api_key = "test-key"

        agent = GraphAgent(name="TestAgent", config=config, auto_initialize=False)

        with pytest.raises(ValueError, match="Failed to process PDF or no content extracted"):
            await agent._process_pdf_and_build_index()

    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test successful agent start."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)

        # Mock all initialization methods
        with patch.object(agent, '_initialize_components') as mock_init, \
             patch.object(agent, '_create_agent') as mock_create:
            
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            await agent.start()

            assert agent._started
            assert agent._agent is mock_agent
            mock_init.assert_not_called()  # auto_initialize=False
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_with_auto_initialize(self):
        """Test agent start with auto initialization."""
        agent = GraphAgent(name="TestAgent", auto_initialize=True)

        with patch.object(agent, '_initialize_components') as mock_init, \
             patch.object(agent, '_create_agent') as mock_create:
            
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            await agent.start()

            assert agent._started
            mock_init.assert_called_once()
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_already_started(self):
        """Test that start skips if already started."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._started = True

        with patch.object(agent, '_initialize_components') as mock_init:
            await agent.start()
            mock_init.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_failure(self):
        """Test agent start failure handling."""
        agent = GraphAgent(name="TestAgent", auto_initialize=True)

        with patch.object(agent, '_initialize_components', side_effect=Exception("Init failed")):
            with pytest.raises(Exception, match="Init failed"):
                await agent.start()

            assert not agent._started

    @pytest.mark.asyncio
    async def test_aclose_success(self):
        """Test successful agent close."""
        mock_store = MagicMock()
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._graph_store = mock_store
        agent._started = True
        agent._initialized = True

        await agent.aclose()

        assert not agent._started

    @pytest.mark.asyncio
    async def test_stop_calls_aclose(self):
        """Test that stop calls aclose."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)

        with patch.object(agent, 'aclose') as mock_aclose:
            await agent.stop()
            mock_aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)

        with patch.object(agent, 'start') as mock_start, \
             patch.object(agent, 'stop') as mock_stop:

            async with agent as ctx_agent:
                assert ctx_agent is agent
                mock_start.assert_called_once()

            mock_stop.assert_called_once()

    def test_create_graph_query_tool(self):
        """Test creation of graph query tool."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Test response from graph"
        mock_engine.get_last_citations.return_value = [
            {
                "title": "Test Citation",
                "snippet": "Test snippet",
                "file_name": "test.pdf",
                "start_page": "1",
                "end_page": "2",
                "score": 0.95,
                "text": "Full citation text"
            }
        ]

        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph

        # Test the tool function using proper JSON format
        import json
        result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": "What is breast cancer treatment?"})))

        assert isinstance(result, AgentResponse)
        assert result.response_str == "[Graph] Test response from graph"
        assert result.route == AgentRouteType.GRAPH
        assert len(result.citations) == 1
        assert result.citations[0].title == "Test Citation"
        assert result.citations[0].source == "knowledge_graph"

    def test_create_graph_query_tool_error_handling(self):
        """Test graph query tool error handling."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.side_effect = Exception("Graph query failed")

        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph

        import json
        result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": "Test query"})))

        assert isinstance(result, AgentResponse)
        assert "Sorry, I encountered an error querying the knowledge graph" in result.response_str
        assert result.route == AgentRouteType.GRAPH

    def test_create_graph_query_tool_citation_error(self):
        """Test graph query tool with citation extraction error."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Test response"
        mock_engine.get_last_citations.side_effect = Exception("Citation error")

        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph

        import json
        result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": "Test query"})))

        assert isinstance(result, AgentResponse)
        assert result.response_str == "[Graph] Test response"
        assert len(result.citations) == 0  # Should handle citation error gracefully

    def test_clean_response_without_citations(self):
        """Test that query engine now returns clean response without citations section."""
        # Mock query engine that returns clean response (new behavior)
        mock_engine = MagicMock()
        mock_engine.query.return_value = (
            "HER2 status is important for treatment decisions [1,2].\n\n"
            "This affects staging and therapeutic approaches [3,4]."
        )
        mock_engine.get_last_citations.return_value = [
            {
                "title": "NCCN Guidelines",
                "snippet": "HER2 testing guidelines",
                "file_name": "nccn_breast_cancer.pdf",
                "start_page": "118",
                "end_page": "118",
                "score": 0.9,
                "text": "HER2 testing guidelines"
            }
        ]
        
        # Set the mock engine
        GraphAgent.set_query_engine(mock_engine)
        tool = GraphAgent.query_knowledge_graph
        
        # Test the tool
        import json
        result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": "How does HER2 status affect treatment?"})))
        
        # Verify that response is clean (no citations section)
        assert "Citations:" not in result.response_str
        assert result.response_str == "[Graph] HER2 status is important for treatment decisions [1,2].\n\nThis affects staging and therapeutic approaches [3,4]."
        
        # Verify that structured citations are still present
        assert len(result.citations) == 1
        assert result.citations[0].title == "NCCN Guidelines"
        assert result.citations[0].snippet == "HER2 testing guidelines"

    def test_create_agent_success(self):
        """Test successful agent creation."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._query_engine = mock_engine

        with patch('bioagents.agents.graph_agent.Agent') as mock_agent_class:
            mock_agent_instance = MagicMock()
            mock_agent_class.return_value = mock_agent_instance

            result = agent._create_agent()

            assert result is mock_agent_instance
            mock_agent_class.assert_called_once()

            # Verify agent was created with correct parameters
            call_args = mock_agent_class.call_args
            assert call_args[1]['name'] == "TestAgent"
            assert call_args[1]['model'] == LLM.GPT_4_1_NANO
            assert len(call_args[1]['tools']) == 1
            assert call_args[1]['tool_use_behavior'] == "stop_on_first_tool"

    def test_create_agent_no_query_engine(self):
        """Test agent creation failure when no query engine."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._query_engine = None
        # Clear the class-level query engine as well
        GraphAgent._query_engine = None

        with pytest.raises(ValueError, match="Query engine must be initialized"):
            agent._create_agent()

    @patch('bioagents.agents.base_agent.BaseAgent.achat')
    @pytest.mark.asyncio
    async def test_achat_success(self, mock_base_achat):
        """Test successful achat call."""
        mock_response = AgentResponse(
            response_str="Test graph response",
            route=AgentRouteType.REASONING
        )
        mock_base_achat.return_value = mock_response

        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._started = True
        agent._agent = MagicMock()

        result = await agent.achat("Test query")

        assert isinstance(result, AgentResponse)
        assert result.response_str == "Test graph response"
        assert result.route == AgentRouteType.GRAPH  # Should be overridden
        mock_base_achat.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    async def test_achat_auto_start(self):
        """Test achat automatically starts agent if not started."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._started = False

        with patch.object(agent, 'start') as mock_start, \
             patch('bioagents.agents.base_agent.BaseAgent.achat') as mock_base_achat:
            
            mock_response = AgentResponse(response_str="Test", route=AgentRouteType.REASONING)
            mock_base_achat.return_value = mock_response
            agent._agent = MagicMock()  # Set after start

            result = await agent.achat("Test query")

            mock_start.assert_called_once()
            assert result.route == AgentRouteType.GRAPH

    @pytest.mark.asyncio
    async def test_achat_no_agent_error(self):
        """Test achat error when agent not initialized."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._started = True
        agent._agent = None

        result = await agent.achat("Test query")

        assert isinstance(result, AgentResponse)
        assert "Sorry, I encountered an error" in result.response_str
        assert result.route == AgentRouteType.GRAPH

    @pytest.mark.asyncio
    async def test_achat_general_error(self):
        """Test achat general error handling."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        agent._started = False  # Force it to call start

        with patch.object(agent, 'start', side_effect=Exception("Start failed")):
            result = await agent.achat("Test query")

            assert isinstance(result, AgentResponse)
            assert "Sorry, I encountered an error: Start failed" in result.response_str
            assert result.route == AgentRouteType.GRAPH


class TestGraphAgentIntegration:
    """Integration tests for GraphAgent with real components."""

    @pytest.mark.asyncio
    async def test_full_integration_mock(self):
        """Test full integration with mocked components."""
        # Create a realistic mock setup
        config = GraphConfig.from_environment()
        config.api.llamacloud_api_key = "test-key"

        mock_store = MagicMock(spec=GraphRAGStore)
        mock_index = MagicMock()
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        
        # Setup mock engine responses
        mock_engine.query.return_value = "HER2-positive breast cancer should be treated with targeted therapy."
        mock_engine.get_last_citations.return_value = [
            {
                "title": "NCCN Breast Cancer Guidelines",
                "snippet": "HER2-positive tumors require targeted therapy",
                "file_name": "nccn_breast_cancer.pdf",
                "start_page": "15",
                "end_page": "16",
                "score": 0.92,
                "text": "For HER2-positive breast cancer, targeted therapy with trastuzumab is recommended."
            }
        ]

        agent = GraphAgent(
            name="Integration Test Agent",
            config=config,
            graph_store=mock_store,
            index=mock_index,
            query_engine=mock_engine,
            auto_initialize=False
        )

        async with agent:
            response = await agent.achat("How to treat HER2-positive breast cancer?")

            assert isinstance(response, AgentResponse)
            assert "HER2-positive breast cancer should be treated" in response.response_str
            assert response.route == AgentRouteType.GRAPH
            assert len(response.citations) == 1
            assert response.citations[0].title == "NCCN Breast Cancer Guidelines"
            assert response.citations[0].source == "knowledge_graph"

    @pytest.mark.asyncio
    async def test_configuration_variations(self):
        """Test agent with different configuration variations."""
        configs = [
            {"similarity_top_k": 10, "max_summaries_to_use": 3},
            {"similarity_top_k": 25, "max_summaries_to_use": 8},
            {"similarity_top_k": 5, "max_summaries_to_use": 2},
        ]

        for config_params in configs:
            config = GraphConfig.from_environment()
            config.query.similarity_top_k = config_params["similarity_top_k"]
            config.query.max_summaries_to_use = config_params["max_summaries_to_use"]

            agent = GraphAgent(
                name=f"Config Test {config_params['similarity_top_k']}",
                config=config,
                auto_initialize=False
            )

            assert agent.config.query.similarity_top_k == config_params["similarity_top_k"]
            assert agent.config.query.max_summaries_to_use == config_params["max_summaries_to_use"]

    def test_error_scenarios(self):
        """Test various error scenarios."""
        # Test with invalid configuration
        config = GraphConfig.from_environment()
        config.query.similarity_top_k = -1  # Invalid value

        with pytest.raises(ValueError):
            config.validate()

        # Test with missing required components
        agent = GraphAgent(name="Error Test", auto_initialize=False)
        # Clear the class-level query engine
        GraphAgent._query_engine = None
        
        with pytest.raises(ValueError, match="Query engine must be initialized"):
            agent._create_agent()


class TestGraphAgentEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query_handling(self):
        """Test handling of empty or whitespace queries."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Please provide a specific question."
        mock_engine.get_last_citations.return_value = []

        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph

        # Test empty string
        import json
        result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": ""})))
        assert isinstance(result, AgentResponse)

        # Test whitespace only
        result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": "   "})))
        assert isinstance(result, AgentResponse)

    def test_very_long_query_handling(self):
        """Test handling of very long queries."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Query processed successfully."
        mock_engine.get_last_citations.return_value = []

        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph

        # Test very long query
        long_query = "What is breast cancer treatment? " * 100
        import json
        result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": long_query})))
        
        assert isinstance(result, AgentResponse)
        mock_engine.query.assert_called_once_with(long_query)

    def test_special_characters_in_query(self):
        """Test handling of queries with special characters."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Special characters handled."
        mock_engine.get_last_citations.return_value = []

        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph

        special_queries = [
            "What about HER2+ & ER- tumors?",
            "Treatment for stage III/IV cancer?",
            "Cost: $10,000-$50,000 range?",
            "Side effects (nausea, fatigue, etc.)?",
        ]

        for query in special_queries:
            import json
            result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": query})))
            assert isinstance(result, AgentResponse)

    def test_unicode_query_handling(self):
        """Test handling of Unicode characters in queries."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Unicode handled correctly."
        mock_engine.get_last_citations.return_value = []

        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph

        unicode_queries = [
            "Traitement du cancer du sein?",  # French
            "乳腺癌治疗方法？",  # Chinese
            "Лечение рака молочной железы?",  # Russian
        ]

        for query in unicode_queries:
            import json
            result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": query})))
            assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test handling of concurrent queries."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Concurrent query response"
        mock_engine.get_last_citations.return_value = []

        agent = GraphAgent(
            name="Concurrent Test",
            query_engine=mock_engine,
            auto_initialize=False
        )
        agent._started = True
        agent._agent = MagicMock()

        with patch('bioagents.agents.base_agent.BaseAgent.achat') as mock_base_achat:
            mock_base_achat.return_value = AgentResponse(
                response_str="Concurrent response",
                route=AgentRouteType.REASONING
            )

            # Run multiple queries concurrently
            queries = [
                "Query 1",
                "Query 2", 
                "Query 3"
            ]

            tasks = [agent.achat(query) for query in queries]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            for result in results:
                assert isinstance(result, AgentResponse)
                assert result.route == AgentRouteType.GRAPH

    def test_malformed_citations_handling(self):
        """Test handling of malformed citation data."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Response with malformed citations"
        
        # Test various malformed citation scenarios
        malformed_citations = [
            None,  # None citation
            {},  # Empty citation
            {"title": None},  # Missing fields
            {"title": "Test", "snippet": None, "score": "invalid"},  # Invalid types
            {"title": "Test", "extra_field": "value"},  # Extra fields
        ]
        
        mock_engine.get_last_citations.return_value = malformed_citations

        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph
        import json
        result = asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": "Test query"})))

        assert isinstance(result, AgentResponse)
        # Should handle malformed citations gracefully
        # Only valid citations should be included (None is skipped, invalid score is rejected by Pydantic)
        # Expected: {}, {"title": None}, {"title": "Test", "extra_field": "value"}
        assert len(result.citations) == 3
        
        # Verify the citations that were successfully processed
        assert result.citations[0].title == "NCCN Guidelines"  # Empty dict with defaults
        assert result.citations[1].title is None  # Title None is allowed
        assert result.citations[2].title == "Test"  # Extra fields are ignored

    @pytest.mark.asyncio
    async def test_dependency_injection_comprehensive(self):
        """Test comprehensive dependency injection capabilities."""
        # Test with custom configuration
        config = GraphConfig.from_environment()
        config.query.similarity_top_k = 10  # Custom setting
        config.query.max_summaries_to_use = 3
        
        # Create mock dependencies
        mock_store = MagicMock(spec=GraphRAGStore)
        mock_index = MagicMock()
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Dependency injection test response"
        mock_engine.get_last_citations.return_value = []
        
        agent = GraphAgent(
            name="Dependency Injection Test Agent",
            config=config,
            graph_store=mock_store,
            index=mock_index,
            query_engine=mock_engine,
            auto_initialize=False
        )
        
        # Verify configuration was injected
        assert agent.config.query.similarity_top_k == 10
        assert agent.config.query.max_summaries_to_use == 3
        
        # Verify dependencies were injected
        assert agent.graph_store is mock_store
        assert agent.index is mock_index
        assert agent.query_engine is mock_engine
        
        # Test that agent can be started and used
        async with agent:
            with patch('bioagents.agents.base_agent.BaseAgent.achat') as mock_base_achat:
                mock_base_achat.return_value = AgentResponse(
                    response_str="DI test response",
                    route=AgentRouteType.REASONING
                )
                
                response = await agent.achat("Test dependency injection")
                assert isinstance(response, AgentResponse)
                assert response.route == AgentRouteType.GRAPH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
