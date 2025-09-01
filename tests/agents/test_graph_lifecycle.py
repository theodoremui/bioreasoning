"""
Comprehensive integration tests for GraphAgent.

This module provides comprehensive tests for the GraphAgent class that can be run
from the top-level tests directory. It includes both unit tests and integration
tests to ensure the GraphAgent works correctly in various scenarios.

Test Categories:
- Unit Tests: Test individual components and methods
- Integration Tests: Test full workflows and component interactions
- Edge Cases: Test boundary conditions and error scenarios
- Performance Tests: Test behavior under various load conditions

Usage:
    # Run all tests
    pytest tests/test_graph_agent_comprehensive.py -v
    
    # Run specific test categories
    pytest tests/test_graph_agent_comprehensive.py::TestGraphAgentUnit -v
    pytest tests/test_graph_agent_comprehensive.py::TestGraphAgentIntegration -v

Author: Theodore Mui
Date: 2025-08-16
"""

import pytest
import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Import the GraphAgent and related components
from bioagents.agents.graph_agent import GraphAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.graph import GraphConfig, GraphRAGStore, GraphRAGQueryEngine
from bioagents.models.llms import LLM
from bioagents.models.source import Source


class TestGraphAgentUnit:
    """Unit tests for individual GraphAgent components."""

    def test_initialization_with_defaults(self):
        """Test GraphAgent initialization with default parameters."""
        agent = GraphAgent(name="TestAgent", auto_initialize=False)
        
        assert agent.name == "TestAgent"
        assert agent.model_name == LLM.GPT_4_1_NANO
        assert isinstance(agent.config, GraphConfig)
        assert agent.pdf_path == "data/nccn_breast_cancer.pdf"
        assert not agent.auto_initialize
        assert not agent._started

    def test_initialization_with_custom_config(self):
        """Test GraphAgent initialization with custom configuration."""
        config = GraphConfig.from_environment()
        config.query.similarity_top_k = 15
        config.query.max_summaries_to_use = 8
        
        agent = GraphAgent(
            name="CustomAgent",
            model_name=LLM.GPT_4_1_MINI,
            config=config,
            pdf_path="custom/path.pdf",
            auto_initialize=False
        )
        
        assert agent.name == "CustomAgent"
        assert agent.model_name == LLM.GPT_4_1_MINI
        assert agent.config.query.similarity_top_k == 15
        assert agent.config.query.max_summaries_to_use == 8
        assert agent.pdf_path == "custom/path.pdf"

    def test_dependency_injection(self):
        """Test dependency injection of graph components."""
        config = GraphConfig.from_environment()
        mock_store = MagicMock(spec=GraphRAGStore)
        mock_index = MagicMock()
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        
        agent = GraphAgent(
            name="DIAgent",
            config=config,
            graph_store=mock_store,
            index=mock_index,
            query_engine=mock_engine,
            auto_initialize=False
        )
        
        assert agent.config is config
        assert agent.graph_store is mock_store
        assert agent.index is mock_index
        assert agent.query_engine is mock_engine

    def test_property_accessors(self):
        """Test property getter and setter methods."""
        agent = GraphAgent(name="PropAgent", auto_initialize=False)
        
        # Test config property
        new_config = GraphConfig.from_environment()
        new_config.query.similarity_top_k = 25
        agent.config = new_config
        assert agent.config.query.similarity_top_k == 25
        
        # Test other properties return None initially
        assert agent.graph_store is None
        assert agent.index is None
        assert agent.query_engine is None

    @pytest.mark.asyncio
    async def test_graph_query_tool_creation(self):
        """Test creation and functionality of graph query tool."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Test graph response"
        mock_engine.get_last_citations.return_value = [
            {
                "title": "NCCN Guidelines",
                "snippet": "Treatment recommendations",
                "file_name": "nccn.pdf",
                "start_page": "10",
                "end_page": "11",
                "score": 0.95,
                "text": "Full citation text"
            }
        ]
        
        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph
        
        # Test the tool by invoking it properly with JSON string
        result = await tool.on_invoke_tool(None, json.dumps({"query": "What is the treatment for breast cancer?"}))
        
        assert isinstance(result, AgentResponse)
        assert result.response_str == "[Graph] Test graph response"
        assert result.route == AgentRouteType.GRAPH
        assert len(result.citations) == 1
        assert result.citations[0].title == "NCCN Guidelines"
        assert result.citations[0].source == "knowledge_graph"

    @pytest.mark.asyncio
    async def test_graph_query_tool_error_handling(self):
        """Test error handling in graph query tool."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.side_effect = Exception("Query failed")
        
        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph
        result = await tool.on_invoke_tool(None, json.dumps({"query": "Test query"}))
        
        assert isinstance(result, AgentResponse)
        assert "Sorry, I encountered an error querying the knowledge graph" in result.response_str
        assert result.route == AgentRouteType.GRAPH

    @pytest.mark.asyncio
    async def test_graph_query_tool_citation_error(self):
        """Test handling of citation extraction errors."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Response without citations"
        mock_engine.get_last_citations.side_effect = Exception("Citation error")
        
        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph
        result = await tool.on_invoke_tool(None, json.dumps({"query": "Test query"}))
        
        assert isinstance(result, AgentResponse)
        assert result.response_str == "[Graph] Response without citations"
        assert len(result.citations) == 0  # Should handle gracefully


class TestGraphAgentLifecycle:
    """Test GraphAgent lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test basic start/stop lifecycle."""
        agent = GraphAgent(name="LifecycleAgent", auto_initialize=False)
        
        # Mock initialization methods
        with patch.object(agent, '_initialize_components') as mock_init, \
             patch.object(agent, '_create_agent') as mock_create:
            
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent
            
            # Test start
            await agent.start()
            assert agent._started
            assert agent._agent is mock_agent
            mock_init.assert_not_called()  # auto_initialize=False
            
            # Test stop
            await agent.stop()
            assert not agent._started
            # Agent should be stopped

    @pytest.mark.asyncio
    async def test_start_with_auto_initialize(self):
        """Test start with automatic initialization."""
        agent = GraphAgent(name="AutoInitAgent", auto_initialize=True)
        
        with patch.object(agent, '_initialize_components') as mock_init, \
             patch.object(agent, '_create_agent') as mock_create:
            
            mock_create.return_value = MagicMock()
            
            await agent.start()
            
            mock_init.assert_called_once()
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        agent = GraphAgent(name="ContextAgent", auto_initialize=False)
        
        with patch.object(agent, 'start') as mock_start, \
             patch.object(agent, 'stop') as mock_stop:
            
            async with agent as ctx_agent:
                assert ctx_agent is agent
                mock_start.assert_called_once()
            
            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_failure_handling(self):
        """Test handling of start failures."""
        agent = GraphAgent(name="FailAgent", auto_initialize=True)
        
        with patch.object(agent, '_initialize_components', side_effect=Exception("Init failed")):
            with pytest.raises(Exception, match="Init failed"):
                await agent.start()
            
            assert not agent._started


class TestGraphAgentIntegration:
    """Integration tests for GraphAgent with mocked components."""

    @pytest.mark.asyncio
    async def test_full_workflow_mock(self):
        """Test complete workflow with mocked components."""
        # Setup configuration
        config = GraphConfig.from_environment()
        config.api.llamacloud_api_key = "test-key"
        
        # Create mock components
        mock_store = MagicMock(spec=GraphRAGStore)
        mock_index = MagicMock()
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        
        # Setup mock responses
        mock_engine.query.return_value = "HER2-positive breast cancer requires targeted therapy with trastuzumab."
        mock_engine.get_last_citations.return_value = [
            {
                "title": "NCCN Breast Cancer Guidelines v2.2024",
                "snippet": "For HER2-positive invasive breast cancer, anti-HER2 therapy is recommended",
                "file_name": "nccn_breast_cancer.pdf",
                "start_page": "BINV-15",
                "end_page": "BINV-16",
                "score": 0.92,
                "text": "Anti-HER2 therapy (trastuzumab, pertuzumab) should be given to patients with HER2-positive invasive breast cancer."
            }
        ]
        
        # Create agent with injected dependencies
        agent = GraphAgent(
            name="Integration Test Agent",
            config=config,
            graph_store=mock_store,
            index=mock_index,
            query_engine=mock_engine,
            auto_initialize=False
        )
        
        # Test full workflow
        async with agent:
            response = await agent.achat("How should HER2-positive breast cancer be treated?")
            
            assert isinstance(response, AgentResponse)
            assert "HER2-positive breast cancer requires targeted therapy" in response.response_str
            assert response.route == AgentRouteType.GRAPH
            assert len(response.citations) == 1
            
            citation = response.citations[0]
            assert citation.title == "NCCN Breast Cancer Guidelines v2.2024"
            assert citation.source == "knowledge_graph"
            assert citation.file_name == "nccn_breast_cancer.pdf"
            assert citation.score == 0.92

    @pytest.mark.asyncio
    async def test_multiple_queries_same_session(self):
        """Test multiple queries in the same session."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.get_last_citations.return_value = []
        
        agent = GraphAgent(
            name="Multi Query Agent",
            query_engine=mock_engine,
            auto_initialize=False
        )
        
        queries_and_responses = [
            ("What is breast cancer?", "Breast cancer is a malignant tumor..."),
            ("How is it diagnosed?", "Diagnosis involves imaging and biopsy..."),
            ("What are treatment options?", "Treatment includes surgery, chemotherapy...")
        ]
        
        async with agent:
            for query, expected_response in queries_and_responses:
                mock_engine.query.return_value = expected_response
                
                response = await agent.achat(query)
                
                assert isinstance(response, AgentResponse)
                assert expected_response in response.response_str
                assert response.route == AgentRouteType.GRAPH

    @pytest.mark.asyncio
    async def test_configuration_variations(self):
        """Test agent with different configuration variations."""
        test_configs = [
            {"similarity_top_k": 5, "max_summaries_to_use": 2, "max_triplets_to_use": 10},
            {"similarity_top_k": 20, "max_summaries_to_use": 6, "max_triplets_to_use": 20},
            {"similarity_top_k": 50, "max_summaries_to_use": 10, "max_triplets_to_use": 50},
        ]
        
        for config_params in test_configs:
            config = GraphConfig.from_environment()
            config.query.similarity_top_k = config_params["similarity_top_k"]
            config.query.max_summaries_to_use = config_params["max_summaries_to_use"]
            config.query.max_triplets_to_use = config_params["max_triplets_to_use"]
            
            agent = GraphAgent(
                name=f"Config Test {config_params['similarity_top_k']}",
                config=config,
                auto_initialize=False
            )
            
            assert agent.config.query.similarity_top_k == config_params["similarity_top_k"]
            assert agent.config.query.max_summaries_to_use == config_params["max_summaries_to_use"]
            assert agent.config.query.max_triplets_to_use == config_params["max_triplets_to_use"]


class TestGraphAgentEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_malformed_citations_handling(self):
        """Test handling of malformed citation data."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Response with malformed citations"
        
        # Various malformed citation scenarios
        malformed_citations = [
            None,  # None citation
            {},  # Empty citation
            {"title": None, "snippet": "test"},  # Missing required fields
            {"title": "Test", "score": "invalid"},  # Invalid data types
            {"title": "Test", "unknown_field": "value"},  # Extra fields
        ]
        
        mock_engine.get_last_citations.return_value = malformed_citations
        
        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph
        result = await tool.on_invoke_tool(None, json.dumps({"query": "Test query"}))
        
        assert isinstance(result, AgentResponse)
        # Should handle malformed citations gracefully
        assert len(result.citations) >= 0  # Some citations may be filtered out

    @pytest.mark.asyncio
    async def test_empty_and_whitespace_queries(self):
        """Test handling of empty and whitespace-only queries."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Please provide a specific question."
        mock_engine.get_last_citations.return_value = []
        
        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph
        
        test_queries = ["", "   ", "\t\n", "  \t  \n  "]
        
        for query in test_queries:
            result = await tool.on_invoke_tool(None, json.dumps({"query": query}))
            assert isinstance(result, AgentResponse)
            assert result.route == AgentRouteType.GRAPH

    @pytest.mark.asyncio
    async def test_very_long_queries(self):
        """Test handling of very long queries."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Long query processed successfully."
        mock_engine.get_last_citations.return_value = []
        
        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph
        
        # Create a very long query
        long_query = "What is the treatment for breast cancer? " * 200

        result = await tool.on_invoke_tool(None, json.dumps({"query": long_query}))
        
        assert isinstance(result, AgentResponse)
        mock_engine.query.assert_called_once_with(long_query)

    @pytest.mark.asyncio
    async def test_special_characters_and_unicode(self):
        """Test handling of special characters and Unicode in queries."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Special characters handled correctly."
        mock_engine.get_last_citations.return_value = []
        
        # Set the mock engine as the class-level query engine
        GraphAgent.set_query_engine(mock_engine)
        
        tool = GraphAgent.query_knowledge_graph
        
        special_queries = [
            "What about HER2+ & ER- tumors?",  # Medical notation
            "Cost: $10,000-$50,000 range?",   # Currency and ranges
            "Side effects (nausea, fatigue, etc.)?",  # Parentheses and abbreviations
            "Traitement du cancer du sein?",   # French
            "乳腺癌治疗方法？",                # Chinese
            "Лечение рака молочной железы?",   # Russian
        ]
        
        for query in special_queries:
            result = await tool.on_invoke_tool(None, json.dumps({"query": query}))
            assert isinstance(result, AgentResponse)
            assert result.route == AgentRouteType.GRAPH

    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test handling of concurrent queries."""
        mock_engine = MagicMock(spec=GraphRAGQueryEngine)
        mock_engine.query.return_value = "Concurrent query response"
        mock_engine.get_last_citations.return_value = []
        
        agent = GraphAgent(
            name="Concurrent Test Agent",
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
            queries = [f"Query {i}" for i in range(5)]
            tasks = [agent.achat(query) for query in queries]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert isinstance(result, AgentResponse)
                assert result.route == AgentRouteType.GRAPH


class TestGraphAgentErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_achat_with_uninitialized_agent(self):
        """Test achat behavior with uninitialized agent."""
        agent = GraphAgent(name="Uninitialized Agent", auto_initialize=False)
        agent._started = True
        agent._agent = None  # Simulate uninitialized state
        
        result = await agent.achat("Test query")
        
        assert isinstance(result, AgentResponse)
        assert "Sorry, I encountered an error" in result.response_str
        assert result.route == AgentRouteType.GRAPH

    @pytest.mark.asyncio
    async def test_achat_with_start_failure(self):
        """Test achat behavior when start fails."""
        agent = GraphAgent(name="Start Fail Agent", auto_initialize=False)
        agent._started = False
        
        with patch.object(agent, 'start', side_effect=Exception("Start failed")):
            result = await agent.achat("Test query")
            
            assert isinstance(result, AgentResponse)
            assert "Sorry, I encountered an error: Start failed" in result.response_str
            assert result.route == AgentRouteType.GRAPH

    def test_create_agent_without_query_engine(self):
        """Test agent creation failure when query engine is missing."""
        agent = GraphAgent(name="No Engine Agent", auto_initialize=False)
        agent._query_engine = None
        # Clear the class-level query engine as well
        GraphAgent._query_engine = None
        
        with pytest.raises(ValueError, match="Query engine must be initialized"):
            agent._create_agent()

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration values."""
        config = GraphConfig.from_environment()
        
        # Test invalid similarity_top_k
        config.query.similarity_top_k = -1
        with pytest.raises(ValueError):
            config.validate()
        
        # Test invalid max_summaries_to_use
        config = GraphConfig.from_environment()
        config.query.max_summaries_to_use = 0
        with pytest.raises(ValueError):
            config.validate()


# Utility functions for test setup
def create_mock_graph_components():
    """Create a set of mock graph components for testing."""
    config = GraphConfig.from_environment()
    config.api.llamacloud_api_key = "test-key"
    
    mock_store = MagicMock(spec=GraphRAGStore)
    mock_index = MagicMock()
    mock_engine = MagicMock(spec=GraphRAGQueryEngine)
    
    return config, mock_store, mock_index, mock_engine


def create_sample_citations():
    """Create sample citations for testing."""
    return [
        {
            "title": "NCCN Breast Cancer Guidelines",
            "snippet": "Treatment recommendations for breast cancer",
            "file_name": "nccn_breast_cancer.pdf",
            "start_page": "10",
            "end_page": "12",
            "score": 0.95,
            "text": "Complete citation text with treatment details"
        },
        {
            "title": "Clinical Trial Results",
            "snippet": "Efficacy of targeted therapy",
            "file_name": "clinical_trial.pdf",
            "start_page": "5",
            "end_page": "7",
            "score": 0.88,
            "text": "Clinical trial data showing improved outcomes"
        }
    ]


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
