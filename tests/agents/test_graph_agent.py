"""
Tests for bioagents.agents.graph_agent module.
"""

import pytest
from unittest.mock import MagicMock, patch

from bioagents.agents.graph_agent import GraphAgent, INSTRUCTIONS, HANDOFF_DESCRIPTION
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM


class TestGraphAgent:
    """Test GraphAgent class."""

    def test_graph_agent_creation_default(self):
        """Test creating GraphAgent with default parameters."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent")
            
            assert agent.name == "TestGraphAgent"
            assert agent.model_name == LLM.GPT_4_1_NANO
            assert agent.instructions == INSTRUCTIONS
            assert agent.handoff_description == HANDOFF_DESCRIPTION
            assert agent.pdf_path == "data/nccn_breast_cancer.pdf"
            assert agent.auto_initialize is True

    def test_graph_agent_instructions_content(self):
        """Test that instructions contain expected content."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent")
            
            assert "NCCN breast cancer guidelines" in agent.instructions
            assert "knowledge graph" in agent.instructions
            assert "directly answer" in agent.instructions
            assert "citation information" in agent.instructions
            assert "[Graph]" in agent.instructions

    def test_graph_agent_handoff_description(self):
        """Test handoff description content."""
        assert "NCCN" in HANDOFF_DESCRIPTION
        assert "breast cancer guidelines" in HANDOFF_DESCRIPTION
        assert "knowledge graph" in HANDOFF_DESCRIPTION

    def test_graph_agent_class_properties(self):
        """Test class-level query engine management."""
        # Test initial state - query_engine is a property that returns _query_engine
        assert GraphAgent._query_engine is None
        
        # Test setting query engine
        mock_query_engine = MagicMock()
        GraphAgent.set_query_engine(mock_query_engine)
        assert GraphAgent._query_engine == mock_query_engine
        
        # Test resetting to None
        GraphAgent.set_query_engine(None)
        assert GraphAgent._query_engine is None

    def test_graph_agent_inheritance(self):
        """Test that GraphAgent properly inherits from BaseAgent."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent")
            
            # Should have BaseAgent attributes
            assert hasattr(agent, 'name')
            assert hasattr(agent, 'model_name')
            assert hasattr(agent, 'instructions')
            assert hasattr(agent, 'timeout')
            assert hasattr(agent, '_response_judge')
            
            # Should have BaseAgent methods
            assert hasattr(agent, 'achat')
            assert hasattr(agent, 'simple_achat')
            assert hasattr(agent, '_construct_response')

    def test_graph_agent_default_model(self):
        """Test that GraphAgent uses GPT_4_1_NANO by default."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent")
            
            assert agent.model_name == LLM.GPT_4_1_NANO

    def test_graph_agent_custom_model(self):
        """Test that GraphAgent can use custom models."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            custom_model = LLM.GPT_4_1_MINI
            agent = GraphAgent(name="TestGraphAgent", model_name=custom_model)
            
            assert agent.model_name == custom_model

    def test_graph_agent_instructions_construction(self):
        """Test that instructions are constructed correctly."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent")
            
            # Instructions should be constructed in __init__
            expected_parts = [
                "NCCN breast cancer guidelines",
                "knowledge graph",
                "directly answer",
                "citation information",
                "## Response Instructions:",
                "[Graph]"
            ]
            
            for part in expected_parts:
                assert part in agent.instructions

    def test_graph_agent_handoff_description_construction(self):
        """Test that handoff description is constructed correctly."""
        expected_parts = [
            "NCCN",
            "breast cancer guidelines",
            "knowledge graph"
        ]
        
        for part in expected_parts:
            assert part in HANDOFF_DESCRIPTION

    def test_graph_agent_agent_name_consistency(self):
        """Test that agent name is consistent across initialization."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent_name = "ConsistentGraphAgent"
            agent = GraphAgent(name=agent_name)
            
            assert agent.name == agent_name
            
            # Test with different name
            different_name = "DifferentGraphAgent"
            different_agent = GraphAgent(name=different_name)
            
            assert different_agent.name == different_name

    def test_graph_agent_instructions_nccn_focus(self):
        """Test that instructions focus on NCCN guidelines."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent")
            
            # Should focus on NCCN guidelines
            assert "NCCN" in agent.instructions
            assert "breast cancer guidelines" in agent.instructions
            assert "knowledge graph" in agent.instructions

    def test_graph_agent_instructions_citation_requirement(self):
        """Test that instructions require citations."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent")
            
            # Should require citations
            assert "citation information" in agent.instructions
            assert "source documents" in agent.instructions

    def test_graph_agent_instructions_prefix_requirement(self):
        """Test that instructions require [Graph] prefix."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent")
            
            # Should require [Graph] prefix
            assert "[Graph]" in agent.instructions
            assert "Prepend the response with" in agent.instructions

    def test_graph_agent_config_injection(self):
        """Test that config is properly injected."""
        mock_config = MagicMock()
        
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config_class:
            mock_config_class.from_environment.return_value = MagicMock()
            agent = GraphAgent(name="TestGraphAgent", config=mock_config)
            
            assert agent.config == mock_config

    def test_graph_agent_pdf_path_customization(self):
        """Test that PDF path can be customized."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            custom_path = "custom/path/to/document.pdf"
            agent = GraphAgent(name="TestGraphAgent", pdf_path=custom_path)
            
            assert agent.pdf_path == custom_path

    def test_graph_agent_auto_initialize_flag(self):
        """Test that auto_initialize flag is properly set."""
        with patch('bioagents.agents.graph_agent.GraphConfig') as mock_config:
            mock_config.from_environment.return_value = MagicMock()
            
            # Test with auto_initialize=True (default)
            agent1 = GraphAgent(name="TestGraphAgent")
            assert agent1.auto_initialize is True
            
            # Test with auto_initialize=False
            agent2 = GraphAgent(name="TestGraphAgent", auto_initialize=False)
            assert agent2.auto_initialize is False
