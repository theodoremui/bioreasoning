"""
Tests for bioagents.agents.common module.
"""

import pytest
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.citation import Citation


class TestAgentRouteType:
    """Test AgentRouteType enum."""
    
    def test_agent_route_values(self):
        """Test that AgentRouteType has expected values."""
        assert AgentRouteType.BIOMCP.value == "biomcp"
        assert AgentRouteType.CHITCHAT.value == "chitchat"
        assert AgentRouteType.WEBSEARCH.value == "websearch"
        assert AgentRouteType.LLAMAMCP.value == "llamamcp"
        assert AgentRouteType.LLAMARAG.value == "llamarag"
        assert AgentRouteType.CONCIERGE.value == "concierge"
        assert AgentRouteType.REASONING.value == "reasoning"
    
    def test_agent_route_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {"BIOMCP", "CHITCHAT", "WEBSEARCH", "LLAMAMCP", "LLAMARAG", "CONCIERGE", "REASONING"}
        actual_members = {member.name for member in AgentRouteType}
        assert actual_members == expected_members


class TestAgentResponse:
    """Test AgentResponse dataclass."""
    
    def test_agent_response_creation_minimal(self):
        """Test creating AgentResponse with minimal parameters."""
        response = AgentResponse(response_str="Test response")
        
        assert response.response_str == "Test response"
        assert response.citations == []
        assert response.judge_response == ""
        assert response.route == AgentRouteType.REASONING
    
    def test_agent_response_creation_full(self, sample_citations):
        """Test creating AgentResponse with all parameters."""
        response = AgentResponse(
            response_str="Test response",
            citations=sample_citations,
            judge_response="Judge response",
            route=AgentRouteType.WEBSEARCH
        )
        
        assert response.response_str == "Test response"
        assert response.citations == sample_citations
        assert response.judge_response == "Judge response"
        assert response.route == AgentRouteType.WEBSEARCH
    
    def test_agent_response_mutable_default_fixed(self):
        """Test that mutable default for citations is properly handled."""
        response1 = AgentResponse(response_str="Response 1")
        response2 = AgentResponse(response_str="Response 2")
        
        # Add citation to first response
        citation = Citation(
            url="https://example.com",
            title="Test",
            snippet="Test snippet",
            source="test"
        )
        response1.citations.append(citation)
        
        # Second response should still have empty citations
        assert len(response1.citations) == 1
        assert len(response2.citations) == 0
    
    def test_agent_response_str_representation(self):
        """Test string representation of AgentResponse."""
        response = AgentResponse(
            response_str="Test response",
            route=AgentRouteType.WEBSEARCH
        )
        
        # Should not raise an error
        str_repr = str(response)
        assert "Test response" in str_repr
    
    def test_agent_response_equality(self, sample_citations):
        """Test equality comparison of AgentResponse objects."""
        response1 = AgentResponse(
            response_str="Test response",
            citations=sample_citations,
            judge_response="Judge",
            route=AgentRouteType.WEBSEARCH
        )
        
        response2 = AgentResponse(
            response_str="Test response",
            citations=sample_citations,
            judge_response="Judge", 
            route=AgentRouteType.WEBSEARCH
        )
        
        assert response1 == response2
    
    def test_agent_response_with_empty_citations(self):
        """Test AgentResponse with explicitly empty citations."""
        response = AgentResponse(
            response_str="Test response",
            citations=[],
            route=AgentRouteType.CHITCHAT
        )
        
        assert response.citations == []
        assert response.route == AgentRouteType.CHITCHAT
    
    def test_agent_response_citations_list_methods(self):
        """Test that citations list supports standard list operations."""
        response = AgentResponse(response_str="Test")
        
        # Test append
        citation = Citation(
            url="https://example.com",
            title="Test",
            snippet="Test snippet", 
            source="test"
        )
        response.citations.append(citation)
        assert len(response.citations) == 1
        
        # Test extend
        more_citations = [
            Citation(url="https://example2.com", title="Test2", snippet="Test2", source="test2")
        ]
        response.citations.extend(more_citations)
        assert len(response.citations) == 2
        
        # Test indexing
        assert response.citations[0] == citation
        
        # Test iteration
        citation_urls = [c.url for c in response.citations]
        assert "https://example.com" in citation_urls
        assert "https://example2.com" in citation_urls 