"""
Tests for bioagents.agents.common module.
"""

import pytest
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.source import Source


class TestAgentRouteType:
    """Test AgentRouteType enum."""

    def test_agent_route_values(self):
        """Test that AgentRouteType has expected values."""
        assert AgentRouteType.BIOMCP.value == "biomcp"
        assert AgentRouteType.CHITCHAT.value == "chitchat"
        assert AgentRouteType.GRAPH.value == "graph"
        assert AgentRouteType.WEBSEARCH.value == "websearch"
        assert AgentRouteType.LLAMAMCP.value == "llamamcp"
        assert AgentRouteType.LLAMARAG.value == "llamarag"
        assert AgentRouteType.CONCIERGE.value == "concierge"
        assert AgentRouteType.REASONING.value == "reasoning"

    def test_agent_route_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {
            "BIOMCP",
            "CHITCHAT",
            "GRAPH",
            "WEBSEARCH",
            "LLAMAMCP",
            "LLAMARAG",
            "CONCIERGE",
            "REASONING",
        }
        actual_members = {member.name for member in AgentRouteType}
        assert actual_members == expected_members


class TestAgentResponse:
    """Test AgentResponse dataclass."""

    def test_agent_response_creation_minimal(self):
        """Test creating AgentResponse with minimal parameters."""
        response = AgentResponse(response_str="Test response")

        assert response.response_str == "Test response"
        assert response.citations == []
        assert response.judgement == ""
        assert response.route == AgentRouteType.REASONING

    def test_agent_response_creation_full(self, sample_citations):
        """Test creating AgentResponse with all parameters."""
        response = AgentResponse(
            response_str="Test response",
            citations=sample_citations,
            judgement="Judge response",
            route=AgentRouteType.WEBSEARCH,
        )

        assert response.response_str == "Test response"
        assert response.citations == sample_citations
        assert response.judgement == "Judge response"
        assert response.route == AgentRouteType.WEBSEARCH

    def test_agent_response_mutable_default_fixed(self):
        """Test that mutable default for citations is properly handled."""
        response1 = AgentResponse(response_str="Response 1")
        response2 = AgentResponse(response_str="Response 2")

        # Add citation to first response
        citation = Source(
            url="https://example.com",
            title="Test",
            snippet="Test snippet",
            source="test",
        )
        response1.citations.append(citation)

        # Second response should still have empty citations
        assert len(response1.citations) == 1
        assert len(response2.citations) == 0

    def test_agent_response_str_representation(self):
        """Test string representation of AgentResponse."""
        response = AgentResponse(
            response_str="Test response", route=AgentRouteType.WEBSEARCH
        )

        # Should not raise an error
        str_repr = str(response)
        assert "Test response" in str_repr

    def test_agent_response_equality(self, sample_citations):
        """Test equality comparison of AgentResponse objects."""
        response1 = AgentResponse(
            response_str="Test response",
            citations=sample_citations,
            judgement="Judge",
            route=AgentRouteType.WEBSEARCH,
        )

        response2 = AgentResponse(
            response_str="Test response",
            citations=sample_citations,
            judgement="Judge",
            route=AgentRouteType.WEBSEARCH,
        )

        assert response1 == response2

    def test_agent_response_with_empty_citations(self):
        """Test AgentResponse with explicitly empty citations."""
        response = AgentResponse(
            response_str="Test response", citations=[], route=AgentRouteType.CHITCHAT
        )

        assert response.citations == []
        assert response.route == AgentRouteType.CHITCHAT

    def test_agent_response_citations_list_methods(self):
        """Test that citations list supports standard list operations."""
        response = AgentResponse(response_str="Test")

        # Test append
        citation = Source(
            url="https://example.com",
            title="Test",
            snippet="Test snippet",
            source="test",
        )
        response.citations.append(citation)
        assert len(response.citations) == 1

        # Test extend
        more_citations = [
            Source(
                url="https://example2.com",
                title="Test2",
                snippet="Test2",
                source="test2",
            )
        ]
        response.citations.extend(more_citations)
        assert len(response.citations) == 2

        # Test indexing
        assert response.citations[0] == citation

        # Test iteration
        citation_urls = [c.url for c in response.citations]
        assert "https://example.com" in citation_urls
        assert "https://example2.com" in citation_urls

    def test_agent_response_with_different_route_types(self):
        """Test AgentResponse with all different route types."""
        routes = [
            AgentRouteType.BIOMCP,
            AgentRouteType.CHITCHAT,
            AgentRouteType.GRAPH,
            AgentRouteType.LLAMAMCP,
            AgentRouteType.LLAMARAG,
            AgentRouteType.WEBSEARCH,
            AgentRouteType.CONCIERGE,
            AgentRouteType.REASONING
        ]
        
        for route in routes:
            response = AgentResponse(
                response_str=f"Response for {route.value}",
                route=route
            )
            assert response.route == route
            assert response.route.value == route.value

    def test_agent_response_judgement_variations(self):
        """Test AgentResponse with different judgement formats."""
        judgements = [
            "",
            "Simple judgement",
            "**Score**: 0.85\n**Assessment**: Good response",
            "Multi-line\njudgement\nwith\nbreaks",
            "Judgement with special chars: !@#$%^&*()",
            "Very long judgement " * 100,  # Test with very long text
        ]
        
        for judgement in judgements:
            response = AgentResponse(
                response_str="Test response",
                judgement=judgement
            )
            assert response.judgement == judgement

    def test_agent_response_response_str_variations(self):
        """Test AgentResponse with different response string formats."""
        responses = [
            "",
            "Simple response",
            "Response with\nnewlines",
            "Response with special chars: !@#$%^&*()",
            "Very long response " * 100,  # Test with very long text
            "Response with unicode: 🚀🌟💡",
        ]
        
        for response_str in responses:
            response = AgentResponse(response_str=response_str)
            assert response.response_str == response_str

    def test_agent_response_citations_edge_cases(self):
        """Test AgentResponse with edge cases for citations."""
        # Test with None citations (should not happen but test robustness)
        response = AgentResponse(response_str="Test")
        response.citations = None
        assert response.citations is None
        
        # Test with very large number of citations
        large_citations = []
        for i in range(1000):
            citation = Source(
                url=f"https://example{i}.com",
                title=f"Article {i}",
                snippet=f"Snippet {i}",
                source="test"
            )
            large_citations.append(citation)
        
        response = AgentResponse(
            response_str="Test",
            citations=large_citations
        )
        assert len(response.citations) == 1000
        assert response.citations[0].url == "https://example0.com"
        assert response.citations[999].url == "https://example999.com"

    def test_agent_response_immutability_after_creation(self):
        """Test that AgentResponse fields can be modified after creation."""
        response = AgentResponse(response_str="Original")
        
        # Modify all fields
        response.response_str = "Modified response"
        response.judgement = "Modified judgement"
        response.route = AgentRouteType.GRAPH
        
        citation = Source(
            url="https://example.com",
            title="Test",
            snippet="Test snippet",
            source="test"
        )
        response.citations.append(citation)
        
        assert response.response_str == "Modified response"
        assert response.judgement == "Modified judgement"
        assert response.route == AgentRouteType.GRAPH
        assert len(response.citations) == 1

    def test_agent_response_repr_method(self):
        """Test the __repr__ method of AgentResponse."""
        response = AgentResponse(
            response_str="Test response",
            judgement="Test judgement",
            route=AgentRouteType.WEBSEARCH
        )
        
        repr_str = repr(response)
        assert isinstance(repr_str, str)
        assert "AgentResponse" in repr_str
        assert "Test response" in repr_str

    def test_agent_response_equality_consistency(self):
        """Test that AgentResponse objects have consistent equality."""
        response1 = AgentResponse(
            response_str="Test",
            citations=[],
            judgement="Judge",
            route=AgentRouteType.WEBSEARCH
        )
        
        response2 = AgentResponse(
            response_str="Test",
            citations=[],
            judgement="Judge",
            route=AgentRouteType.WEBSEARCH
        )
        
        # Same objects should be equal
        assert response1 == response2
        
        # Different objects should not be equal
        response3 = AgentResponse(response_str="Different")
        assert response1 != response3

    def test_agent_route_type_string_conversion(self):
        """Test AgentRouteType string conversion."""
        for route in AgentRouteType:
            # str(route) returns the enum representation like "AgentRouteType.BIOMCP"
            # route.value returns the actual value like "biomcp"
            assert isinstance(str(route), str)
            assert isinstance(route.value, str)
            assert route.value == route.value  # Basic sanity check

    def test_agent_route_type_comparison(self):
        """Test AgentRouteType comparison operations."""
        assert AgentRouteType.WEBSEARCH == AgentRouteType.WEBSEARCH
        assert AgentRouteType.WEBSEARCH != AgentRouteType.GRAPH
        assert AgentRouteType.WEBSEARCH in AgentRouteType

    def test_agent_route_type_iteration(self):
        """Test iterating over AgentRouteType enum."""
        routes = list(AgentRouteType)
        assert len(routes) == 8  # Should have 8 route types
        
        # All should be AgentRouteType instances
        for route in routes:
            assert isinstance(route, AgentRouteType)

    def test_agent_response_with_complex_citations(self):
        """Test AgentResponse with complex citation scenarios."""
        # Test with citations having all possible fields
        complex_citation = Source(
            url="https://example.com/path?param=value#fragment",
            title="Complex Title with Special Chars: !@#$%",
            snippet="Complex snippet with\nnewlines and special chars: 🚀",
            source="complex_source"
        )
        
        response = AgentResponse(
            response_str="Complex response",
            citations=[complex_citation],
            judgement="Complex judgement with score: 0.95",
            route=AgentRouteType.LLAMARAG
        )
        
        assert len(response.citations) == 1
        assert response.citations[0].url == "https://example.com/path?param=value#fragment"
        assert response.citations[0].title == "Complex Title with Special Chars: !@#$%"
        assert "🚀" in response.citations[0].snippet
