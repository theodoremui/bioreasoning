"""
Tests for bioagents.models.citation module.
"""

import pytest
from bioagents.models.source import Source


class TestCitation:
    """Test Citation dataclass."""
    
    def test_citation_creation_minimal(self):
        """Test creating Citation with minimal required parameters."""
        citation = Source(
            url="https://example.com",
            title="Test Article",
            snippet="Test snippet",
            source="web"
        )
        
        assert citation.url == "https://example.com"
        assert citation.title == "Test Article"
        assert citation.snippet == "Test snippet"
        assert citation.source == "web"
    
    def test_citation_creation_empty_snippet(self):
        """Test creating Citation with empty snippet."""
        citation = Source(
            url="https://example.com",
            title="Test Article",
            snippet="",
            source="pubmed"
        )
        
        assert citation.snippet == ""
        assert citation.source == "pubmed"
    
    def test_citation_equality(self):
        """Test equality comparison of Citation objects."""
        citation1 = Source(
            url="https://example.com",
            title="Test Article",
            snippet="Test snippet",
            source="web"
        )
        
        citation2 = Source(
            url="https://example.com", 
            title="Test Article",
            snippet="Test snippet",
            source="web"
        )
        
        assert citation1 == citation2
    
    def test_citation_inequality(self):
        """Test inequality of Citation objects with different values."""
        citation1 = Source(
            url="https://example.com",
            title="Test Article 1",
            snippet="Test snippet",
            source="web"
        )
        
        citation2 = Source(
            url="https://example.com",
            title="Test Article 2", 
            snippet="Test snippet",
            source="web"
        )
        
        assert citation1 != citation2
    
    def test_citation_str_representation(self):
        """Test string representation of Citation."""
        citation = Source(
            url="https://example.com",
            title="Test Article",
            snippet="Test snippet",
            source="web"
        )
        
        str_repr = str(citation)
        assert "Test Article" in str_repr
        assert "https://example.com" in str_repr
    
    def test_citation_with_long_snippet(self):
        """Test Citation with long snippet."""
        long_snippet = "This is a very long snippet " * 20
        citation = Source(
            url="https://example.com",
            title="Test Article",
            snippet=long_snippet,
            source="pubmed"
        )
        
        assert citation.snippet == long_snippet
        assert len(citation.snippet) > 100
    
    def test_citation_with_special_characters(self):
        """Test Citation with special characters in fields."""
        citation = Source(
            url="https://example.com/article?id=123&category=research",
            title="Test Article: A Study on AI & ML",
            snippet="This snippet contains special chars: @#$%^&*()",
            source="arxiv"
        )
        
        assert "?" in citation.url
        assert "&" in citation.url
        assert ":" in citation.title
        assert "&" in citation.title
        assert "@#$%^&*()" in citation.snippet
    
    def test_citation_different_sources(self):
        """Test Citation with different source types."""
        sources = ["web", "pubmed", "arxiv", "custom_db", "manual"]
        
        for source in sources:
            citation = Source(
                url=f"https://example.com/{source}",
                title=f"Article from {source}",
                snippet=f"Snippet from {source}",
                source=source
            )
            assert citation.source == source
    
    def test_citation_unicode_content(self):
        """Test Citation with unicode content."""
        citation = Source(
            url="https://example.com/unicode",
            title="Article with 中文 and émojis 🧬",
            snippet="This contains unicode: αβγ δε ζη",
            source="international"
        )
        
        assert "中文" in citation.title
        assert "🧬" in citation.title
        assert "αβγ" in citation.snippet
    
    def test_citation_with_none_values(self):
        """Test that Citation can handle None values (though not recommended)."""
        citation = Source(
            url=None,
            title="Test",
            snippet="Test",
            source="web"
        )
        
        assert citation.url is None
        assert citation.title == "Test"
        assert citation.snippet == "Test"
        assert citation.source == "web"
    
    def test_citation_empty_url(self):
        """Test Citation with empty URL."""
        citation = Source(
            url="",
            title="Test Article",
            snippet="Test snippet",
            source="manual"
        )
        
        assert citation.url == ""
        # Should still be a valid citation object
        assert citation.title == "Test Article" 