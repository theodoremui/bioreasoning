"""
Tests for Configuration Management

Comprehensive tests for the configuration system including validation,
environment variable loading, and error handling.

Author: Theodore Mui
Date: 2025-08-24
"""

import os
import pytest
from unittest.mock import patch

from bioagents.graph.config import (
    GraphConfig, PerformanceConfig, DatabaseConfig, 
    APIConfig, QueryConfig, CacheConfig
)


class TestPerformanceConfig:
    """Tests for PerformanceConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PerformanceConfig()
        assert config.kg_extraction_workers == 8
        assert config.pdf_parse_workers == 8
        assert config.max_paths_per_chunk == 2
        assert config.max_cluster_size == 5
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = PerformanceConfig(
            kg_extraction_workers=4,
            pdf_parse_workers=6,
            max_paths_per_chunk=3,
            max_cluster_size=10
        )
        assert config.kg_extraction_workers == 4
        assert config.pdf_parse_workers == 6
        assert config.max_paths_per_chunk == 3
        assert config.max_cluster_size == 10
    
    def test_validation_kg_workers(self):
        """Test validation of kg_extraction_workers."""
        with pytest.raises(ValueError, match="kg_extraction_workers must be >= 1"):
            PerformanceConfig(kg_extraction_workers=0)
    
    def test_validation_pdf_workers(self):
        """Test validation of pdf_parse_workers."""
        with pytest.raises(ValueError, match="pdf_parse_workers must be >= 1"):
            PerformanceConfig(pdf_parse_workers=-1)
    
    def test_validation_max_paths(self):
        """Test validation of max_paths_per_chunk."""
        with pytest.raises(ValueError, match="max_paths_per_chunk must be >= 1"):
            PerformanceConfig(max_paths_per_chunk=0)
    
    def test_validation_cluster_size(self):
        """Test validation of max_cluster_size."""
        with pytest.raises(ValueError, match="max_cluster_size must be >= 1"):
            PerformanceConfig(max_cluster_size=0)


class TestDatabaseConfig:
    """Tests for DatabaseConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        assert config.neo4j_username == "neo4j"
        assert config.neo4j_password == "Salesforce1"
        assert config.neo4j_url == "bolt://localhost:7687"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            neo4j_username="custom_user",
            neo4j_password="custom_pass",
            neo4j_url="bolt://custom:7687"
        )
        assert config.neo4j_username == "custom_user"
        assert config.neo4j_password == "custom_pass"
        assert config.neo4j_url == "bolt://custom:7687"
    
    def test_validation_empty_username(self):
        """Test validation of empty username."""
        with pytest.raises(ValueError, match="neo4j_username cannot be empty"):
            DatabaseConfig(neo4j_username="")
    
    def test_validation_empty_password(self):
        """Test validation of empty password."""
        with pytest.raises(ValueError, match="neo4j_password cannot be empty"):
            DatabaseConfig(neo4j_password="")
    
    def test_validation_empty_url(self):
        """Test validation of empty URL."""
        with pytest.raises(ValueError, match="neo4j_url cannot be empty"):
            DatabaseConfig(neo4j_url="")


class TestAPIConfig:
    """Tests for APIConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = APIConfig()
        assert config.llamacloud_api_key is None
        assert config.openai_model == "gpt-4.1-mini"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = APIConfig(
            llamacloud_api_key="test_key",
            openai_model="gpt-3.5-turbo"
        )
        assert config.llamacloud_api_key == "test_key"
        assert config.openai_model == "gpt-3.5-turbo"
    
    def test_validation_empty_model(self):
        """Test validation of empty model."""
        with pytest.raises(ValueError, match="openai_model cannot be empty"):
            APIConfig(openai_model="")


class TestQueryConfig:
    """Tests for QueryConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = QueryConfig()
        assert config.similarity_top_k == 20
        assert config.max_summaries_to_use == 6
        assert config.max_triplets_to_use == 20
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = QueryConfig(
            similarity_top_k=10,
            max_summaries_to_use=3,
            max_triplets_to_use=15
        )
        assert config.similarity_top_k == 10
        assert config.max_summaries_to_use == 3
        assert config.max_triplets_to_use == 15
    
    def test_validation_similarity_top_k(self):
        """Test validation of similarity_top_k."""
        with pytest.raises(ValueError, match="similarity_top_k must be >= 1"):
            QueryConfig(similarity_top_k=0)
    
    def test_validation_max_summaries(self):
        """Test validation of max_summaries_to_use."""
        with pytest.raises(ValueError, match="max_summaries_to_use must be >= 1"):
            QueryConfig(max_summaries_to_use=0)
    
    def test_validation_max_triplets(self):
        """Test validation of max_triplets_to_use."""
        with pytest.raises(ValueError, match="max_triplets_to_use must be >= 1"):
            QueryConfig(max_triplets_to_use=0)


class TestCacheConfig:
    """Tests for CacheConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.communities_cache_file == "data/nccn_communities.json"
        assert config.validate_cache_signature is True
        assert config.prefer_cache_when_graph_empty is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = CacheConfig(
            communities_cache_file="custom/cache.json",
            validate_cache_signature=False,
            prefer_cache_when_graph_empty=False
        )
        assert config.communities_cache_file == "custom/cache.json"
        assert config.validate_cache_signature is False
        assert config.prefer_cache_when_graph_empty is False
    
    def test_validation_empty_cache_file(self):
        """Test validation of empty cache file."""
        with pytest.raises(ValueError, match="communities_cache_file cannot be empty"):
            CacheConfig(communities_cache_file="")


class TestGraphConfig:
    """Tests for GraphConfig class."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        config = GraphConfig()
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.query, QueryConfig)
        assert isinstance(config.cache, CacheConfig)
    
    def test_custom_initialization(self):
        """Test initialization with custom components."""
        perf = PerformanceConfig(kg_extraction_workers=4)
        db = DatabaseConfig(neo4j_username="custom")
        
        config = GraphConfig(performance=perf, database=db)
        assert config.performance.kg_extraction_workers == 4
        assert config.database.neo4j_username == "custom"
    
    @patch.dict(os.environ, {
        'KG_EXTRACTION_WORKERS': '6',
        'PDF_PARSE_WORKERS': '4',
        'NEO4J_USERNAME': 'test_user',
        'LLAMACLOUD_API_KEY': 'test_key',
        'SIMILARITY_TOP_K': '15'
    })
    def test_from_environment(self):
        """Test loading configuration from environment variables."""
        config = GraphConfig.from_environment()
        
        assert config.performance.kg_extraction_workers == 6
        assert config.performance.pdf_parse_workers == 4
        assert config.database.neo4j_username == "test_user"
        assert config.api.llamacloud_api_key == "test_key"
        assert config.query.similarity_top_k == 15
    
    def test_get_method(self):
        """Test get method with dot notation."""
        config = GraphConfig()
        
        # Test existing keys
        assert config.get('performance.kg_extraction_workers') == 8
        assert config.get('database.neo4j_username') == "neo4j"
        assert config.get('api.openai_model') == "gpt-4.1-mini"
        
        # Test non-existing keys with default
        assert config.get('nonexistent.key', 'default') == 'default'
        assert config.get('performance.nonexistent', 42) == 42
    
    def test_validate_method(self):
        """Test validate method."""
        config = GraphConfig()
        assert config.validate() is True
        
        # Test with invalid configuration
        config.performance.kg_extraction_workers = -1
        with pytest.raises(ValueError):
            config.validate()
    
    def test_to_dict_method(self):
        """Test to_dict method."""
        config = GraphConfig()
        config_dict = config.to_dict()
        
        assert 'performance' in config_dict
        assert 'database' in config_dict
        assert 'api' in config_dict
        assert 'query' in config_dict
        assert 'cache' in config_dict
        
        assert config_dict['performance']['kg_extraction_workers'] == 8
        assert config_dict['database']['neo4j_username'] == "neo4j"
    
    def test_display_summary(self):
        """Test display_summary method."""
        config = GraphConfig()
        summary = config.display_summary()
        
        assert "🚀 Graph Processing Configuration:" in summary
        assert "KG Extraction Workers: 8" in summary
        assert "Neo4j URL: bolt://localhost:7687" in summary
        assert "OpenAI Model: gpt-4.1-mini" in summary
    
    def test_environment_variable_types(self):
        """Test that environment variables are properly converted to correct types."""
        with patch.dict(os.environ, {
            'KG_EXTRACTION_WORKERS': '10',
            'VALIDATE_CACHE_SIGNATURE': 'false',
            'PREFER_CACHE_WHEN_GRAPH_EMPTY': 'true'
        }):
            config = GraphConfig.from_environment()
            
            assert isinstance(config.performance.kg_extraction_workers, int)
            assert config.performance.kg_extraction_workers == 10
            assert isinstance(config.cache.validate_cache_signature, bool)
            assert config.cache.validate_cache_signature is False
            assert isinstance(config.cache.prefer_cache_when_graph_empty, bool)
            assert config.cache.prefer_cache_when_graph_empty is True
    
    def test_nested_get_with_invalid_path(self):
        """Test get method with invalid nested path."""
        config = GraphConfig()
        
        # Test invalid attribute path
        assert config.get('performance.invalid_attr') is None
        assert config.get('invalid_section.attr') is None
        assert config.get('performance.invalid_attr', 'fallback') == 'fallback'
