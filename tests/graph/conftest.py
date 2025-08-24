"""
Test Configuration and Fixtures

Provides common fixtures and configuration for graph package tests.

Author: Theodore Mui
Date: 2025-08-24
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any
from llama_index.core.schema import TextNode, Document
from llama_index.core.llms import LLM
from llama_index.core import PropertyGraphIndex

from bioagents.graph.config import GraphConfig, PerformanceConfig, DatabaseConfig
from bioagents.graph.interfaces import (
    ICommunityBuilder, ICommunityCache, IProvenanceBuilder, 
    ISummaryGenerator, IKnowledgeExtractor, IEntityResolver,
    IRankingStrategy, ICitationBuilder, IDocumentProcessor
)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return GraphConfig(
        performance=PerformanceConfig(
            kg_extraction_workers=2,
            pdf_parse_workers=2,
            max_paths_per_chunk=1,
            max_cluster_size=3
        ),
        database=DatabaseConfig(
            neo4j_username="test",
            neo4j_password="test",
            neo4j_url="bolt://localhost:7687"
        )
    )


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock(spec=LLM)
    llm.chat.return_value = Mock()
    llm.chat.return_value.__str__ = Mock(return_value="Test response")
    llm.apredict.return_value = '{"entities": [], "relationships": []}'
    return llm


@pytest.fixture
def sample_text_nodes():
    """Sample text nodes for testing."""
    return [
        TextNode(
            text="Albert Einstein developed the theory of relativity.",
            metadata={
                "doc_id": "test_doc",
                "doc_title": "Test Document",
                "file_path": "/test/path.pdf",
                "page_number": 1,
                "paragraph_index": 0,
                "char_start": 0,
                "char_end": 50,
            }
        ),
        TextNode(
            text="The theory explains the laws of physics.",
            metadata={
                "doc_id": "test_doc",
                "doc_title": "Test Document", 
                "file_path": "/test/path.pdf",
                "page_number": 1,
                "paragraph_index": 1,
                "char_start": 51,
                "char_end": 90,
            }
        )
    ]


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(text="Albert Einstein developed the theory of relativity."),
        Document(text="The theory explains the laws of physics.")
    ]


@pytest.fixture
def sample_triplets():
    """Sample triplets for testing."""
    entity1 = Mock()
    entity1.name = "Albert Einstein"
    entity1.properties = {"entity_description": "Physicist"}
    
    entity2 = Mock()
    entity2.name = "Theory of Relativity"
    entity2.properties = {"entity_description": "Physics theory"}
    
    relation = Mock()
    relation.source_id = "Albert Einstein"
    relation.target_id = "Theory of Relativity"
    relation.label = "developed"
    relation.properties = {
        "relationship_description": "Einstein developed the theory",
        "triplet_key": "Albert Einstein|developed|Theory of Relativity",
        "source_snippet": "Einstein developed relativity theory",
        "provenance_id": "test_prov_id"
    }
    
    return [(entity1, relation, entity2)]


@pytest.fixture
def sample_community_data():
    """Sample community data for testing."""
    return {
        "graph_signature": "test_signature",
        "max_cluster_size": 5,
        "entity_info": {
            "Albert Einstein": [0],
            "Theory of Relativity": [0]
        },
        "community_summary": {
            0: "Einstein developed relativity theory"
        },
        "cluster_assignments": {
            "Albert Einstein": 0,
            "Theory of Relativity": 0
        },
        "community_info": {
            0: [
                {
                    "detail": "Albert Einstein -> Theory of Relativity -> developed -> Einstein developed the theory",
                    "triplet_key": "Albert Einstein|developed|Theory of Relativity"
                }
            ]
        },
        "triplet_provenance": {
            "Albert Einstein|developed|Theory of Relativity": [
                {
                    "source_doc_id": "test_doc",
                    "source_snippet": "Einstein developed relativity theory",
                    "provenance_id": "test_prov_id"
                }
            ]
        },
        "algorithm_metadata": {
            "algorithm": "test_algorithm",
            "parameters": {"max_cluster_size": 5}
        }
    }


@pytest.fixture
def mock_community_builder():
    """Mock community builder for testing."""
    builder = Mock(spec=ICommunityBuilder)
    builder.build_communities.return_value = (
        {"Albert Einstein": [0]},  # entity_info
        {0: [{"detail": "test detail", "triplet_key": "test_key"}]}  # community_info
    )
    builder.get_algorithm_metadata.return_value = {
        "algorithm": "test_algorithm",
        "parameters": {"max_cluster_size": 5}
    }
    return builder


@pytest.fixture
def mock_community_cache():
    """Mock community cache for testing."""
    cache = Mock(spec=ICommunityCache)
    cache.save_communities.return_value = None
    cache.load_communities.return_value = None
    cache.compute_signature.return_value = "test_signature"
    return cache


@pytest.fixture
def mock_summary_generator():
    """Mock summary generator for testing."""
    generator = Mock(spec=ISummaryGenerator)
    generator.generate_summary.return_value = "Test summary"
    generator.get_system_prompt.return_value = "Test prompt"
    return generator


@pytest.fixture
def mock_provenance_builder():
    """Mock provenance builder for testing."""
    builder = Mock(spec=IProvenanceBuilder)
    builder.build_triplet_provenance.return_value = {
        "test_key": [{"provenance_id": "test_id", "source_snippet": "test snippet"}]
    }
    builder.extract_provenance_from_relation.return_value = {
        "provenance_id": "test_id",
        "source_snippet": "test snippet"
    }
    return builder


@pytest.fixture
def mock_entity_resolver():
    """Mock entity resolver for testing."""
    resolver = Mock(spec=IEntityResolver)
    resolver.resolve_entities.return_value = ["Albert Einstein", "Theory of Relativity"]
    resolver.get_vocabulary.return_value = {"Albert Einstein", "Theory of Relativity"}
    return resolver


@pytest.fixture
def mock_ranking_strategy():
    """Mock ranking strategy for testing."""
    strategy = Mock(spec=IRankingStrategy)
    strategy.rank_communities.return_value = [(0, 1.0)]
    strategy.rank_triplets.return_value = [("test_key", "test detail")]
    return strategy


@pytest.fixture
def mock_citation_builder():
    """Mock citation builder for testing."""
    builder = Mock(spec=ICitationBuilder)
    builder.build_citations.return_value = (
        ["- test detail [1]"],  # context_blocks
        [{"id": 1, "snippet": "test snippet"}]  # citations
    )
    builder.format_citation.return_value = "[1] Test: test snippet"
    return builder


@pytest.fixture
def mock_graph_store():
    """Mock graph store for testing."""
    store = Mock()
    store.get_triplets.return_value = []
    store.has_graph_data.return_value = True
    store.supports_vector_queries.return_value = False
    store.get_community_summaries.return_value = {0: "Test summary"}
    store.entity_info = {"Albert Einstein": [0]}
    store.community_info = {0: [{"detail": "test", "triplet_key": "test_key"}]}
    store.triplet_provenance = {"test_key": [{"source_snippet": "test"}]}
    return store


@pytest.fixture
def mock_property_graph_index():
    """Mock property graph index for testing."""
    index = Mock(spec=PropertyGraphIndex)
    retriever = Mock()
    retriever.retrieve.return_value = []
    index.as_retriever.return_value = retriever
    return index


@pytest.fixture
def temp_cache_file(tmp_path):
    """Temporary cache file for testing."""
    return tmp_path / "test_cache.json"


@pytest.fixture
def temp_document_file(tmp_path):
    """Temporary document file for testing."""
    doc_file = tmp_path / "test_document.pdf"
    doc_file.write_text("Test document content")
    return str(doc_file)
