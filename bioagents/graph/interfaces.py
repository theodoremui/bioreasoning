"""
Graph Processing Interfaces

This module defines abstract base classes and interfaces for the graph processing system,
following the Interface Segregation Principle (ISP) and Dependency Inversion Principle (DIP).

Each interface defines a specific contract, allowing for flexible implementations and easy testing.

Author: Theodore Mui
Date: 2025-08-24
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, Document


class ICommunityBuilder(ABC):
    """Interface for community detection and building algorithms.

    Separates community detection logic from storage concerns.
    """

    @abstractmethod
    def build_communities(
        self, graph
    ) -> Tuple[Dict[str, List[int]], Dict[int, List[Dict[str, Any]]]]:
        """Build communities from a graph.

        Args:
            graph: NetworkX graph or similar graph structure

        Returns:
            Tuple of (entity_info, community_info)
            - entity_info: Maps entity names to list of community IDs
            - community_info: Maps community ID to list of relationship details
        """
        pass

    @abstractmethod
    def get_algorithm_metadata(self) -> Dict[str, Any]:
        """Get metadata about the algorithm used for community detection."""
        pass


class ICommunityCache(ABC):
    """Interface for community data persistence and caching.

    Handles loading and saving of community data with validation.
    """

    @abstractmethod
    def save_communities(self, filepath: str, data: Dict[str, Any]) -> None:
        """Save community data to file."""
        pass

    @abstractmethod
    def load_communities(
        self, filepath: str, validate_signature: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load community data from file with optional validation."""
        pass

    @abstractmethod
    def compute_signature(self, triplets: List[Tuple]) -> str:
        """Compute a signature for cache validation."""
        pass


class IProvenanceBuilder(ABC):
    """Interface for building provenance information from graph data.

    Handles the creation and management of triplet provenance data.
    """

    @abstractmethod
    def build_triplet_provenance(
        self, triplets: List[Tuple]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build provenance information from triplets."""
        pass

    @abstractmethod
    def extract_provenance_from_relation(self, relation: Any) -> Dict[str, Any]:
        """Extract provenance data from a relation object."""
        pass


class ISummaryGenerator(ABC):
    """Interface for generating community summaries.

    Abstracts the LLM-based summary generation process.
    """

    @abstractmethod
    def generate_summary(self, text: str) -> str:
        """Generate a summary for the given text."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt used for summary generation."""
        pass


class IKnowledgeExtractor(ABC):
    """Interface for knowledge extraction from text.

    Defines the contract for extracting entities and relationships from text nodes.
    """

    @abstractmethod
    async def extract_knowledge(self, node: BaseNode) -> BaseNode:
        """Extract knowledge from a text node.

        Args:
            node: Input text node

        Returns:
            Node with extracted entities and relationships in metadata
        """
        pass

    @abstractmethod
    def parse_extraction_response(
        self, response: str
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Parse LLM response into entities and relationships."""
        pass


class IEntityResolver(ABC):
    """Interface for resolving entities from queries.

    Handles entity identification and resolution strategies.
    """

    @abstractmethod
    def resolve_entities(self, query: str, similarity_top_k: int) -> List[str]:
        """Resolve entities relevant to the query."""
        pass

    @abstractmethod
    def get_vocabulary(self) -> set:
        """Get the available entity vocabulary."""
        pass


class IRankingStrategy(ABC):
    """Interface for ranking strategies.

    Defines contracts for ranking communities and triplets.
    """

    @abstractmethod
    def rank_communities(
        self, community_summaries: Dict[int, str], query: str, candidate_ids: List[int]
    ) -> List[Tuple[int, float]]:
        """Rank communities by relevance to query."""
        pass

    @abstractmethod
    def rank_triplets(
        self, triplets: List[Tuple[str, str]], query: str
    ) -> List[Tuple[str, str]]:
        """Rank triplets by relevance to query."""
        pass


class ICitationBuilder(ABC):
    """Interface for building citations from provenance data.

    Handles the creation of citation information for query responses.
    """

    @abstractmethod
    def build_citations(
        self,
        triplets: List[Tuple[str, str]],
        provenance_data: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Build citation context and metadata."""
        pass

    @abstractmethod
    def format_citation(self, citation: Dict[str, Any]) -> str:
        """Format a single citation for display."""
        pass


class IDocumentProcessor(ABC):
    """Interface for document processing operations.

    Handles PDF parsing, cleaning, and node creation.
    """

    @abstractmethod
    async def process_document(self, file_path: str) -> List[BaseNode]:
        """Process a document and return text nodes with provenance."""
        pass

    @abstractmethod
    async def clean_document_content(self, content: Any) -> Any:
        """Clean document content by removing unwanted text."""
        pass


class IQueryEngine(ABC):
    """Interface for query processing engines.

    Defines the contract for processing queries and generating responses.
    """

    @abstractmethod
    def query(self, query_str: str) -> str:
        """Process a query and return a response with citations."""
        pass

    @abstractmethod
    def get_last_citations(self) -> List[Dict[str, Any]]:
        """Get citations from the last query for programmatic access."""
        pass


class IGraphStore(ABC):
    """Interface for graph storage operations.

    Defines the contract for graph data storage and retrieval.
    """

    @abstractmethod
    def get_triplets(self) -> List[Tuple]:
        """Get all triplets from the graph store."""
        pass

    @abstractmethod
    def has_graph_data(self) -> bool:
        """Check if the graph store contains data."""
        pass

    @abstractmethod
    def supports_vector_queries(self) -> bool:
        """Check if the store supports vector queries."""
        pass


class IConfiguration(ABC):
    """Interface for configuration management.

    Handles loading, validation, and access to configuration settings.
    """

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the configuration."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        pass
