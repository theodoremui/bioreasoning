"""
Graph Storage

This module provides graph storage implementations with community building,
caching, and provenance support. Follows SOLID principles with clear
separation of concerns and dependency injection.

Author: Theodore Mui
Date: 2025-08-24
"""

import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from bioagents.utils.text_utils import make_contextual_snippet

from .cache import CommunityCache
from .communities import CommunityBuilderFactory
from .interfaces import (
    ICommunityBuilder,
    ICommunityCache,
    IGraphStore,
    IProvenanceBuilder,
    ISummaryGenerator,
)
from .provenance import ProvenanceBuilder
from .summaries import SummaryGeneratorFactory

try:
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
except ImportError:
    # Lightweight fallback for testing environments
    class Neo4jPropertyGraphStore:
        def __init__(self, *args, **kwargs):
            pass

        def get_triplets(self):
            return []


class GraphRAGStore(Neo4jPropertyGraphStore, IGraphStore):
    """Enhanced property graph store with community building and caching.

    This class extends Neo4jPropertyGraphStore with advanced features while
    following SOLID principles:

    - Single Responsibility: Each component handles one concern
    - Open/Closed: Extensible through dependency injection
    - Liskov Substitution: Implements IGraphStore interface
    - Interface Segregation: Uses focused interfaces
    - Dependency Inversion: Depends on abstractions, not concretions

    Features:
        - Community detection and building
        - LLM-based community summaries
        - Persistent caching with validation
        - Provenance tracking and backfilling
        - Configurable algorithms and strategies

    Example:
        store = GraphRAGStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            max_cluster_size=5
        )
        store.ensure_communities("cache.json")
        summaries = store.get_community_summaries()
    """

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "neo4j",
        max_cluster_size: int = 5,
        community_builder: Optional[ICommunityBuilder] = None,
        cache_manager: Optional[ICommunityCache] = None,
        summary_generator: Optional[ISummaryGenerator] = None,
        provenance_builder: Optional[IProvenanceBuilder] = None,
        lazy_connection: bool = True,  # New parameter for lazy connection
    ):
        """Initialize the graph store with dependency injection.

        Args:
            username: Neo4j username
            password: Neo4j password
            url: Neo4j connection URL
            database: Neo4j database name
            max_cluster_size: Maximum cluster size for community detection
            community_builder: Community building strategy (optional)
            cache_manager: Cache management strategy (optional)
            summary_generator: Summary generation strategy (optional)
            provenance_builder: Provenance building strategy (optional)
            lazy_connection: If True, defer Neo4j connection until needed
        """
        # Store connection parameters for lazy initialization
        self._neo4j_username = username
        self._neo4j_password = password
        self._neo4j_url = url
        self._neo4j_database = database
        self._lazy_connection = lazy_connection
        self._neo4j_store: Optional[Neo4jPropertyGraphStore] = None

        if not lazy_connection:
            # Initialize base Neo4j store immediately if not lazy
            super().__init__(
                username=username, password=password, url=url, database=database
            )

        # Configuration
        self.max_cluster_size = max_cluster_size

        # Dependency injection with defaults
        self._community_builder = (
            community_builder
            or CommunityBuilderFactory.create_builder(
                algorithm="auto", max_cluster_size=max_cluster_size
            )
        )
        self._cache_manager = cache_manager or CommunityCache()
        self._summary_generator = (
            summary_generator or SummaryGeneratorFactory.create_openai_generator()
        )
        self._provenance_builder = provenance_builder or ProvenanceBuilder()

        # State management
        self._community_summary: Dict[int, str] = {}
        self._entity_info: Optional[Dict[str, List[int]]] = None
        self._cluster_assignments: Optional[Dict[str, int]] = None
        self._community_info: Optional[Dict[int, List[Dict[str, Any]]]] = None
        self._algorithm_metadata: Optional[Dict[str, Any]] = None
        self._triplet_provenance: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def _ensure_neo4j_connection(self) -> None:
        """Ensure Neo4j connection is established (lazy initialization)."""
        if self._lazy_connection and self._neo4j_store is None:
            print("🔌 Establishing Neo4j connection...")
            # Initialize the Neo4j store now
            super(GraphRAGStore, self).__init__(
                username=self._neo4j_username,
                password=self._neo4j_password,
                url=self._neo4j_url,
                database=self._neo4j_database,
            )
            self._neo4j_store = self  # Mark as initialized

    @property
    def supports_vector_queries(self) -> bool:
        """Return whether this store supports vector queries."""
        return False

    @property
    def community_summary(self) -> Dict[int, str]:
        """Get community summaries."""
        return self._community_summary.copy()

    @property
    def entity_info(self) -> Optional[Dict[str, List[int]]]:
        """Get entity to community mapping."""
        return self._entity_info.copy() if self._entity_info else None

    @property
    def community_info(self) -> Optional[Dict[int, List[Dict[str, Any]]]]:
        """Get community relationship details."""
        return self._community_info.copy() if self._community_info else None

    @property
    def triplet_provenance(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Get triplet provenance data."""
        return self._triplet_provenance.copy() if self._triplet_provenance else None

    @property
    def algorithm_metadata(self) -> Optional[Dict[str, Any]]:
        """Get algorithm metadata."""
        return self._algorithm_metadata.copy() if self._algorithm_metadata else None

    def build_communities(self, skip_summaries: bool = False) -> None:
        """Build communities from the graph using the configured strategy.

        This method coordinates the community building process:
        1. Creates NetworkX graph from triplets
        2. Runs community detection algorithm
        3. Generates summaries for communities (optional)
        4. Stores all results in instance state

        Args:
            skip_summaries: If True, skip expensive LLM-based summary generation
        """
        # Create NetworkX graph
        nx_graph = self._create_nx_graph()

        # Build communities using injected strategy
        entity_info, community_info = self._community_builder.build_communities(
            nx_graph
        )

        # Store results
        self._entity_info = entity_info
        self._community_info = community_info
        self._algorithm_metadata = self._community_builder.get_algorithm_metadata()

        # Generate cluster assignments for reproducibility
        self._cluster_assignments = self._extract_cluster_assignments()

        # Generate summaries only if not skipped
        if not skip_summaries:
            self._generate_community_summaries()
        else:
            # Initialize empty summaries that can be populated lazily later
            self._community_summary = (
                {
                    cid: f"Community {cid} (summary not generated)"
                    for cid in self._community_info.keys()
                }
                if self._community_info
                else {}
            )

    def _create_nx_graph(self) -> nx.Graph:
        """Create NetworkX graph from triplets.

        Returns:
            NetworkX graph with nodes as entity names and edges with metadata
        """
        # Ensure Neo4j connection is established
        self._ensure_neo4j_connection()

        nx_graph = nx.Graph()
        triplets = self.get_triplets()

        for entity1, relation, entity2 in triplets:
            # Add nodes
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)

            # Create triplet key for tracking
            triplet_key = f"{relation.source_id}|{relation.label}|{relation.target_id}"

            # Add edge with metadata
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties.get("relationship_description", ""),
                triplet_key=triplet_key,
            )

        return nx_graph

    def _extract_cluster_assignments(self) -> Dict[str, int]:
        """Extract cluster assignments for reproducibility.

        Returns:
            Dictionary mapping node names to cluster IDs
        """
        assignments = {}
        if self._entity_info:
            for entity, cluster_ids in self._entity_info.items():
                if cluster_ids:
                    # Use first cluster ID if entity belongs to multiple clusters
                    assignments[str(entity)] = int(cluster_ids[0])
        return assignments

    def _generate_community_summaries(
        self, community_ids: Optional[List[int]] = None
    ) -> None:
        """Generate LLM-based summaries for communities.

        Args:
            community_ids: Specific community IDs to generate summaries for.
                          If None, generates for all communities.
        """
        if not self._community_info:
            return

        # Initialize community_summary if it doesn't exist
        if not hasattr(self, "_community_summary") or self._community_summary is None:
            self._community_summary = {}

        # Determine which communities to process
        target_communities = (
            community_ids
            if community_ids is not None
            else list(self._community_info.keys())
        )

        for community_id in target_communities:
            if community_id not in self._community_info:
                continue

            details = self._community_info[community_id]
            try:
                # Extract text from details
                if details and isinstance(details[0], dict):
                    text_lines = [
                        d.get("detail", "") for d in details if d.get("detail")
                    ]
                else:
                    text_lines = [str(d) for d in details if str(d).strip()]

                if text_lines:
                    details_text = "\n".join(text_lines) + "."
                    summary = self._summary_generator.generate_summary(details_text)
                    self._community_summary[community_id] = summary

            except Exception as e:
                # Provide fallback summary on error
                self._community_summary[community_id] = (
                    f"Summary generation failed: {str(e)}"
                )

    def get_community_summaries(
        self, generate_if_missing: bool = False
    ) -> Dict[int, str]:
        """Get community summaries, building them if necessary.

        Args:
            generate_if_missing: If True, generate summaries for communities that don't have them

        Returns:
            Dictionary mapping community ID to summary text
        """
        if not self._community_summary:
            self.build_communities(skip_summaries=not generate_if_missing)
        elif generate_if_missing and self._community_info:
            # Check if we have placeholder summaries that need to be generated
            placeholder_communities = [
                cid
                for cid, summary in self._community_summary.items()
                if "summary not generated" in summary
            ]

            if placeholder_communities:
                print(
                    f"🔄 Generating summaries for {len(placeholder_communities)} communities..."
                )
                self._generate_community_summaries(
                    community_ids=placeholder_communities
                )

        return self._community_summary.copy()

    def ensure_summaries_generated(self) -> None:
        """Ensure all community summaries are generated (lazy loading).

        This method can be called when summaries are actually needed,
        allowing for fast initialization followed by on-demand summary generation.
        """
        if not self._community_summary or not self._community_info:
            return

        # Find communities that need summaries
        missing_summaries = []
        for cid in self._community_info.keys():
            if (
                cid not in self._community_summary
                or "summary not generated" in self._community_summary.get(cid, "")
            ):
                missing_summaries.append(cid)

        if missing_summaries:
            print(f"🔄 Generating {len(missing_summaries)} community summaries...")
            self._generate_community_summaries(community_ids=missing_summaries)

    def has_graph_data(self) -> bool:
        """Check if the graph store contains data.

        Returns:
            True if triplets are available, False otherwise
        """
        try:
            # Ensure Neo4j connection is established
            self._ensure_neo4j_connection()
            triplets = self.get_triplets()
            return bool(triplets)
        except Exception:
            return False

    def ensure_communities(
        self,
        persist_path: Optional[str] = None,
        validate_signature: bool = False,  # Changed default to False for faster init
        prefer_cache_when_graph_empty: bool = True,
        force_rebuild: bool = False,
        skip_summaries: bool = False,  # New option to skip expensive LLM calls
    ) -> None:
        """Ensure communities are available, using cache when possible.

        Args:
            persist_path: Path to cache file (optional)
            validate_signature: Whether to validate cache against current graph
            prefer_cache_when_graph_empty: Use cache even if graph is empty
            force_rebuild: Force rebuilding communities even if cache exists
            skip_summaries: Skip LLM-based summary generation for faster initialization
        """
        if persist_path and not force_rebuild:
            loaded = self._try_load_from_cache(
                persist_path, validate_signature, prefer_cache_when_graph_empty
            )
            if loaded:
                # Auto-backfill provenance if missing (but don't save to avoid slow I/O)
                if not self._triplet_provenance and self.has_graph_data():
                    self.build_triplet_provenance_from_graph()
                    # Only save if explicitly requested
                    if validate_signature:
                        self._try_save_to_cache(persist_path)
                return

        # Build communities if cache loading failed
        self.build_communities(skip_summaries=skip_summaries)

        # Save to cache if path provided
        if persist_path:
            self._try_save_to_cache(persist_path)

    def _try_load_from_cache(
        self,
        persist_path: str,
        validate_signature: bool,
        prefer_cache_when_graph_empty: bool,
    ) -> bool:
        """Try to load communities from cache.

        Args:
            persist_path: Path to cache file
            validate_signature: Whether to validate cache signature
            prefer_cache_when_graph_empty: Use cache even if graph is empty

        Returns:
            True if successfully loaded from cache
        """
        try:
            # Load cache data
            data = self._cache_manager.load_communities(
                persist_path, validate_signature=False
            )
            if not data:
                return False

            # Validate signature if required and graph has data
            if validate_signature and self.has_graph_data():
                current_signature = self._compute_graph_signature()
                cached_signature = data.get("graph_signature")
                if cached_signature != current_signature:
                    return False
            elif not prefer_cache_when_graph_empty and not self.has_graph_data():
                return False

            # Load data into instance state
            self._load_cache_data(data)
            return True

        except Exception:
            return False

    def _try_save_to_cache(self, persist_path: str) -> None:
        """Try to save communities to cache.

        Args:
            persist_path: Path to cache file
        """
        try:
            data = self._prepare_cache_data()
            self._cache_manager.save_communities(persist_path, data)
        except Exception:
            pass  # Silent failure for cache operations

    def _load_cache_data(self, data: Dict[str, Any]) -> None:
        """Load cache data into instance state.

        Args:
            data: Cache data dictionary
        """
        self._entity_info = data.get("entity_info", {})
        self._community_summary = data.get("community_summary", {})
        self._cluster_assignments = data.get("cluster_assignments", {})
        self._community_info = data.get("community_info", {})
        self._triplet_provenance = data.get("triplet_provenance", {})
        self._algorithm_metadata = data.get("algorithm_metadata")

    def _prepare_cache_data(self) -> Dict[str, Any]:
        """Prepare data for caching.

        Returns:
            Dictionary with all cache data
        """
        return {
            "graph_signature": self._compute_graph_signature(),
            "max_cluster_size": self.max_cluster_size,
            "entity_info": self._entity_info or {},
            "community_summary": self._community_summary or {},
            "cluster_assignments": self._cluster_assignments or {},
            "community_info": self._community_info or {},
            "triplet_provenance": self._triplet_provenance or {},
            "algorithm_metadata": self._algorithm_metadata
            or {
                "algorithm": "unknown",
                "parameters": {"max_cluster_size": self.max_cluster_size},
            },
        }

    def _compute_graph_signature(self) -> str:
        """Compute signature for cache validation.

        Returns:
            SHA256 hash of current graph state
        """
        try:
            # Ensure Neo4j connection is established
            self._ensure_neo4j_connection()
            triplets = self.get_triplets()
            return self._cache_manager.compute_signature(triplets)
        except Exception:
            return ""

    def build_triplet_provenance_from_graph(self) -> None:
        """Build provenance information from current graph triplets.

        Uses the injected provenance builder to extract provenance data
        from relation properties and store it for citation purposes.
        """
        try:
            # Ensure Neo4j connection is established
            self._ensure_neo4j_connection()
            triplets = self.get_triplets()
            self._triplet_provenance = (
                self._provenance_builder.build_triplet_provenance(triplets)
            )
        except Exception:
            self._triplet_provenance = {}

    def get_cache_info(self, persist_path: str) -> Optional[Dict[str, Any]]:
        """Get information about cache file.

        Args:
            persist_path: Path to cache file

        Returns:
            Cache metadata or None if unavailable
        """
        return self._cache_manager.get_cache_metadata(persist_path)

    def validate_cache(self, persist_path: str) -> bool:
        """Validate cache file against current graph.

        Args:
            persist_path: Path to cache file

        Returns:
            True if cache is valid
        """
        current_signature = self._compute_graph_signature()
        return self._cache_manager.validate_cache_integrity(
            persist_path, current_signature
        )
