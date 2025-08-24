"""
Configuration Management

This module provides configuration management for the graph processing system,
following the Single Responsibility Principle and providing validation and type safety.

Author: Theodore Mui
Date: 2025-08-24
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .interfaces import IConfiguration


@dataclass
class PerformanceConfig:
    """Performance-related configuration settings."""

    kg_extraction_workers: int = 8
    pdf_parse_workers: int = 8
    max_paths_per_chunk: int = 2
    max_cluster_size: int = 5

    def __post_init__(self):
        """Validate performance configuration values."""
        if self.kg_extraction_workers < 1:
            raise ValueError("kg_extraction_workers must be >= 1")
        if self.pdf_parse_workers < 1:
            raise ValueError("pdf_parse_workers must be >= 1")
        if self.max_paths_per_chunk < 1:
            raise ValueError("max_paths_per_chunk must be >= 1")
        if self.max_cluster_size < 1:
            raise ValueError("max_cluster_size must be >= 1")


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    neo4j_username: str = "neo4j"
    neo4j_password: str = "Salesforce1"
    neo4j_url: str = "bolt://localhost:7687"

    def __post_init__(self):
        """Validate database configuration."""
        if not self.neo4j_username:
            raise ValueError("neo4j_username cannot be empty")
        if not self.neo4j_password:
            raise ValueError("neo4j_password cannot be empty")
        if not self.neo4j_url:
            raise ValueError("neo4j_url cannot be empty")


@dataclass
class APIConfig:
    """API configuration settings."""

    llamacloud_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"

    def __post_init__(self):
        """Validate API configuration."""
        if not self.openai_model:
            raise ValueError("openai_model cannot be empty")


@dataclass
class QueryConfig:
    """Query engine configuration settings."""

    similarity_top_k: int = 20
    max_summaries_to_use: int = 6
    max_triplets_to_use: int = 20

    def __post_init__(self):
        """Validate query configuration."""
        if self.similarity_top_k < 1:
            raise ValueError("similarity_top_k must be >= 1")
        if self.max_summaries_to_use < 1:
            raise ValueError("max_summaries_to_use must be >= 1")
        if self.max_triplets_to_use < 1:
            raise ValueError("max_triplets_to_use must be >= 1")


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    communities_cache_file: str = "data/nccn_communities.json"
    validate_cache_signature: bool = True
    prefer_cache_when_graph_empty: bool = True

    def __post_init__(self):
        """Validate cache configuration."""
        if not self.communities_cache_file:
            raise ValueError("communities_cache_file cannot be empty")


class GraphConfig(IConfiguration):
    """Main configuration class for the graph processing system.

    Provides centralized configuration management with validation and environment
    variable support. Follows the Single Responsibility Principle by focusing
    solely on configuration concerns.

    Example:
        config = GraphConfig.from_environment()
        workers = config.performance.kg_extraction_workers
        db_url = config.database.neo4j_url
    """

    def __init__(
        self,
        performance: Optional[PerformanceConfig] = None,
        database: Optional[DatabaseConfig] = None,
        api: Optional[APIConfig] = None,
        query: Optional[QueryConfig] = None,
        cache: Optional[CacheConfig] = None,
    ):
        """Initialize configuration with optional overrides.

        Args:
            performance: Performance configuration
            database: Database configuration
            api: API configuration
            query: Query configuration
            cache: Cache configuration
        """
        self.performance = performance or PerformanceConfig()
        self.database = database or DatabaseConfig()
        self.api = api or APIConfig()
        self.query = query or QueryConfig()
        self.cache = cache or CacheConfig()

    @classmethod
    def from_environment(cls) -> "GraphConfig":
        """Create configuration from environment variables.

        Loads configuration values from environment variables with sensible defaults.

        Returns:
            GraphConfig instance populated from environment
        """
        performance = PerformanceConfig(
            kg_extraction_workers=int(os.getenv("KG_EXTRACTION_WORKERS", "8")),
            pdf_parse_workers=int(os.getenv("PDF_PARSE_WORKERS", "8")),
            max_paths_per_chunk=int(os.getenv("MAX_PATHS_PER_CHUNK", "2")),
            max_cluster_size=int(os.getenv("MAX_CLUSTER_SIZE", "5")),
        )

        database = DatabaseConfig(
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "Salesforce1"),
            neo4j_url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        )

        api = APIConfig(
            llamacloud_api_key=os.getenv("LLAMACLOUD_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        )

        query = QueryConfig(
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "20")),
            max_summaries_to_use=int(os.getenv("MAX_SUMMARIES_TO_USE", "6")),
            max_triplets_to_use=int(os.getenv("MAX_TRIPLETS_TO_USE", "20")),
        )

        cache = CacheConfig(
            communities_cache_file=os.getenv(
                "COMMUNITIES_CACHE_FILE", "data/nccn_communities.json"
            ),
            validate_cache_signature=os.getenv(
                "VALIDATE_CACHE_SIGNATURE", "true"
            ).lower()
            == "true",
            prefer_cache_when_graph_empty=os.getenv(
                "PREFER_CACHE_WHEN_GRAPH_EMPTY", "true"
            ).lower()
            == "true",
        )

        return cls(performance, database, api, query, cache)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'performance.kg_extraction_workers')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            workers = config.get('performance.kg_extraction_workers', 4)
        """
        parts = key.split(".")
        obj = self

        try:
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return default

    def validate(self) -> bool:
        """Validate all configuration sections.

        Returns:
            True if all configuration is valid

        Raises:
            ValueError: If any configuration is invalid
        """
        try:
            # Validation is performed in __post_init__ of each config class
            # Re-create to trigger validation
            PerformanceConfig(**self.performance.__dict__)
            DatabaseConfig(**self.database.__dict__)
            APIConfig(**self.api.__dict__)
            QueryConfig(**self.query.__dict__)
            CacheConfig(**self.cache.__dict__)
            return True
        except ValueError:
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "performance": self.performance.__dict__,
            "database": self.database.__dict__,
            "api": self.api.__dict__,
            "query": self.query.__dict__,
            "cache": self.cache.__dict__,
        }

    def display_summary(self) -> str:
        """Generate a human-readable configuration summary.

        Returns:
            Formatted configuration summary
        """
        lines = [
            "🚀 Graph Processing Configuration:",
            f"   • KG Extraction Workers: {self.performance.kg_extraction_workers}",
            f"   • PDF Parse Workers: {self.performance.pdf_parse_workers}",
            f"   • Max Paths per Chunk: {self.performance.max_paths_per_chunk}",
            f"   • Max Cluster Size: {self.performance.max_cluster_size}",
            f"   • Neo4j URL: {self.database.neo4j_url}",
            f"   • OpenAI Model: {self.api.openai_model}",
            f"   • Similarity Top K: {self.query.similarity_top_k}",
            f"   • Cache File: {self.cache.communities_cache_file}",
        ]
        return "\n".join(lines)
