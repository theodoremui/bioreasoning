"""
Graph Processing Package

This package provides a comprehensive graph-based knowledge extraction and querying system
following SOLID principles and best practices for maintainability and extensibility.

The package is organized into the following modules:

Core Interfaces:
    - interfaces: Abstract base classes defining contracts for all components

Storage Layer:
    - stores: Graph storage implementations with community building and caching
    - cache: Caching strategies and persistence management

Processing Layer:
    - extractors: Knowledge extraction from text documents
    - processors: Document processing and cleaning utilities

Query Layer:
    - engines: Query processing with citation and provenance support
    - ranking: Ranking algorithms for communities and triplets

Configuration:
    - config: Configuration management and validation
    - constants: Global constants and templates

Utilities:
    - utils: Common utilities and helper functions

Author: Theodore Mui
Date: 2025-08-24
"""

from .config import GraphConfig
from .engines import GraphRAGQueryEngine
from .extractors import GraphRAGExtractor
from .processors import DocumentProcessor
from .stores import GraphRAGStore

__version__ = "1.0.0"

__all__ = [
    "GraphConfig",
    "GraphRAGStore",
    "GraphRAGExtractor",
    "GraphRAGQueryEngine",
    "DocumentProcessor",
]
