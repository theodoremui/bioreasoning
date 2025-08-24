"""
Entity Resolution

This module provides entity resolution functionality for query processing,
following the Single Responsibility Principle and Strategy Pattern.

Author: Theodore Mui
Date: 2025-08-24
"""

import re
from typing import List, Optional, Set

from llama_index.core import PropertyGraphIndex

from .constants import MEDICAL_TERM_MAPPINGS, PATTERNS
from .interfaces import IEntityResolver


class GraphEntityResolver(IEntityResolver):
    """Entity resolver using graph vocabulary and retrieval strategies.

    Resolves entities relevant to queries using multiple strategies:
    1. Retriever-based extraction from indexed triplets
    2. Vocabulary scanning with exact and partial matching
    3. Medical term mapping for domain-specific queries
    """

    def __init__(
        self,
        index: PropertyGraphIndex,
        graph_store,
        medical_mappings: Optional[dict] = None,
    ):
        """Initialize entity resolver with dependencies.

        Args:
            index: Property graph index for retrieval
            graph_store: Graph store for vocabulary access
            medical_mappings: Medical term mappings (optional)
        """
        self.index = index
        self.graph_store = graph_store
        self.medical_mappings = medical_mappings or MEDICAL_TERM_MAPPINGS.copy()
        self._vocabulary_cache = None

    def resolve_entities(self, query: str, similarity_top_k: int) -> List[str]:
        """Resolve entities relevant to the query.

        Uses a multi-strategy approach:
        1. Try retriever-based extraction first
        2. Fall back to vocabulary scanning with enhanced matching
        3. Apply medical term mappings for better recall

        Args:
            query: Query string
            similarity_top_k: Number of similar items to retrieve

        Returns:
            List of relevant entity names
        """
        # Strategy 1: Retriever-based extraction
        entities = self._resolve_via_retriever(query, similarity_top_k)
        if entities:
            return entities

        # Strategy 2: Enhanced vocabulary scanning
        return self._resolve_via_vocabulary(query)

    def get_vocabulary(self) -> Set[str]:
        """Get the available entity vocabulary from the graph.

        Returns:
            Set of entity names available in the graph
        """
        if self._vocabulary_cache is None:
            self._vocabulary_cache = self._build_vocabulary()
        return self._vocabulary_cache

    def _resolve_via_retriever(self, query: str, similarity_top_k: int) -> List[str]:
        """Resolve entities using the graph index retriever.

        Args:
            query: Query string
            similarity_top_k: Number of similar items to retrieve

        Returns:
            List of entity names from retrieval
        """
        try:
            nodes_retrieved = self.index.as_retriever(
                similarity_top_k=similarity_top_k
            ).retrieve(query)
        except Exception:
            return []

        entities = set()
        pattern = PATTERNS["triplet_pattern"]

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                subject, _, obj = match[0], match[1], match[2]
                entities.add(subject)
                entities.add(obj)

        return list(entities)

    def _resolve_via_vocabulary(self, query: str) -> List[str]:
        """Resolve entities using vocabulary scanning with enhanced matching.

        Args:
            query: Query string

        Returns:
            List of entity names from vocabulary matching
        """
        vocabulary = self.get_vocabulary()
        if not vocabulary:
            return []

        query_lower = query.lower()
        query_tokens = set(re.findall(PATTERNS["word_tokens"], query_lower))

        fallback_entities = []

        # Exact substring matching
        exact_matches = [name for name in vocabulary if name.lower() in query_lower]
        fallback_entities.extend(exact_matches)

        # Token-based matching for better recall
        for name in vocabulary:
            name_tokens = set(re.findall(PATTERNS["word_tokens"], name.lower()))
            if query_tokens & name_tokens:  # If any tokens overlap
                if name not in fallback_entities:
                    fallback_entities.append(name)

        # Medical term mapping
        medical_entities = self._resolve_medical_terms(query_lower, vocabulary)
        for entity in medical_entities:
            if entity not in fallback_entities:
                fallback_entities.append(entity)

        return fallback_entities

    def _resolve_medical_terms(
        self, query_lower: str, vocabulary: Set[str]
    ) -> List[str]:
        """Resolve medical terms using domain-specific mappings.

        Args:
            query_lower: Lowercase query string
            vocabulary: Available entity vocabulary

        Returns:
            List of medical entity names
        """
        medical_entities = []

        for query_term, entity_variants in self.medical_mappings.items():
            if query_term in query_lower:
                for variant in entity_variants:
                    if variant in vocabulary:
                        medical_entities.append(variant)

        return medical_entities

    def _build_vocabulary(self) -> Set[str]:
        """Build vocabulary from graph triplets.

        Returns:
            Set of entity names from the graph
        """
        vocabulary = set()

        try:
            triplets = self.graph_store.get_triplets()
        except Exception:
            return vocabulary

        for entity1, _, entity2 in triplets:
            try:
                vocabulary.add(str(entity1.name))
            except (AttributeError, TypeError):
                pass
            try:
                vocabulary.add(str(entity2.name))
            except (AttributeError, TypeError):
                pass

        return vocabulary

    def add_medical_mapping(self, term: str, variants: List[str]) -> None:
        """Add medical term mapping.

        Args:
            term: Base medical term
            variants: List of entity variants
        """
        self.medical_mappings[term.lower()] = variants

    def clear_vocabulary_cache(self) -> None:
        """Clear the vocabulary cache to force rebuild."""
        self._vocabulary_cache = None

    def get_resolution_stats(self, query: str, similarity_top_k: int) -> dict:
        """Get statistics about entity resolution for a query.

        Args:
            query: Query string
            similarity_top_k: Number of similar items to retrieve

        Returns:
            Dictionary with resolution statistics
        """
        retriever_entities = self._resolve_via_retriever(query, similarity_top_k)
        vocabulary_entities = self._resolve_via_vocabulary(query)

        return {
            "query": query,
            "retriever_entities_count": len(retriever_entities),
            "vocabulary_entities_count": len(vocabulary_entities),
            "total_vocabulary_size": len(self.get_vocabulary()),
            "medical_mappings_count": len(self.medical_mappings),
            "used_retriever": len(retriever_entities) > 0,
        }
