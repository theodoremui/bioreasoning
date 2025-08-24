"""
Knowledge Extractors

This module provides knowledge extraction functionality from text documents,
following SOLID principles with clear separation of concerns and dependency injection.

Author: Theodore Mui
Date: 2025-08-24
"""

import asyncio
import hashlib
import json
import re
from typing import Any, Callable, List, Optional, Tuple, Union

from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import BaseNode, TransformComponent

from bioagents.utils.text_utils import make_contextual_snippet

from .constants import KG_TRIPLET_EXTRACT_TEMPLATE, PATTERNS
from .interfaces import IKnowledgeExtractor


class ResponseParser:
    """Parser for LLM extraction responses.

    Handles parsing of structured JSON responses from LLM knowledge extraction,
    following SRP by focusing solely on response parsing concerns.
    """

    def __init__(self, json_pattern: str = PATTERNS["json_extraction"]):
        """Initialize parser with JSON extraction pattern.

        Args:
            json_pattern: Regular expression pattern for JSON extraction
        """
        self.json_pattern = json_pattern

    def parse_extraction_response(
        self, response_str: str
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Parse LLM response into entities and relationships.

        Args:
            response_str: Raw LLM response string

        Returns:
            Tuple of (entities, relationships) as lists of tuples
        """
        entities = []
        relationships = []

        # Extract JSON from response
        match = re.search(self.json_pattern, response_str, re.DOTALL)
        if not match:
            return entities, relationships

        json_str = match.group(0)

        try:
            data = json.loads(json_str)

            # Parse entities
            entities = [
                (
                    entity["entity_name"],
                    entity["entity_type"],
                    entity["entity_description"],
                )
                for entity in data.get("entities", [])
                if self._validate_entity(entity)
            ]

            # Parse relationships
            relationships = [
                (
                    relation["source_entity"],
                    relation["target_entity"],
                    relation["relation"],
                    relation["relationship_description"],
                )
                for relation in data.get("relationships", [])
                if self._validate_relationship(relation)
            ]

        except json.JSONDecodeError as e:
            # Log error but return empty results rather than raising
            print(f"Error parsing JSON: {e}")

        return entities, relationships

    def _validate_entity(self, entity: dict) -> bool:
        """Validate entity structure.

        Args:
            entity: Entity dictionary

        Returns:
            True if entity is valid
        """
        required_fields = ["entity_name", "entity_type", "entity_description"]
        return all(
            field in entity and entity[field] and str(entity[field]).strip()
            for field in required_fields
        )

    def _validate_relationship(self, relationship: dict) -> bool:
        """Validate relationship structure.

        Args:
            relationship: Relationship dictionary

        Returns:
            True if relationship is valid
        """
        required_fields = [
            "source_entity",
            "target_entity",
            "relation",
            "relationship_description",
        ]
        return all(
            field in relationship
            and relationship[field]
            and str(relationship[field]).strip()
            for field in required_fields
        )


class ProvenanceEnricher:
    """Enriches extracted knowledge with provenance information.

    Handles the addition of provenance metadata to extracted entities and relations,
    following SRP by focusing solely on provenance enrichment.
    """

    def __init__(self, snippet_max_length: int = 500):
        """Initialize provenance enricher.

        Args:
            snippet_max_length: Maximum length for contextual snippets
        """
        self.snippet_max_length = snippet_max_length

    def enrich_entities(
        self, entities: List[Tuple], node: BaseNode
    ) -> List[EntityNode]:
        """Enrich entities with provenance metadata.

        Args:
            entities: List of (name, type, description) tuples
            node: Source text node

        Returns:
            List of EntityNode objects with provenance
        """
        entity_nodes = []
        entity_metadata = node.metadata.copy()

        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata.copy()
            )
            entity_nodes.append(entity_node)

        return entity_nodes

    def enrich_relations(
        self, relationships: List[Tuple], node: BaseNode
    ) -> List[Relation]:
        """Enrich relationships with provenance metadata.

        Args:
            relationships: List of (subject, object, relation, description) tuples
            node: Source text node

        Returns:
            List of Relation objects with provenance
        """
        relation_nodes = []
        relation_metadata_base = node.metadata.copy()

        # Extract base provenance information
        base_prov = self._extract_base_provenance(node, relation_metadata_base)

        # Create contextual snippet
        snippet = self._create_snippet(node)

        for subject, obj, relation, description in relationships:
            relation_metadata = relation_metadata_base.copy()
            relation_metadata["relationship_description"] = description

            # Create triplet key
            triplet_key = f"{subject}|{relation}|{obj}"

            # Generate provenance ID
            prov_id = self._generate_provenance_id(base_prov, triplet_key)

            # Add provenance metadata
            relation_metadata.update(
                {
                    "triplet_key": triplet_key,
                    "source_doc_id": base_prov["source_doc_id"],
                    "source_doc_title": base_prov["source_doc_title"],
                    "source_file_path": base_prov["source_file_path"],
                    "source_page": base_prov["source_page"],
                    "source_paragraph_index": base_prov["source_paragraph_index"],
                    "char_start": base_prov["char_start"],
                    "char_end": base_prov["char_end"],
                    "extraction_node_id": base_prov["extraction_node_id"],
                    "source_snippet": snippet,
                    "provenance_id": prov_id,
                }
            )

            rel_node = Relation(
                label=relation,
                source_id=subject,
                target_id=obj,
                properties=relation_metadata,
            )

            relation_nodes.append(rel_node)

        return relation_nodes

    def _extract_base_provenance(self, node: BaseNode, metadata: dict) -> dict:
        """Extract base provenance information from node.

        Args:
            node: Source text node
            metadata: Node metadata

        Returns:
            Dictionary with base provenance information
        """
        return {
            "source_doc_id": metadata.get("doc_id"),
            "source_doc_title": metadata.get("doc_title"),
            "source_file_path": metadata.get("file_path"),
            "source_page": metadata.get("page_number"),
            "source_paragraph_index": metadata.get("paragraph_index"),
            "char_start": metadata.get("char_start"),
            "char_end": metadata.get("char_end"),
            "extraction_node_id": getattr(node, "node_id", None),
        }

    def _create_snippet(self, node: BaseNode) -> str:
        """Create contextual snippet from node content.

        Args:
            node: Source text node

        Returns:
            Contextual snippet text
        """
        try:
            full_text = node.get_content(metadata_mode="llm") or ""
        except Exception:
            full_text = ""

        snippet = make_contextual_snippet(
            full_text, "", max_length=self.snippet_max_length
        )

        # Fallback to truncated full text if snippet is empty
        if not snippet and full_text:
            snippet = full_text[: self.snippet_max_length]

        return snippet

    def _generate_provenance_id(self, base_prov: dict, triplet_key: str) -> str:
        """Generate unique provenance ID.

        Args:
            base_prov: Base provenance information
            triplet_key: Triplet key string

        Returns:
            SHA256 hash as provenance ID
        """
        prov_payload = json.dumps(
            {**base_prov, "triplet_key": triplet_key},
            default=str,
            ensure_ascii=False,
        )
        return hashlib.sha256(prov_payload.encode("utf-8")).hexdigest()


class GraphRAGExtractor(TransformComponent, IKnowledgeExtractor):
    """Knowledge extractor for graph-based RAG systems.

    Extracts entities and relationships from text nodes using LLM-based processing
    with provenance tracking. Follows SOLID principles with dependency injection
    and clear separation of concerns.

    Features:
        - Configurable LLM and prompts
        - Parallel processing with worker pools
        - Provenance tracking and enrichment
        - Robust error handling
        - Extensible parsing strategies

    Example:
        extractor = GraphRAGExtractor(
            llm=my_llm,
            extract_prompt=custom_prompt,
            max_paths_per_chunk=5,
            num_workers=8
        )
        enriched_nodes = await extractor.acall(text_nodes)
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Optional[Callable] = None,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
        response_parser: Optional[ResponseParser] = None,
        provenance_enricher: Optional[ProvenanceEnricher] = None,
    ):
        """Initialize the knowledge extractor.

        Args:
            llm: Language model for extraction (optional, uses Settings.llm)
            extract_prompt: Extraction prompt template (optional, uses default)
            parse_fn: Response parsing function (optional, uses default parser)
            max_paths_per_chunk: Maximum relationships to extract per chunk
            num_workers: Number of parallel workers
            response_parser: Custom response parser (optional)
            provenance_enricher: Custom provenance enricher (optional)
        """
        # Handle prompt conversion
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        # Initialize with dependency injection
        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt
            or PromptTemplate(KG_TRIPLET_EXTRACT_TEMPLATE),
            parse_fn=parse_fn,  # Keep for backward compatibility
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

        # Inject dependencies with defaults
        self._response_parser = response_parser or ResponseParser()
        self._provenance_enricher = provenance_enricher or ProvenanceEnricher()

        # Use injected parser if no parse_fn provided
        if not parse_fn:
            self.parse_fn = self._response_parser.parse_extraction_response

    @classmethod
    def class_name(cls) -> str:
        """Get class name for serialization."""
        return "GraphRAGExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract knowledge from nodes synchronously.

        Args:
            nodes: List of text nodes to process
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments

        Returns:
            List of nodes with extracted knowledge in metadata
        """
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def extract_knowledge(self, node: BaseNode) -> BaseNode:
        """Extract knowledge from a single text node.

        Args:
            node: Input text node

        Returns:
            Node with extracted entities and relationships in metadata
        """
        return await self._aextract(node)

    def parse_extraction_response(
        self, response: str
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Parse LLM response into entities and relationships.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (entities, relationships)
        """
        return self._response_parser.parse_extraction_response(response)

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract knowledge from a node asynchronously.

        Args:
            node: Input text node

        Returns:
            Node with extracted knowledge in metadata
        """
        if not hasattr(node, "text"):
            raise ValueError("Node must have text attribute")

        # Get text content for extraction
        text = node.get_content(metadata_mode="llm")
        if not text or not text.strip():
            return node  # Return unchanged if no text

        # Extract entities and relationships using LLM
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, relationships = self.parse_fn(llm_response)
        except Exception as e:
            # Log error and continue with empty results
            print(f"Extraction failed for node: {e}")
            entities, relationships = [], []

        # Get existing knowledge from metadata
        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        # Enrich entities and relationships with provenance
        if entities:
            entity_nodes = self._provenance_enricher.enrich_entities(entities, node)
            existing_nodes.extend(entity_nodes)

        if relationships:
            relation_nodes = self._provenance_enricher.enrich_relations(
                relationships, node
            )
            existing_relations.extend(relation_nodes)

        # Store enriched knowledge in metadata
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract knowledge from multiple nodes asynchronously.

        Args:
            nodes: List of text nodes to process
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments

        Returns:
            List of nodes with extracted knowledge
        """
        if not nodes:
            return []

        # Create extraction jobs
        jobs = [self._aextract(node) for node in nodes]

        # Run jobs with worker pool
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting knowledge from text",
        )

    def get_extraction_stats(self, nodes: List[BaseNode]) -> dict:
        """Get statistics about extraction results.

        Args:
            nodes: Processed nodes

        Returns:
            Dictionary with extraction statistics
        """
        total_entities = 0
        total_relations = 0
        nodes_with_knowledge = 0

        for node in nodes:
            entities = node.metadata.get(KG_NODES_KEY, [])
            relations = node.metadata.get(KG_RELATIONS_KEY, [])

            total_entities += len(entities)
            total_relations += len(relations)

            if entities or relations:
                nodes_with_knowledge += 1

        return {
            "total_nodes": len(nodes),
            "nodes_with_knowledge": nodes_with_knowledge,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "avg_entities_per_node": total_entities / len(nodes) if nodes else 0,
            "avg_relations_per_node": total_relations / len(nodes) if nodes else 0,
        }
