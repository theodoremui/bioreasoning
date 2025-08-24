"""
Query Engines

This module provides query processing engines for graph-based RAG systems,
following SOLID principles with clear separation of concerns and dependency injection.

Author: Theodore Mui
Date: 2025-08-24
"""

import re
from typing import Any, Dict, List, Optional

from llama_index.core import PropertyGraphIndex
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.query_engine import CustomQueryEngine

from .citations import EnhancedCitationBuilder
from .constants import (
    ANSWER_AGGREGATION_SYSTEM_PROMPT,
    CITED_CONTEXT_SYSTEM_PROMPT,
    ONCOLOGY_ASSISTANT_SYSTEM_PROMPT,
)
from .entities import GraphEntityResolver
from .interfaces import (
    ICitationBuilder,
    IEntityResolver,
    IQueryEngine,
    IRankingStrategy,
)
from .ranking import RankingStrategyFactory


class GraphRAGQueryEngine(CustomQueryEngine, IQueryEngine):
    """Advanced query engine for graph-based RAG with citations and provenance.

    This query engine follows SOLID principles and uses dependency injection
    to provide flexible, testable, and maintainable query processing.

    Features:
        - Multi-strategy entity resolution
        - Configurable ranking algorithms
        - Citation generation with provenance
        - Fallback strategies for robustness
        - Comprehensive error handling

    Architecture:
        - Entity resolution: Identifies relevant entities from queries
        - Community ranking: Ranks communities by relevance
        - Triplet ranking: Ranks triplets within communities
        - Citation building: Creates citations with provenance
        - Answer generation: Generates responses with citations

    Example:
        engine = GraphRAGQueryEngine(
            graph_store=store,
            llm=llm,
            index=index,
            entity_resolver=custom_resolver,
            ranking_strategy=custom_ranker,
            citation_builder=custom_builder
        )
        response = engine.query("How to treat breast cancer?")
    """

    # Declare all fields as Pydantic fields
    graph_store: Any = Field(description="Graph store for data access")
    llm: LLM = Field(description="Language model for response generation")
    index: PropertyGraphIndex = Field(description="Property graph index for retrieval")
    similarity_top_k: int = Field(
        default=20, description="Number of similar entities to retrieve"
    )
    max_summaries_to_use: int = Field(
        default=6, description="Maximum community summaries to use"
    )
    max_triplets_to_use: int = Field(default=20, description="Maximum triplets to use")

    # Internal fields for dependency injection (excluded from serialization)
    entity_resolver: Optional[IEntityResolver] = Field(default=None, exclude=True)
    ranking_strategy: Optional[IRankingStrategy] = Field(default=None, exclude=True)
    citation_builder: Optional[ICitationBuilder] = Field(default=None, exclude=True)
    last_citations: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)

    def __init__(
        self,
        graph_store,
        llm: LLM,
        index: PropertyGraphIndex,
        similarity_top_k: int = 20,
        max_summaries_to_use: int = 6,
        max_triplets_to_use: int = 20,
        entity_resolver: Optional[IEntityResolver] = None,
        ranking_strategy: Optional[IRankingStrategy] = None,
        citation_builder: Optional[ICitationBuilder] = None,
        **kwargs,
    ):
        """Initialize the query engine with dependency injection.

        Args:
            graph_store: Graph store for data access
            llm: Language model for response generation
            index: Property graph index for retrieval
            similarity_top_k: Number of similar entities to retrieve
            max_summaries_to_use: Maximum community summaries to use
            max_triplets_to_use: Maximum triplets to use
            entity_resolver: Entity resolution strategy (optional)
            ranking_strategy: Ranking strategy (optional)
            citation_builder: Citation building strategy (optional)
        """
        # Initialize parent with all fields
        super().__init__(
            graph_store=graph_store,
            llm=llm,
            index=index,
            similarity_top_k=similarity_top_k,
            max_summaries_to_use=max_summaries_to_use,
            max_triplets_to_use=max_triplets_to_use,
            **kwargs,
        )

        # Dependency injection with defaults
        self.entity_resolver = entity_resolver or GraphEntityResolver(
            index=index, graph_store=graph_store
        )
        self.ranking_strategy = (
            ranking_strategy or RankingStrategyFactory.create_default_ranking()
        )
        self.citation_builder = citation_builder or EnhancedCitationBuilder(
            graph_store=graph_store
        )

        # State for last query
        self.last_citations = []

    def query(self, query_str: str) -> str:
        """Process a query and return a response with citations.

        Args:
            query_str: Query string to process

        Returns:
            Response string with citations section
        """
        return self.custom_query(query_str)

    def custom_query(self, query_str: str) -> str:
        """Main query processing method with comprehensive strategy.

        Processing steps:
        1. Resolve entities relevant to the query
        2. Retrieve communities containing those entities
        3. Rank communities by relevance to query
        4. Extract and rank triplets from top communities
        5. Build citations from triplet provenance
        6. Generate answer using cited context
        7. Append formatted citations section

        Args:
            query_str: Query string to process

        Returns:
            Complete response with citations
        """
        try:
            # Step 1: Resolve entities
            entities = self.entity_resolver.resolve_entities(
                query_str, self.similarity_top_k
            )

            # Step 2: Get communities containing entities
            community_ids = self._retrieve_entity_communities(entities)

            # Step 2.5: Ensure summaries are generated on-demand (lazy loading)
            self.graph_store.ensure_summaries_generated()
            community_summaries = self.graph_store.get_community_summaries()

            # Step 3: Handle fallbacks for empty community resolution
            if not community_ids and community_summaries:
                community_ids = list(community_summaries.keys())

            # Step 4: Normalize community IDs for consistent lookup
            community_ids = self._normalize_community_ids(
                community_ids, community_summaries
            )

            # Step 5: Rank communities by relevance
            ranked_communities = self.ranking_strategy.rank_communities(
                community_summaries, query_str, community_ids
            )
            chosen_community_ids = [
                cid for cid, _ in ranked_communities[: self.max_summaries_to_use]
            ]

            # Step 6: Fallback if ranking produced no results
            if not chosen_community_ids and community_summaries:
                chosen_community_ids = list(community_summaries.keys())[
                    : self.max_summaries_to_use
                ]

            # Step 7: Extract and rank triplets
            candidate_triplets, detail_only_blocks = self._extract_candidate_triplets(
                chosen_community_ids
            )
            ranked_triplets = self.ranking_strategy.rank_triplets(
                candidate_triplets, query_str
            )
            chosen_triplets = ranked_triplets[: self.max_triplets_to_use]

            # Step 8: Generate response with citations
            return self._generate_response_with_citations(
                chosen_triplets,
                detail_only_blocks,
                community_summaries,
                chosen_community_ids,
                query_str,
            )

        except Exception as e:
            # Robust error handling
            return f"Query processing failed: {str(e)}"

    def get_last_citations(self) -> List[Dict[str, Any]]:
        """Get citations from the last query for programmatic access.

        Returns:
            List of citation metadata dictionaries
        """
        return self.last_citations.copy()

    def _retrieve_entity_communities(self, entities: List[str]) -> List[int]:
        """Retrieve community IDs for given entities.

        Args:
            entities: List of entity names

        Returns:
            List of community IDs
        """
        community_ids = []
        entity_info = self.graph_store.entity_info or {}

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))  # Remove duplicates

    def _normalize_community_ids(
        self, community_ids: List[int], community_summaries: Dict[int, str]
    ) -> List[int]:
        """Normalize community IDs for consistent dictionary access.

        Args:
            community_ids: List of community IDs to normalize
            community_summaries: Community summaries dictionary

        Returns:
            List of normalized community IDs
        """
        if not community_summaries:
            return community_ids

        # Determine target type from summaries keys
        sample_key = next(iter(community_summaries.keys()))
        target_type = type(sample_key)

        normalized_ids = []
        for cid in community_ids:
            try:
                if target_type == int:
                    normalized_cid = (
                        int(cid) if isinstance(cid, str) and cid.isdigit() else int(cid)
                    )
                else:
                    normalized_cid = str(cid)
                normalized_ids.append(normalized_cid)
            except (ValueError, TypeError):
                continue

        return normalized_ids

    def _extract_candidate_triplets(self, community_ids: List[int]) -> tuple:
        """Extract candidate triplets from communities.

        Args:
            community_ids: List of community IDs to extract from

        Returns:
            Tuple of (candidate_triplets, detail_only_blocks)
        """
        candidate_triplets = []
        detail_only_blocks = []

        community_info = self.graph_store.community_info
        if not isinstance(community_info, dict):
            return candidate_triplets, detail_only_blocks

        for cid in community_ids:
            # Ensure consistent integer access
            int_cid = int(cid) if isinstance(cid, str) and cid.isdigit() else cid
            items = community_info.get(int_cid, [])

            for item in items:
                if isinstance(item, dict):
                    detail = item.get("detail", "")
                    triplet_key = item.get("triplet_key")

                    if triplet_key:
                        candidate_triplets.append((triplet_key, detail))
                    elif detail:
                        detail_only_blocks.append(f"- {detail}")
                else:
                    detail_only_blocks.append(f"- {str(item)}")

        return candidate_triplets, detail_only_blocks

    def _generate_response_with_citations(
        self,
        chosen_triplets: List[tuple],
        detail_only_blocks: List[str],
        community_summaries: Dict[int, str],
        chosen_community_ids: List[int],
        query_str: str,
    ) -> str:
        """Generate response with citations using multiple fallback strategies.

        Args:
            chosen_triplets: Selected triplets for response
            detail_only_blocks: Detail blocks without triplet keys
            community_summaries: Community summaries dictionary
            chosen_community_ids: Selected community IDs
            query_str: Original query string

        Returns:
            Complete response with citations
        """
        # Strategy 1: Use triplets with citations
        if chosen_triplets:
            return self._generate_cited_response(chosen_triplets, query_str)

        # Strategy 2: Use detail blocks without citations
        elif detail_only_blocks:
            return self._generate_detail_response(detail_only_blocks, query_str)

        # Strategy 3: Use community summaries
        else:
            return self._generate_summary_response(
                community_summaries, chosen_community_ids, query_str
            )

    def _generate_cited_response(self, triplets: List[tuple], query_str: str) -> str:
        """Generate response with citations from triplets.

        Args:
            triplets: List of (triplet_key, detail) tuples
            query_str: Query string

        Returns:
            Response with citations section
        """
        # Build citations
        provenance_data = self.graph_store.triplet_provenance or {}
        context_blocks, citations = self.citation_builder.build_citations(
            triplets, provenance_data
        )

        # Generate answer
        cited_context = "\n".join(context_blocks)
        answer = self._generate_answer_from_cited_context(cited_context, query_str)

        # Format citations section
        citations_section = self.citation_builder.format_citations_section(citations)

        # Store for programmatic access
        self.last_citations = citations

        return answer.rstrip() + citations_section

    def _generate_detail_response(
        self, detail_blocks: List[str], query_str: str
    ) -> str:
        """Generate response from detail blocks without citations.

        Args:
            detail_blocks: List of detail strings
            query_str: Query string

        Returns:
            Response without citations
        """
        detail_context = "\n".join(detail_blocks)
        answer = self._generate_answer_from_cited_context(detail_context, query_str)
        self.last_citations = []
        return answer

    def _generate_summary_response(
        self,
        community_summaries: Dict[int, str],
        community_ids: List[int],
        query_str: str,
    ) -> str:
        """Generate response from community summaries.

        Args:
            community_summaries: Community summaries dictionary
            community_ids: List of community IDs to use
            query_str: Query string

        Returns:
            Response from summaries
        """
        community_answers = []
        for cid in community_ids:
            if cid in community_summaries:
                answer = self._generate_answer_from_summary(
                    community_summaries[cid], query_str
                )
                community_answers.append(answer)

        final_answer = self._aggregate_answers(community_answers)
        self.last_citations = []
        return final_answer

    def _generate_answer_from_cited_context(
        self, cited_context: str, query: str
    ) -> str:
        """Generate answer using cited context with inline markers.

        Args:
            cited_context: Context with citation markers
            query: Query string

        Returns:
            Generated answer with inline citations
        """
        messages = [
            ChatMessage(role="system", content=CITED_CONTEXT_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Evidence bullets (with citations):\n{cited_context}\n\n"
                f"Question: {query}\n\n"
                f"Write a concise answer that includes inline [n] markers referencing the relevant bullets.",
            ),
        ]

        response = self.llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def _generate_answer_from_summary(self, community_summary: str, query: str) -> str:
        """Generate answer from a single community summary.

        Args:
            community_summary: Community summary text
            query: Query string

        Returns:
            Generated answer
        """
        messages = [
            ChatMessage(role="system", content=ONCOLOGY_ASSISTANT_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Context:\n{community_summary}\n\n"
                f"Question: {query}\n\n"
                f"Answer strictly from the context above.",
            ),
        ]

        response = self.llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def _aggregate_answers(self, community_answers: List[str]) -> str:
        """Aggregate multiple community answers into a coherent response.

        Args:
            community_answers: List of individual answers

        Returns:
            Aggregated final answer
        """
        if not community_answers:
            return "No relevant knowledge found in cached community summaries."

        answers_bulleted = "\n- " + "\n- ".join(
            a.strip() for a in community_answers if a.strip()
        )

        messages = [
            ChatMessage(role="system", content=ANSWER_AGGREGATION_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Combine these into one coherent answer:\n{answers_bulleted}",
            ),
        ]

        response = self.llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def get_query_stats(self, query_str: str) -> Dict[str, Any]:
        """Get statistics about query processing.

        Args:
            query_str: Query string

        Returns:
            Dictionary with query processing statistics
        """
        entities = self._entity_resolver.resolve_entities(
            query_str, self.similarity_top_k
        )
        community_ids = self._retrieve_entity_communities(entities)
        community_summaries = self.graph_store.get_community_summaries()

        return {
            "query": query_str,
            "resolved_entities": len(entities),
            "relevant_communities": len(community_ids),
            "total_communities": len(community_summaries),
            "last_citations_count": len(self.last_citations),
            "has_triplet_provenance": bool(self.graph_store.triplet_provenance),
        }
