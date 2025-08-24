"""
Ranking Strategies

This module provides ranking algorithms for communities and triplets in query processing,
following the Strategy Pattern and Single Responsibility Principle.

Author: Theodore Mui
Date: 2025-08-24
"""

import re
from typing import Dict, List, Optional, Tuple

from .constants import MEDICAL_EXPANSIONS, PATTERNS, RANKING_WEIGHTS
from .interfaces import IRankingStrategy


class KeywordOverlapRanking(IRankingStrategy):
    """Ranking strategy based on keyword overlap between query and content.

    Uses multiple scoring approaches including basic term overlap, token matching,
    and medical term expansion for enhanced relevance in medical domains.
    """

    def __init__(
        self,
        basic_weight: float = RANKING_WEIGHTS["basic_score_weight"],
        token_weight: float = RANKING_WEIGHTS["token_score_weight"],
        expanded_weight: float = RANKING_WEIGHTS["expanded_score_weight"],
        medical_expansions: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize ranking strategy with weights and expansions.

        Args:
            basic_weight: Weight for basic term overlap scoring
            token_weight: Weight for token-based scoring
            expanded_weight: Weight for expanded term scoring
            medical_expansions: Medical term expansions dictionary
        """
        self.basic_weight = basic_weight
        self.token_weight = token_weight
        self.expanded_weight = expanded_weight
        self.medical_expansions = medical_expansions or MEDICAL_EXPANSIONS.copy()

    def rank_communities(
        self,
        community_summaries: Dict[int, str],
        query: str,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """Rank communities by relevance to query using enhanced keyword overlap.

        Args:
            community_summaries: Dictionary mapping community ID to summary text
            query: Query string
            candidate_ids: List of candidate community IDs to rank

        Returns:
            List of (community_id, score) tuples sorted by relevance
        """
        # Extract query terms and tokens
        query_terms = {
            t.lower() for t in re.findall(PATTERNS["query_terms_basic"], query)
        }
        query_tokens = set(re.findall(PATTERNS["word_tokens"], query.lower()))

        # Expand query terms with medical synonyms
        expanded_query_terms = self._expand_query_terms(query_tokens, query_terms)

        scored = []
        for cid in candidate_ids:
            summary = community_summaries.get(cid, "")
            if not summary:
                scored.append((cid, 0.0))
                continue

            # Extract terms and tokens from summary
            text_terms = {
                t.lower() for t in re.findall(PATTERNS["query_terms_basic"], summary)
            }
            text_tokens = set(re.findall(PATTERNS["word_tokens"], summary.lower()))

            # Calculate different types of overlap
            basic_score = len(query_terms & text_terms)
            token_score = len(query_tokens & text_tokens)
            expanded_score = len(expanded_query_terms & text_terms)

            # Weighted final score
            final_score = (
                basic_score * self.basic_weight
                + token_score * self.token_weight
                + expanded_score * self.expanded_weight
            )

            scored.append((cid, final_score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def rank_triplets(
        self, triplets: List[Tuple[str, str]], query: str
    ) -> List[Tuple[str, str]]:
        """Rank triplets by relevance to query using keyword overlap.

        Args:
            triplets: List of (triplet_key, detail) tuples
            query: Query string

        Returns:
            List of (triplet_key, detail) tuples sorted by relevance
        """
        query_terms = {
            t.lower() for t in re.findall(PATTERNS["query_terms_detailed"], query)
        }

        scored = []
        for triplet_key, detail in triplets:
            detail_terms = {
                t.lower() for t in re.findall(PATTERNS["query_terms_detailed"], detail)
            }
            overlap_score = len(query_terms & detail_terms)
            scored.append((overlap_score, (triplet_key, detail)))

        # Sort by score descending and return triplets
        scored.sort(key=lambda x: x[0], reverse=True)
        return [triplet for _, triplet in scored]

    def _expand_query_terms(self, query_tokens: set, query_terms: set) -> set:
        """Expand query terms with medical synonyms and related terms.

        Args:
            query_tokens: Set of query word tokens
            query_terms: Set of query terms

        Returns:
            Expanded set of query terms
        """
        expanded_terms = set(query_terms)

        for token in query_tokens:
            if token in self.medical_expansions:
                expanded_terms.update(self.medical_expansions[token])

        return expanded_terms

    def add_medical_expansion(self, term: str, expansions: List[str]) -> None:
        """Add medical term expansion.

        Args:
            term: Base medical term
            expansions: List of expansion terms
        """
        self.medical_expansions[term.lower()] = expansions

    def get_medical_expansions(self) -> Dict[str, List[str]]:
        """Get current medical expansions.

        Returns:
            Dictionary of medical term expansions
        """
        return self.medical_expansions.copy()


class SemanticSimilarityRanking(IRankingStrategy):
    """Ranking strategy based on semantic similarity (placeholder for future implementation).

    This class provides a framework for semantic similarity-based ranking
    that could use embeddings or other semantic matching techniques.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        """Initialize semantic similarity ranking.

        Args:
            similarity_threshold: Minimum similarity threshold
        """
        self.similarity_threshold = similarity_threshold

    def rank_communities(
        self,
        community_summaries: Dict[int, str],
        query: str,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """Rank communities using semantic similarity (placeholder).

        Args:
            community_summaries: Dictionary mapping community ID to summary text
            query: Query string
            candidate_ids: List of candidate community IDs to rank

        Returns:
            List of (community_id, score) tuples sorted by relevance
        """
        # Placeholder implementation - falls back to keyword overlap
        keyword_ranker = KeywordOverlapRanking()
        return keyword_ranker.rank_communities(
            community_summaries, query, candidate_ids
        )

    def rank_triplets(
        self, triplets: List[Tuple[str, str]], query: str
    ) -> List[Tuple[str, str]]:
        """Rank triplets using semantic similarity (placeholder).

        Args:
            triplets: List of (triplet_key, detail) tuples
            query: Query string

        Returns:
            List of (triplet_key, detail) tuples sorted by relevance
        """
        # Placeholder implementation - falls back to keyword overlap
        keyword_ranker = KeywordOverlapRanking()
        return keyword_ranker.rank_triplets(triplets, query)


class HybridRanking(IRankingStrategy):
    """Hybrid ranking strategy combining multiple ranking approaches.

    Combines keyword overlap and semantic similarity (when available) with
    configurable weights for different ranking strategies.
    """

    def __init__(
        self,
        keyword_weight: float = 0.7,
        semantic_weight: float = 0.3,
        keyword_ranker: Optional[IRankingStrategy] = None,
        semantic_ranker: Optional[IRankingStrategy] = None,
    ):
        """Initialize hybrid ranking with strategy weights.

        Args:
            keyword_weight: Weight for keyword-based ranking
            semantic_weight: Weight for semantic similarity ranking
            keyword_ranker: Keyword ranking strategy (optional)
            semantic_ranker: Semantic ranking strategy (optional)
        """
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

        # Dependency injection with defaults
        self.keyword_ranker = keyword_ranker or KeywordOverlapRanking()
        self.semantic_ranker = semantic_ranker or SemanticSimilarityRanking()

        # Normalize weights
        total_weight = keyword_weight + semantic_weight
        if total_weight > 0:
            self.keyword_weight /= total_weight
            self.semantic_weight /= total_weight

    def rank_communities(
        self,
        community_summaries: Dict[int, str],
        query: str,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """Rank communities using hybrid approach.

        Args:
            community_summaries: Dictionary mapping community ID to summary text
            query: Query string
            candidate_ids: List of candidate community IDs to rank

        Returns:
            List of (community_id, score) tuples sorted by relevance
        """
        # Get rankings from both strategies
        keyword_rankings = self.keyword_ranker.rank_communities(
            community_summaries, query, candidate_ids
        )
        semantic_rankings = self.semantic_ranker.rank_communities(
            community_summaries, query, candidate_ids
        )

        # Combine scores
        combined_scores = self._combine_rankings(keyword_rankings, semantic_rankings)

        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores

    def rank_triplets(
        self, triplets: List[Tuple[str, str]], query: str
    ) -> List[Tuple[str, str]]:
        """Rank triplets using hybrid approach.

        Args:
            triplets: List of (triplet_key, detail) tuples
            query: Query string

        Returns:
            List of (triplet_key, detail) tuples sorted by relevance
        """
        # For triplets, use keyword ranking as primary strategy
        # (semantic ranking for triplets would require more complex implementation)
        return self.keyword_ranker.rank_triplets(triplets, query)

    def _combine_rankings(
        self,
        keyword_rankings: List[Tuple[int, float]],
        semantic_rankings: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """Combine rankings from different strategies.

        Args:
            keyword_rankings: Rankings from keyword strategy
            semantic_rankings: Rankings from semantic strategy

        Returns:
            Combined rankings with weighted scores
        """
        # Create score dictionaries for easy lookup
        keyword_scores = {cid: score for cid, score in keyword_rankings}
        semantic_scores = {cid: score for cid, score in semantic_rankings}

        # Get all community IDs
        all_cids = set(keyword_scores.keys()) | set(semantic_scores.keys())

        # Combine scores with weights
        combined = []
        for cid in all_cids:
            keyword_score = keyword_scores.get(cid, 0.0)
            semantic_score = semantic_scores.get(cid, 0.0)

            combined_score = (
                keyword_score * self.keyword_weight
                + semantic_score * self.semantic_weight
            )

            combined.append((cid, combined_score))

        return combined


class RankingStrategyFactory:
    """Factory for creating ranking strategies.

    Implements the Factory Pattern to provide different ranking strategies
    based on requirements and available resources.
    """

    @staticmethod
    def create_keyword_ranking(**kwargs) -> IRankingStrategy:
        """Create keyword overlap ranking strategy.

        Args:
            **kwargs: Configuration parameters for KeywordOverlapRanking

        Returns:
            KeywordOverlapRanking instance
        """
        return KeywordOverlapRanking(**kwargs)

    @staticmethod
    def create_semantic_ranking(**kwargs) -> IRankingStrategy:
        """Create semantic similarity ranking strategy.

        Args:
            **kwargs: Configuration parameters for SemanticSimilarityRanking

        Returns:
            SemanticSimilarityRanking instance
        """
        return SemanticSimilarityRanking(**kwargs)

    @staticmethod
    def create_hybrid_ranking(**kwargs) -> IRankingStrategy:
        """Create hybrid ranking strategy.

        Args:
            **kwargs: Configuration parameters for HybridRanking

        Returns:
            HybridRanking instance
        """
        return HybridRanking(**kwargs)

    @staticmethod
    def create_default_ranking() -> IRankingStrategy:
        """Create default ranking strategy.

        Returns:
            Default ranking strategy (KeywordOverlapRanking)
        """
        return KeywordOverlapRanking()
