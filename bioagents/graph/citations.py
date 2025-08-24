"""
Citation Management

This module provides citation building and formatting functionality for query responses,
following the Single Responsibility Principle and focusing on citation concerns.

Author: Theodore Mui
Date: 2025-08-24
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from bioagents.utils.text_utils import make_contextual_snippet

from .interfaces import ICitationBuilder


class CitationBuilder(ICitationBuilder):
    """Builder for citation information from provenance data.

    Creates citation context and metadata for query responses, handling
    deduplication, formatting, and fallback strategies for missing data.

    Features:
        - Deduplication by provenance ID
        - Fallback strategies for missing snippets
        - Configurable citation formatting
        - Context block generation with citation markers
    """

    def __init__(self, snippet_max_length: int = 500):
        """Initialize citation builder.

        Args:
            snippet_max_length: Maximum length for citation snippets
        """
        self.snippet_max_length = snippet_max_length

    def build_citations(
        self,
        triplets: List[Tuple[str, str]],
        provenance_data: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Build citation context and metadata from triplets and provenance.

        Args:
            triplets: List of (triplet_key, detail) tuples
            provenance_data: Dictionary mapping triplet keys to provenance lists

        Returns:
            Tuple of (context_blocks, citations)
            - context_blocks: List of formatted context strings with citation markers
            - citations: List of citation metadata dictionaries
        """
        citations = []
        context_blocks = []
        seen_provenance = set()

        for idx, (triplet_key, detail) in enumerate(triplets, start=1):
            # Get provenance for this triplet
            prov_list = provenance_data.get(triplet_key, [])
            prov = prov_list[0] if prov_list else {}

            # Create citation ID
            citation_id = idx

            # Add context block with citation marker
            context_blocks.append(f"- {detail} [{citation_id}]")

            # Create unique signature for deduplication
            signature = self._create_provenance_signature(prov, triplet_key)

            # Skip if we've already seen this provenance
            if signature in seen_provenance:
                continue
            seen_provenance.add(signature)

            # Build citation metadata
            citation = self._build_citation_metadata(citation_id, triplet_key, prov)
            citations.append(citation)

        return context_blocks, citations

    def format_citation(self, citation: Dict[str, Any]) -> str:
        """Format a single citation for display.

        Args:
            citation: Citation metadata dictionary

        Returns:
            Formatted citation string
        """
        title = citation.get("title") or citation.get("doc_id") or "Source"
        page = citation.get("page")
        paragraph = citation.get("paragraph")
        snippet = citation.get("snippet", "")

        # Build location string
        location_parts = []
        if page is not None:
            location_parts.append(f"p. {page}")
        if paragraph is not None:
            location_parts.append(f"¶ {paragraph}")

        location_str = f" ({', '.join(location_parts)})" if location_parts else ""

        # Format final citation
        return f"[{citation['id']}] {title}{location_str}: {snippet}"

    def format_citations_section(self, citations: List[Dict[str, Any]]) -> str:
        """Format complete citations section.

        Args:
            citations: List of citation metadata dictionaries

        Returns:
            Formatted citations section string
        """
        if not citations:
            return ""

        lines = ["\nCitations:"]
        for citation in citations:
            formatted_citation = self.format_citation(citation)
            lines.append(formatted_citation)

        return "\n".join(lines)

    def _create_provenance_signature(
        self, prov: Dict[str, Any], triplet_key: str
    ) -> str:
        """Create unique signature for provenance deduplication.

        Args:
            prov: Provenance metadata dictionary
            triplet_key: Triplet key string

        Returns:
            Unique signature string
        """
        return (
            prov.get("provenance_id")
            or f"{triplet_key}|{prov.get('source_doc_id')}|{prov.get('source_page')}|{prov.get('source_paragraph_index')}"
        )

    def _build_citation_metadata(
        self, citation_id: int, triplet_key: str, prov: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build citation metadata dictionary.

        Args:
            citation_id: Unique citation ID
            triplet_key: Triplet key string
            prov: Provenance metadata dictionary

        Returns:
            Citation metadata dictionary
        """
        return {
            "id": citation_id,
            "triplet_key": triplet_key,
            "doc_id": prov.get("source_doc_id"),
            "title": prov.get("source_doc_title"),
            "page": prov.get("source_page"),
            "paragraph": prov.get("source_paragraph_index"),
            "file_path": prov.get("source_file_path"),
            "snippet": prov.get("source_snippet", ""),
            "provenance_id": prov.get("provenance_id"),
        }


class EnhancedCitationBuilder(CitationBuilder):
    """Enhanced citation builder with fallback strategies for missing snippets.

    Extends the base citation builder with additional strategies for finding
    snippet text when it's missing from provenance data.
    """

    def __init__(
        self,
        snippet_max_length: int = 500,
        graph_store=None,
        enable_fallback_strategies: bool = True,
    ):
        """Initialize enhanced citation builder.

        Args:
            snippet_max_length: Maximum length for citation snippets
            graph_store: Graph store for fallback snippet retrieval
            enable_fallback_strategies: Whether to use fallback strategies
        """
        super().__init__(snippet_max_length)
        self.graph_store = graph_store
        self.enable_fallback_strategies = enable_fallback_strategies

    def build_citations(
        self,
        triplets: List[Tuple[str, str]],
        provenance_data: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Build citations with enhanced snippet retrieval.

        Args:
            triplets: List of (triplet_key, detail) tuples
            provenance_data: Dictionary mapping triplet keys to provenance lists

        Returns:
            Tuple of (context_blocks, citations) with enhanced snippets
        """
        context_blocks, citations = super().build_citations(triplets, provenance_data)

        # Enhance citations with fallback snippet strategies
        if self.enable_fallback_strategies:
            self._enhance_citations_with_fallback_snippets(citations, provenance_data)

        return context_blocks, citations

    def _enhance_citations_with_fallback_snippets(
        self,
        citations: List[Dict[str, Any]],
        provenance_data: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Enhance citations with fallback snippet strategies.

        Args:
            citations: List of citation dictionaries to enhance
            provenance_data: Provenance data for fallback lookup
        """
        for citation in citations:
            raw_snippet = citation.get("snippet", "")

            # If no snippet, try fallback strategies
            if not raw_snippet:
                raw_snippet = self._get_fallback_snippet(citation, provenance_data)

            # Clean and format snippet
            if raw_snippet:
                citation["snippet"] = make_contextual_snippet(
                    raw_snippet, "", max_length=self.snippet_max_length
                )

    def _get_fallback_snippet(
        self, citation: Dict[str, Any], provenance_data: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Get fallback snippet using multiple strategies.

        Args:
            citation: Citation metadata dictionary
            provenance_data: Provenance data for lookup

        Returns:
            Fallback snippet text or empty string
        """
        triplet_key = citation.get("triplet_key")
        if not triplet_key:
            return ""

        # Strategy 1: Try to get snippet from triplet provenance
        if triplet_key in provenance_data:
            prov_list = provenance_data[triplet_key]
            for prov in prov_list:
                snippet = prov.get("source_snippet")
                if snippet:
                    return snippet

        # Strategy 2: Try to get detail from community info
        if self.graph_store and hasattr(self.graph_store, "community_info"):
            return self._get_snippet_from_community_info(triplet_key)

        return ""

    def _get_snippet_from_community_info(self, triplet_key: str) -> str:
        """Get snippet from graph store community info.

        Args:
            triplet_key: Triplet key to search for

        Returns:
            Detail text from community info or empty string
        """
        try:
            community_info = self.graph_store.community_info or {}
            for community_id, items in community_info.items():
                for item in items:
                    if (
                        isinstance(item, dict)
                        and item.get("triplet_key") == triplet_key
                    ):
                        return item.get("detail", "")
        except (AttributeError, TypeError):
            pass

        return ""

    def get_citation_stats(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about citations.

        Args:
            citations: List of citation dictionaries

        Returns:
            Dictionary with citation statistics
        """
        if not citations:
            return {"total_citations": 0}

        citations_with_snippets = sum(1 for c in citations if c.get("snippet"))
        unique_docs = len(set(c.get("doc_id") for c in citations if c.get("doc_id")))
        unique_pages = len(
            set(c.get("page") for c in citations if c.get("page") is not None)
        )

        return {
            "total_citations": len(citations),
            "citations_with_snippets": citations_with_snippets,
            "snippet_coverage": citations_with_snippets / len(citations),
            "unique_documents": unique_docs,
            "unique_pages": unique_pages,
            "avg_snippet_length": sum(len(c.get("snippet", "")) for c in citations)
            / len(citations),
        }
