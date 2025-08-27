"""
Provenance Management

This module provides provenance tracking and building functionality for graph triplets,
following the Single Responsibility Principle by focusing solely on provenance concerns.

Author: Theodore Mui
Date: 2025-08-24
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from bioagents.utils.text_utils import make_contextual_snippet

from .interfaces import IProvenanceBuilder


class ProvenanceBuilder(IProvenanceBuilder):
    """Builder for triplet provenance information.

    Extracts and builds provenance data from graph relations, providing
    traceability back to source documents and text locations.

    Features:
        - Extracts provenance from relation properties
        - Generates unique provenance IDs
        - Creates contextual snippets
        - Handles missing data gracefully
    """

    def __init__(self, snippet_max_length: int = 1000):
        """Initialize provenance builder.

        Args:
            snippet_max_length: Maximum length for contextual snippets
        """
        self.snippet_max_length = snippet_max_length

    def build_triplet_provenance(
        self, triplets: List[Tuple]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build provenance information from triplets.

        Args:
            triplets: List of (entity1, relation, entity2) triplets

        Returns:
            Dictionary mapping triplet keys to provenance data lists
        """
        provenance = {}

        for triplet in triplets:
            if len(triplet) < 3:
                continue

            entity1, relation, entity2 = triplet[0], triplet[1], triplet[2]

            try:
                prov_data = self.extract_provenance_from_relation(relation)
                if prov_data:
                    triplet_key = prov_data.get(
                        "triplet_key",
                        f"{relation.source_id}|{relation.label}|{relation.target_id}",
                    )

                    provenance.setdefault(triplet_key, [])

                    # Deduplicate by provenance_id
                    if not any(
                        p.get("provenance_id") == prov_data.get("provenance_id")
                        for p in provenance[triplet_key]
                    ):
                        provenance[triplet_key].append(prov_data)

            except Exception:
                # Skip triplets that can't be processed
                continue

        return provenance

    def extract_provenance_from_relation(self, relation: Any) -> Dict[str, Any]:
        """Extract provenance data from a relation object.

        Args:
            relation: Relation object with properties

        Returns:
            Dictionary with provenance information
        """
        # Get properties from relation and related entities
        props_r = getattr(relation, "properties", {}) or {}

        # Create triplet key
        triplet_key = (
            props_r.get("triplet_key")
            or f"{relation.source_id}|{relation.label}|{relation.target_id}"
        )

        # Extract provenance fields with fallbacks
        prov = {
            "source_doc_id": self._get_field(props_r, ["doc_id", "source_doc_id"]),
            "source_doc_title": self._get_field(
                props_r, ["doc_title", "source_doc_title"]
            ),
            "source_file_path": self._get_field(
                props_r, ["file_path", "source_file_path"]
            ),
            "source_page": self._get_field(props_r, ["page_number", "source_page"]),
            "source_paragraph_index": self._get_field(
                props_r, ["paragraph_index", "source_paragraph_index"]
            ),
            "char_start": self._get_field(props_r, ["char_start"]),
            "char_end": self._get_field(props_r, ["char_end"]),
            "extraction_node_id": self._get_field(props_r, ["extraction_node_id"]),
            "triplet_key": triplet_key,
        }

        # Create contextual snippet
        prov["source_snippet"] = self._create_snippet(props_r)

        # Generate provenance ID if not present
        prov["provenance_id"] = self._get_or_generate_provenance_id(props_r, prov)

        return prov

    def _get_field(self, properties: Dict[str, Any], field_names: List[str]) -> Any:
        """Get field value with fallback options.

        Args:
            properties: Properties dictionary
            field_names: List of field names to try in order

        Returns:
            First available field value or None
        """
        for field_name in field_names:
            value = properties.get(field_name)
            if value is not None:
                return value
        return None

    def _create_snippet(self, properties: Dict[str, Any]) -> str:
        """Create contextual snippet from available text, cleaning any metadata prefixes.

        This method processes existing graph relationship properties and cleans
        any polluted source_snippet values that may contain metadata prefixes.

        Args:
            properties: Properties dictionary from graph relationships

        Returns:
            Clean contextual snippet text without metadata prefixes
        """
        # Try different sources for snippet text
        snippet_sources = [
            properties.get("source_snippet"),
            properties.get("relationship_description"),
            properties.get("entity_description"),
        ]

        raw_text = ""
        for source in snippet_sources:
            if source and str(source).strip():
                raw_text = str(source)
                break

        if raw_text:
            # Clean any metadata prefixes that may exist in stored data
            clean_text = self._clean_metadata_prefix(raw_text)
            return make_contextual_snippet(
                clean_text, "", max_length=self.snippet_max_length
            )
        return ""

    def _clean_metadata_prefix(self, text: str) -> str:
        """Clean metadata prefixes from text that may exist in stored graph data.
        
        This handles cases where the Neo4j database contains relationships with
        polluted source_snippet properties from before our extraction fixes.
        
        Args:
            text: Raw text that may contain metadata prefixes
            
        Returns:
            Clean text with metadata prefixes removed
        """
        if not text:
            return ""
            
        text = str(text).strip()
        
        # Pattern 1: Look for " - " that separates metadata from content
        dash_patterns = [" - ", " – ", " — "]
        for pattern in dash_patterns:
            if pattern in text:
                parts = text.split(pattern)
                if len(parts) > 1:
                    # Check if the part before dash looks like metadata
                    before_dash = parts[0]
                    metadata_indicators = [
                        "doc_id:", "doc_title:", "file_path:", "page_number:", 
                        "paragraph_index:", "char_start:", "char_end:"
                    ]
                    
                    if any(indicator in before_dash for indicator in metadata_indicators):
                        # Join everything after the first dash as content
                        content = pattern.join(parts[1:]).strip()
                        if content:
                            return content
        
        # Pattern 2: Look for content after metadata fields using regex
        import re
        metadata_pattern = r'^.*?(?:char_end:\s*\d+|paragraph_index:\s*\d+|page_number:\s*\d+)(?:\s+.*?)?\s*[-–—]\s*(.+)$'
        match = re.match(metadata_pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content:
                return content
        
        # Pattern 3: If no clear separator, but text starts with metadata indicators,
        # try to find where the actual content begins
        if any(text.startswith(f"{indicator} ") for indicator in ["doc_id:", "file_path:"]):
            words = text.split()
            for i, word in enumerate(words):
                # Look for words that start with capital letters and don't contain colons
                if (len(word) > 2 and 
                    word[0].isupper() and 
                    ":" not in word and 
                    not word.replace("-", "").replace("_", "").isdigit()):
                    # Found potential start of content
                    content = " ".join(words[i:]).strip()
                    if len(content) > 20:  # Make sure it's substantial content
                        return content
        
        # If no patterns matched, return the original text
        return text

    def _get_or_generate_provenance_id(
        self, properties: Dict[str, Any], prov_data: Dict[str, Any]
    ) -> str:
        """Get existing provenance ID or generate a new one.

        Args:
            properties: Properties dictionary
            prov_data: Provenance data dictionary

        Returns:
            Provenance ID string
        """
        existing_id = properties.get("provenance_id")
        if existing_id:
            return str(existing_id)

        # Generate new ID based on provenance data
        payload = json.dumps(prov_data, default=str, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def create_provenance_summary(
        self, provenance_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Create a summary of provenance data.

        Args:
            provenance_data: Dictionary of provenance information

        Returns:
            Summary statistics and information
        """
        total_triplets = len(provenance_data)
        total_provenance_entries = sum(
            len(entries) for entries in provenance_data.values()
        )

        # Count unique documents
        unique_docs = set()
        unique_pages = set()

        for entries in provenance_data.values():
            for entry in entries:
                doc_id = entry.get("source_doc_id")
                if doc_id:
                    unique_docs.add(doc_id)

                page = entry.get("source_page")
                if page is not None:
                    unique_pages.add(page)

        return {
            "total_triplets_with_provenance": total_triplets,
            "total_provenance_entries": total_provenance_entries,
            "unique_documents": len(unique_docs),
            "unique_pages": len(unique_pages),
            "average_provenance_per_triplet": (
                total_provenance_entries / total_triplets if total_triplets > 0 else 0
            ),
        }

    def validate_provenance_data(
        self, provenance_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Validate provenance data and return any issues found.

        Args:
            provenance_data: Dictionary of provenance information

        Returns:
            List of validation issues (empty if all valid)
        """
        issues = []

        for triplet_key, entries in provenance_data.items():
            if not triplet_key:
                issues.append("Found empty triplet key")
                continue

            if not entries:
                issues.append(f"No provenance entries for triplet: {triplet_key}")
                continue

            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    issues.append(
                        f"Invalid provenance entry type for {triplet_key}[{i}]"
                    )
                    continue

                # Check required fields
                if not entry.get("provenance_id"):
                    issues.append(f"Missing provenance_id for {triplet_key}[{i}]")

                # Check for reasonable data
                if not any(
                    [
                        entry.get("source_doc_id"),
                        entry.get("source_snippet"),
                        entry.get("source_file_path"),
                    ]
                ):
                    issues.append(f"No source information for {triplet_key}[{i}]")

        return issues
