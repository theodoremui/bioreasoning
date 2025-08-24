"""
Cache Management

This module provides caching functionality for community data and graph signatures,
following the Single Responsibility Principle by focusing solely on caching concerns.

Author: Theodore Mui
Date: 2025-08-24
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import ICommunityCache


class CommunityCache(ICommunityCache):
    """Community data cache with signature validation.

    Handles persistence and loading of community data with integrity validation
    through graph signatures. Follows SRP by focusing only on caching operations.

    Features:
        - Signature-based cache validation
        - Atomic file operations
        - Directory creation
        - Error handling and recovery
    """

    def __init__(self, encoding: str = "utf-8"):
        """Initialize cache with encoding settings.

        Args:
            encoding: File encoding to use for cache files
        """
        self._encoding = encoding

    def save_communities(self, filepath: str, data: Dict[str, Any]) -> None:
        """Save community data to file with atomic operations.

        Args:
            filepath: Path to save the cache file
            data: Community data to save

        Raises:
            OSError: If file operations fail
            ValueError: If data is invalid
        """
        if not data:
            raise ValueError("Cannot save empty data")

        # Ensure destination directory exists
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Write to temporary file first for atomic operation
        temp_filepath = f"{filepath}.tmp"
        try:
            with open(temp_filepath, "w", encoding=self._encoding) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomic move
            os.rename(temp_filepath, filepath)

        except Exception:
            # Cleanup temporary file on error
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass  # Best effort cleanup
            raise

    def load_communities(
        self, filepath: str, validate_signature: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load community data from file with optional validation.

        Args:
            filepath: Path to the cache file
            validate_signature: Whether to validate the cache signature

        Returns:
            Loaded community data or None if loading fails
        """
        try:
            with open(filepath, "r", encoding=self._encoding) as f:
                data = json.load(f)

            # Basic validation
            if not isinstance(data, dict):
                return None

            # Normalize data types for consistency
            normalized_data = self._normalize_community_data(data)
            return normalized_data

        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def compute_signature(self, triplets: List[Tuple]) -> str:
        """Compute a stable signature for cache validation.

        Args:
            triplets: List of triplets to compute signature from

        Returns:
            SHA256 hash of the triplets
        """
        if not triplets:
            return ""

        items = []
        for triplet in triplets:
            if len(triplet) >= 3:
                entity1, relation, entity2 = triplet[0], triplet[1], triplet[2]

                # Extract names and descriptions safely
                entity1_name = getattr(entity1, "name", str(entity1))
                entity2_name = getattr(entity2, "name", str(entity2))
                relation_label = getattr(relation, "label", str(relation))

                # Extract relationship description
                relation_desc = ""
                try:
                    if hasattr(relation, "properties") and relation.properties:
                        relation_desc = relation.properties.get(
                            "relationship_description", ""
                        )
                except (AttributeError, TypeError):
                    pass

                items.append(
                    (entity1_name, relation_label, entity2_name, relation_desc)
                )

        # Ensure deterministic ordering
        items.sort()

        # Create stable JSON representation
        payload = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def _normalize_community_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize community data for consistent types.

        Args:
            data: Raw community data

        Returns:
            Normalized community data
        """
        normalized = data.copy()

        # Normalize community_summary keys to integers
        if "community_summary" in normalized:
            raw_cs = normalized["community_summary"] or {}
            norm_cs = {}
            for cid, summary in raw_cs.items():
                try:
                    int_cid = int(cid)
                except (ValueError, TypeError):
                    int_cid = cid  # Keep original if conversion fails
                norm_cs[int_cid] = summary
            normalized["community_summary"] = norm_cs

        # Normalize community_info structure
        if "community_info" in normalized:
            raw_ci = normalized["community_info"] or {}
            norm_ci = {}
            for cid, items in raw_ci.items():
                # Convert string keys to integers
                try:
                    int_cid = int(cid)
                except (ValueError, TypeError):
                    int_cid = cid  # Keep original if conversion fails

                # Ensure items are in dict format with 'detail' and 'triplet_key'
                new_items = []
                for item in items:
                    if isinstance(item, dict):
                        new_items.append(
                            {
                                "detail": item.get("detail", ""),
                                "triplet_key": item.get("triplet_key"),
                            }
                        )
                    else:
                        new_items.append({"detail": str(item), "triplet_key": None})
                norm_ci[int_cid] = new_items
            normalized["community_info"] = norm_ci

        return normalized

    def validate_cache_integrity(self, filepath: str, current_signature: str) -> bool:
        """Validate cache file integrity against current signature.

        Args:
            filepath: Path to cache file
            current_signature: Current graph signature to validate against

        Returns:
            True if cache is valid, False otherwise
        """
        data = self.load_communities(filepath, validate_signature=False)
        if not data:
            return False

        cached_signature = data.get("graph_signature")
        return cached_signature == current_signature

    def get_cache_metadata(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Get metadata from cache file without loading full data.

        Args:
            filepath: Path to cache file

        Returns:
            Cache metadata or None if unavailable
        """
        try:
            with open(filepath, "r", encoding=self._encoding) as f:
                # Load only the first part to get metadata
                data = json.load(f)

            return {
                "graph_signature": data.get("graph_signature"),
                "max_cluster_size": data.get("max_cluster_size"),
                "algorithm_metadata": data.get("algorithm_metadata"),
                "file_size": os.path.getsize(filepath),
                "last_modified": os.path.getmtime(filepath),
            }
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
