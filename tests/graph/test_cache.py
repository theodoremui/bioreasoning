"""
Tests for Cache Management

Comprehensive tests for the cache management system including file operations,
signature validation, and data normalization.

Author: Theodore Mui
Date: 2025-08-24
"""

import json
import os
import pytest
from unittest.mock import Mock, patch

from bioagents.graph.cache import CommunityCache


class TestCommunityCache:
    """Tests for CommunityCache class."""
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = CommunityCache()
        assert cache._encoding == "utf-8"
        
        cache_custom = CommunityCache(encoding="latin-1")
        assert cache_custom._encoding == "latin-1"
    
    def test_save_communities_success(self, temp_cache_file):
        """Test successful community data saving."""
        cache = CommunityCache()
        test_data = {
            "graph_signature": "test_sig",
            "community_summary": {"0": "test summary"}  # JSON converts int keys to strings
        }
        
        cache.save_communities(str(temp_cache_file), test_data)
        
        # Verify file was created and contains correct data
        assert temp_cache_file.exists()
        with open(temp_cache_file, 'r') as f:
            saved_data = json.load(f)
        assert saved_data == test_data
    
    def test_save_communities_creates_directory(self, tmp_path):
        """Test that save_communities creates necessary directories."""
        cache = CommunityCache()
        nested_path = tmp_path / "nested" / "dir" / "cache.json"
        test_data = {"test": "data"}
        
        cache.save_communities(str(nested_path), test_data)
        
        assert nested_path.exists()
        with open(nested_path, 'r') as f:
            saved_data = json.load(f)
        assert saved_data == test_data
    
    def test_save_communities_empty_data(self, temp_cache_file):
        """Test saving empty data raises ValueError."""
        cache = CommunityCache()
        
        with pytest.raises(ValueError, match="Cannot save empty data"):
            cache.save_communities(str(temp_cache_file), {})
        
        with pytest.raises(ValueError, match="Cannot save empty data"):
            cache.save_communities(str(temp_cache_file), None)
    
    def test_save_communities_atomic_operation(self, temp_cache_file):
        """Test that save operation is atomic."""
        cache = CommunityCache()
        test_data = {"test": "data"}
        
        # Mock json.dump to raise an exception
        with patch('json.dump', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                cache.save_communities(str(temp_cache_file), test_data)
        
        # Verify original file doesn't exist and temp file was cleaned up
        assert not temp_cache_file.exists()
        temp_file = temp_cache_file.parent / f"{temp_cache_file.name}.tmp"
        assert not temp_file.exists()
    
    def test_load_communities_success(self, temp_cache_file, sample_community_data):
        """Test successful community data loading."""
        # Save test data first
        with open(temp_cache_file, 'w') as f:
            json.dump(sample_community_data, f)
        
        cache = CommunityCache()
        loaded_data = cache.load_communities(str(temp_cache_file))
        
        assert loaded_data is not None
        assert loaded_data["graph_signature"] == "test_signature"
        assert 0 in loaded_data["community_summary"]
    
    def test_load_communities_file_not_found(self, tmp_path):
        """Test loading from non-existent file."""
        cache = CommunityCache()
        non_existent = tmp_path / "nonexistent.json"
        
        result = cache.load_communities(str(non_existent))
        assert result is None
    
    def test_load_communities_invalid_json(self, temp_cache_file):
        """Test loading invalid JSON file."""
        # Write invalid JSON
        with open(temp_cache_file, 'w') as f:
            f.write("invalid json content")
        
        cache = CommunityCache()
        result = cache.load_communities(str(temp_cache_file))
        assert result is None
    
    def test_load_communities_non_dict_data(self, temp_cache_file):
        """Test loading non-dictionary data."""
        # Write valid JSON but not a dictionary
        with open(temp_cache_file, 'w') as f:
            json.dump(["not", "a", "dict"], f)
        
        cache = CommunityCache()
        result = cache.load_communities(str(temp_cache_file))
        assert result is None
    
    def test_compute_signature_empty_triplets(self):
        """Test signature computation with empty triplets."""
        cache = CommunityCache()
        signature = cache.compute_signature([])
        assert signature == ""
    
    def test_compute_signature_with_triplets(self, sample_triplets):
        """Test signature computation with sample triplets."""
        cache = CommunityCache()
        signature = cache.compute_signature(sample_triplets)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length
        
        # Test consistency - same triplets should produce same signature
        signature2 = cache.compute_signature(sample_triplets)
        assert signature == signature2
    
    def test_compute_signature_deterministic(self):
        """Test that signature computation is deterministic."""
        cache = CommunityCache()
        
        # Create mock triplets
        triplets1 = []
        triplets2 = []
        
        for i in range(2):
            entity1 = Mock()
            entity1.name = f"Entity{i}"
            entity2 = Mock()
            entity2.name = f"Target{i}"
            relation = Mock()
            relation.label = "relates_to"
            relation.properties = {"relationship_description": f"desc{i}"}
            
            triplets1.append((entity1, relation, entity2))
            triplets2.append((entity1, relation, entity2))
        
        sig1 = cache.compute_signature(triplets1)
        sig2 = cache.compute_signature(triplets2)
        assert sig1 == sig2
    
    def test_normalize_community_data(self):
        """Test community data normalization."""
        cache = CommunityCache()
        raw_data = {
            "community_summary": {
                "0": "summary 0",  # String key
                "1": "summary 1"
            },
            "community_info": {
                "0": [
                    {"detail": "detail 0", "triplet_key": "key0"},
                    "legacy string item"  # Legacy format
                ],
                "1": [
                    {"detail": "detail 1", "triplet_key": "key1"}
                ]
            }
        }
        
        normalized = cache._normalize_community_data(raw_data)
        
        # Check community_summary normalization
        assert 0 in normalized["community_summary"]
        assert 1 in normalized["community_summary"]
        assert normalized["community_summary"][0] == "summary 0"
        
        # Check community_info normalization
        assert 0 in normalized["community_info"]
        assert len(normalized["community_info"][0]) == 2
        
        # Check legacy item conversion
        legacy_item = normalized["community_info"][0][1]
        assert legacy_item["detail"] == "legacy string item"
        assert legacy_item["triplet_key"] is None
    
    def test_validate_cache_integrity_valid(self, temp_cache_file):
        """Test cache integrity validation with valid cache."""
        cache = CommunityCache()
        test_data = {"graph_signature": "test_signature"}
        
        with open(temp_cache_file, 'w') as f:
            json.dump(test_data, f)
        
        is_valid = cache.validate_cache_integrity(str(temp_cache_file), "test_signature")
        assert is_valid is True
    
    def test_validate_cache_integrity_invalid(self, temp_cache_file):
        """Test cache integrity validation with invalid cache."""
        cache = CommunityCache()
        test_data = {"graph_signature": "old_signature"}
        
        with open(temp_cache_file, 'w') as f:
            json.dump(test_data, f)
        
        is_valid = cache.validate_cache_integrity(str(temp_cache_file), "new_signature")
        assert is_valid is False
    
    def test_validate_cache_integrity_no_file(self, tmp_path):
        """Test cache integrity validation with missing file."""
        cache = CommunityCache()
        non_existent = tmp_path / "nonexistent.json"
        
        is_valid = cache.validate_cache_integrity(str(non_existent), "any_signature")
        assert is_valid is False
    
    def test_get_cache_metadata_success(self, temp_cache_file):
        """Test getting cache metadata."""
        cache = CommunityCache()
        test_data = {
            "graph_signature": "test_sig",
            "max_cluster_size": 5,
            "algorithm_metadata": {"algorithm": "test"}
        }
        
        with open(temp_cache_file, 'w') as f:
            json.dump(test_data, f)
        
        metadata = cache.get_cache_metadata(str(temp_cache_file))
        
        assert metadata is not None
        assert metadata["graph_signature"] == "test_sig"
        assert metadata["max_cluster_size"] == 5
        assert metadata["algorithm_metadata"]["algorithm"] == "test"
        assert "file_size" in metadata
        assert "last_modified" in metadata
    
    def test_get_cache_metadata_no_file(self, tmp_path):
        """Test getting metadata from non-existent file."""
        cache = CommunityCache()
        non_existent = tmp_path / "nonexistent.json"
        
        metadata = cache.get_cache_metadata(str(non_existent))
        assert metadata is None
    
    def test_get_cache_metadata_invalid_json(self, temp_cache_file):
        """Test getting metadata from invalid JSON file."""
        with open(temp_cache_file, 'w') as f:
            f.write("invalid json")
        
        cache = CommunityCache()
        metadata = cache.get_cache_metadata(str(temp_cache_file))
        assert metadata is None
    
    def test_encoding_parameter(self, temp_cache_file):
        """Test that encoding parameter is respected."""
        cache = CommunityCache(encoding="utf-16")
        test_data = {"test": "data with unicode: 测试"}
        
        cache.save_communities(str(temp_cache_file), test_data)
        
        # Verify file was saved with correct encoding
        with open(temp_cache_file, 'r', encoding='utf-16') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data
        
        # Verify loading respects encoding
        loaded_data = cache.load_communities(str(temp_cache_file))
        assert loaded_data == test_data
