import json
import os
import tempfile

import pytest

import importlib.util
from pathlib import Path


def _load_graphrag_store():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "bioagents" / "nccn-kg.py"
    spec = importlib.util.spec_from_file_location("nccn_kg", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create spec for bioagents/nccn-kg.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphRAGStore


class DummyEntity:
    def __init__(self, name: str):
        self.name = name


class DummyRelation:
    def __init__(self, label: str, source_id: str, target_id: str, properties: dict):
        self.label = label
        self.source_id = source_id
        self.target_id = target_id
        self.properties = properties


GraphRAGStore = _load_graphrag_store()


class DummyGraphStore(GraphRAGStore):
    def __init__(self):
        # Don't call super to avoid real Neo4j connection
        pass

    def get_triplets(self):
        # Deterministic small graph
        e1 = DummyEntity("HER2")
        e2 = DummyEntity("Trastuzumab")
        e3 = DummyEntity("Breast Cancer")

        r1 = DummyRelation(
            label="treats",
            source_id=e2.name,
            target_id=e3.name,
            properties={
                "relationship_description": "Trastuzumab treats HER2+ breast cancer"
            },
        )
        r2 = DummyRelation(
            label="overexpressed_in",
            source_id=e1.name,
            target_id=e3.name,
            properties={"relationship_description": "HER2 is overexpressed in subset"},
        )
        return [
            (e2, r1, e3),
            (e1, r2, e3),
        ]

    def has_graph_data(self) -> bool:
        # In tests, we simulate graph presence
        return True


def test_build_and_save_load_state(tmp_path):
    store = DummyGraphStore()
    # Build communities (uses hierarchical_leiden on a tiny graph)
    store.build_communities()

    persist_file = tmp_path / "state.json"
    store.save_communities(str(persist_file))

    assert persist_file.exists()
    data = json.loads(persist_file.read_text(encoding="utf-8"))
    # Ensure extended fields present
    assert "cluster_assignments" in data
    assert "community_info" in data
    assert "algorithm_metadata" in data

    # Now load into a fresh store and validate round-trip
    store2 = DummyGraphStore()
    # Graph signature must match since triplets are deterministic
    loaded = store2.load_communities(str(persist_file), validate_signature=True)
    assert loaded is True
    assert store2.entity_info
    assert store2.community_summary
    assert isinstance(store2.cluster_assignments, dict)
    assert isinstance(store2.community_info, dict)
    assert store2.algorithm_metadata


def test_signature_validation(tmp_path):
    store = DummyGraphStore()
    store.build_communities()
    persist_file = tmp_path / "state.json"
    store.save_communities(str(persist_file))

    # Tamper signature
    data = json.loads(persist_file.read_text(encoding="utf-8"))
    data["graph_signature"] = "bad"
    persist_file.write_text(json.dumps(data), encoding="utf-8")

    store2 = DummyGraphStore()
    ok = store2.load_communities(str(persist_file), validate_signature=True)
    assert ok is False


def test_prefer_cache_when_graph_empty(tmp_path):
    class EmptyGraphStore(DummyGraphStore):
        def get_triplets(self):
            return []

        def has_graph_data(self) -> bool:
            return False

    # First, build and save with a non-empty store
    base = DummyGraphStore()
    base.build_communities()
    f = tmp_path / "state.json"
    base.save_communities(str(f))

    # Now, simulate restart with empty graph. ensure_communities should load cache without signature validation
    empty = EmptyGraphStore()
    empty.ensure_communities(
        persist_path=str(f), validate_signature=True, prefer_cache_when_graph_empty=True
    )
    assert empty.community_summary  # loaded from cache
    assert isinstance(empty.cluster_assignments, dict)
