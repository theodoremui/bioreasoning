import importlib.util
from pathlib import Path

from typing import Any


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "notebooks" / "nccn-kg.py"
    spec = importlib.util.spec_from_file_location("nccn_kg", str(module_path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_engine_with_citations(monkeypatch):
    mod = _load_module()

    class DummyStore(mod.GraphRAGStore):
        def __init__(self):
            # bypass real init
            pass

        @property
        def supports_vector_queries(self) -> bool:
            return False

    store = DummyStore()
    store.entity_info = {"HER2-Positive Breast Cancer": ["1"]}
    store.community_summary = {"1": "HER2 targeted therapy includes trastuzumab."}
    store.community_info = {
        "1": [
            {
                "detail": "HER2-Positive Breast Cancer -> Trastuzumab -> treats -> Trastuzumab is recommended for HER2+ disease",
                "triplet_key": "HER2-Positive Breast Cancer|treats|Trastuzumab",
            }
        ]
    }
    store.triplet_provenance = {
        "HER2-Positive Breast Cancer|treats|Trastuzumab": [
            {
                "source_doc_id": "nccn-v4-2025",
                "source_doc_title": "NCCN Guidelines Version 4.2025",
                "source_file_path": "docs/nccn.pdf",
                "source_page": 48,
                "source_paragraph_index": 3,
                "source_snippet": "Trastuzumab is indicated for HER2-positive breast cancer...",
                "provenance_id": "abc123",
            }
        ]
    }

    # minimal retriever
    # Build a minimal PropertyGraphIndex instance
    index = mod.PropertyGraphIndex(
        nodes=[],
        kg_extractors=[],
        property_graph_store=store,
        show_progress=False,
    )

    # LLM stub
    from llama_index.core.llms import LLM, ChatMessage

    class StubLLM(LLM):
        def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
            return "assistant: Use trastuzumab for HER2+ cases [1]"

        # implement abstract methods with simple stubs
        def complete(self, prompt: str, **kwargs: Any) -> str:  # type: ignore[override]
            return "assistant: stub"

        async def achat(self, messages: list[ChatMessage], **kwargs: Any):  # type: ignore[override]
            return "assistant: stub"

        def acomplete(self, prompt: str, **kwargs: Any):  # type: ignore[override]
            raise NotImplementedError

        def stream_complete(self, prompt: str, **kwargs: Any):  # type: ignore[override]
            yield "assistant: stub"

        def astream_complete(self, prompt: str, **kwargs: Any):  # type: ignore[override]
            raise NotImplementedError

        def astream_chat(self, messages: list[ChatMessage], **kwargs: Any):  # type: ignore[override]
            raise NotImplementedError

        def stream_chat(self, messages: list[ChatMessage], **kwargs: Any):  # type: ignore[override]
            yield "assistant: stub"

        @property
        def metadata(self):  # type: ignore[override]
            class _M:
                model_name = "stub"

            return _M()

    engine = mod.GraphRAGQueryEngine(
        graph_store=store, index=index, llm=StubLLM(), similarity_top_k=5
    )

    answer = engine.custom_query("How to treat HER2-positive breast cancer?")
    assert isinstance(answer, str)
    assert "trastuzumab" in answer.lower()
    assert "[1]" in answer
