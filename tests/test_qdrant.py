import os
import pytest
from qdrant_client import QdrantClient

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env if present
load_dotenv(find_dotenv())

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


@pytest.fixture(scope="module")
def qdrant_client() -> QdrantClient:
    """Provide a QdrantClient if environment is configured; otherwise skip.

    Skips the test module when QDRANT_URL is not set or connection fails.
    """
    if not QDRANT_URL:
        pytest.skip("QDRANT_URL is not set; skipping Qdrant integration tests")

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Lightweight sanity check: attempt to fetch collections; if unreachable, skip
        _ = client.get_collections()
        return client
    except Exception as exc:
        pytest.skip(f"Could not connect to Qdrant at {QDRANT_URL}: {exc}")


@pytest.mark.integration
def test_get_collections_non_empty(qdrant_client: QdrantClient) -> None:
    """Ensure at least one collection exists on the target Qdrant instance."""
    resp = qdrant_client.get_collections()
    # Support both SDK response shapes
    collections = resp.collections if hasattr(resp, "collections") else resp

    assert isinstance(collections, list), "Expected a list of collections"
    assert (
        len(collections) > 0
    ), "Qdrant has no collections; create one before running this test"
