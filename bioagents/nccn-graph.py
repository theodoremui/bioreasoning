"""
NCCN Breast Cancer Guidelines - Knowledge Graph Processing Script

This script is used to process the NCCN Breast Cancer Guidelines into a knowledge graph.

The script uses the LlamaIndex library to process the PDF document and build a knowledge graph.
The knowledge graph is then used to answer questions about the NCCN Breast Cancer Guidelines.

The script is also used to build the knowledge graph for the BioAgents application.

Author: Theodore Mui
Date: 2025-08-24
"""

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import asyncio
import os
import warnings

# Suppress deprecation warning from Setuptools about pkg_resources used in graspologic
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
)

from llama_index.core import PropertyGraphIndex
from llama_index.llms.openai import OpenAI

from bioagents.graph import (
    DocumentProcessor,
    GraphConfig,
    GraphRAGExtractor,
    GraphRAGQueryEngine,
    GraphRAGStore,
)

# Import refactored components
from bioagents.utils.spinner import ProgressSpinner as Spinner

# Initialize configuration from environment
config = GraphConfig.from_environment()

# Initialize LLM
llm = OpenAI(model=config.api.openai_model)


async def process_nccn_pdf(
    pdf_path: str = "data/nccn_breast_cancer.pdf", graph_store: GraphRAGStore = None
) -> PropertyGraphIndex:
    """Process NCCN PDF document and build knowledge graph.

    Args:
        pdf_path: Path to the PDF document
        graph_store: Graph store instance

    Returns:
        PropertyGraphIndex with extracted knowledge
    """

    # Initialize document processor
    with Spinner("Initializing document processor"):
        if not config.api.llamacloud_api_key:
            raise ValueError("LLAMACLOUD_API_KEY is required for document processing")

        processor = DocumentProcessor(
            api_key=config.api.llamacloud_api_key,
            num_workers=config.performance.pdf_parse_workers,
            verbose=False,
            language="en",
        )

    # Process document
    with Spinner("Processing PDF document"):
        nodes = await processor.process_document(pdf_path)

    if not nodes:
        print("✗ Failed to process PDF or no content extracted")
        return None

    # Initialize knowledge extractor
    with Spinner("Initializing knowledge extractor"):
        extractor = GraphRAGExtractor(
            llm=llm,
            max_paths_per_chunk=config.performance.max_paths_per_chunk,
            num_workers=config.performance.kg_extraction_workers,
        )

        # Build knowledge graph
        with Spinner("Building knowledge graph"):
            index = PropertyGraphIndex(
                nodes=nodes,
                kg_extractors=[extractor],
                property_graph_store=graph_store,
                show_progress=False,  # Disable built-in progress
            )

    # Build communities
    persist_path = os.path.join(
        os.path.dirname(__file__), "..", config.cache.communities_cache_file
    )
    with Spinner("Building communities"):
        graph_store.ensure_communities(persist_path=persist_path)

    return index


async def load_index(graph_store: GraphRAGStore) -> PropertyGraphIndex:
    """Load index from existing graph store.

    Args:
        graph_store: Graph store instance

    Returns:
        PropertyGraphIndex loaded from store
    """
    try:
        with Spinner("Loading graph index"):
            index = PropertyGraphIndex(
                nodes=[],
                kg_extractors=[],
                property_graph_store=graph_store,
                show_progress=False,
            )

        # Load communities from cache (fast initialization - skip expensive summaries)
        persist_path = os.path.join(
            os.path.dirname(__file__), "..", config.cache.communities_cache_file
        )
        with Spinner("Loading communities from cache"):
            graph_store.ensure_communities(
                persist_path=persist_path,
                validate_signature=False,  # Skip expensive validation for faster startup
                skip_summaries=True,  # Skip expensive LLM calls during initialization
            )

    except Exception as e:
        print(f"✗ Error loading index: {e}")
        return None

    return index


async def main():
    """Main application entry point."""

    # Display configuration
    print(config.display_summary())
    print()

    # Initialize graph store (with lazy Neo4j connection)
    with Spinner("Initializing graph store"):
        graph_store = GraphRAGStore(
            username=config.database.neo4j_username,
            password=config.database.neo4j_password,
            url=config.database.neo4j_url,
            max_cluster_size=config.performance.max_cluster_size,
            lazy_connection=True,  # Defer Neo4j connection until needed
        )

    # Choose processing mode
    process_pdf = False  # Set to True to process new PDF

    if process_pdf:
        pdf_path = "../data/nccn_breast_cancer.pdf"
        index = await process_nccn_pdf(pdf_path, graph_store)
    else:
        index = await load_index(graph_store)

    if not index:
        print("✗ Failed to initialize index")
        return

    # Setup query engine
    with Spinner("Setting up query engine"):
        query_engine = GraphRAGQueryEngine(
            graph_store=graph_store,
            llm=llm,
            index=index,
            similarity_top_k=config.query.similarity_top_k,
            max_summaries_to_use=config.query.max_summaries_to_use,
            max_triplets_to_use=config.query.max_triplets_to_use,
        )

    print("\n🎯 GraphRAG Query Engine Ready!")
    print("=" * 50)

    # Interactive query loop
    query = "How best to treat breast cancer for patients with HER2?"
    while query.strip() != "":
        print(f"\n📝 Query: {query}")

        with Spinner("Processing query"):
            response = query_engine.query(query)

        print(f"\n📋 Response:")
        print(response)
        print("\n" + "=" * 50)

        query = input("\nEnter your query (or press Enter to exit): ")

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()

    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise
