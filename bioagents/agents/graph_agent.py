# ------------------------------------------------------------------------------
# graph_agent.py
#
# A knowledge graph agent that can answer questions about the NCCN breast cancer
# guidelines knowledge graph using GraphRAG query engine.
# Provides a simple async chat interface with integrated knowledge graph querying.
#
# This agent follows SOLID principles and uses dependency injection for
# configuration, graph store, index, and query engine components.
#
# Example:
#     # Simple usage with default configuration
#     agent = GraphAgent(name="Graph Agent")
#     result = await agent.achat("What is the recommended treatment for breast cancer?")
#     print(result)
#
#     # Advanced usage with custom configuration
#     config = GraphConfig.from_environment()
#     agent = GraphAgent(name="Graph Agent", config=config)
#     async with agent:
#         result = await agent.achat("How to treat HER2-positive breast cancer?")
#         print(result)
#
# Author: Theodore Mui
# Date: 2025-08-16
# ------------------------------------------------------------------------------

import asyncio
import os
from typing import Optional, override

from agents import Agent, ModelSettings, function_tool
from agents.tracing import set_tracing_disabled
from llama_index.core import PropertyGraphIndex
from llama_index.llms.openai import OpenAI
from loguru import logger

set_tracing_disabled(disabled=True)

from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.commons import classproperty
from bioagents.graph import (
    DocumentProcessor,
    GraphConfig,
    GraphRAGExtractor,
    GraphRAGQueryEngine,
    GraphRAGStore,
)
from bioagents.models.llms import LLM
from bioagents.models.source import Source
from bioagents.utils.spinner import ProgressSpinner as Spinner


INSTRUCTIONS = f"""\
You are an expert that can answer questions about the NCCN breast cancer guidelines knowledge graph.
You should always directly answer the user's question, without asking for permission, any preambles.
Your response should include relevant citation information from the source documents.\n

## Response Instructions:
- Prepend the response with '[Graph]'
"""

HANDOFF_DESCRIPTION = (
    "You are an expert that can answer questions about the NCCN "
    "breast cancer guidelines knowledge graph."
)


class GraphAgent(BaseAgent):
    """
    Knowledge graph agent for NCCN breast cancer guidelines.
    
    This agent provides expert knowledge about breast cancer treatment guidelines
    using a GraphRAG query engine. It follows SOLID principles with dependency
    injection for configuration and components.
    
    Features:
        - GraphRAG-powered knowledge retrieval
        - Citation generation with provenance
        - Configurable query parameters
        - Robust error handling and fallbacks
        - Async lifecycle management
    
    Architecture:
        - Uses GraphConfig for configuration management
        - Integrates GraphRAGStore for graph storage
        - Leverages GraphRAGQueryEngine for intelligent querying
        - Provides tool-based interface for agent framework
    
    Example:
        agent = GraphAgent(name="Graph Expert")
        async with agent:
            response = await agent.achat("How to treat HER2-positive breast cancer?")
            print(response.response_str)
            for citation in response.citations:
                print(f"Source: {citation.title}")
    """

    _query_engine: Optional[GraphRAGQueryEngine] = None

    @classproperty
    def query_engine(cls) -> Optional[GraphRAGQueryEngine]:
        """Get the shared query engine instance."""
        return cls._query_engine

    @classmethod
    def set_query_engine(cls, query_engine: GraphRAGQueryEngine) -> None:
        """Set the shared query engine instance."""
        cls._query_engine = query_engine

    def __init__(
        self,
        name: str,
        model_name: str = LLM.GPT_4_1_NANO,
        config: Optional[GraphConfig] = None,
        graph_store: Optional[GraphRAGStore] = None,
        index: Optional[PropertyGraphIndex] = None,
        query_engine: Optional[GraphRAGQueryEngine] = None,
        pdf_path: str = "data/nccn_breast_cancer.pdf",
        auto_initialize: bool = True,
    ):
        """Initialize GraphAgent with optional dependency injection.
        
        Args:
            name: Agent name for identification
            model_name: LLM model to use for responses
            config: Graph configuration (defaults to environment-based config)
            graph_store: Graph storage instance (created if not provided)
            index: Property graph index (created if not provided)
            query_engine: Query engine instance (created if not provided)
            pdf_path: Path to PDF document for processing
            auto_initialize: Whether to initialize components automatically
        """
        self.instructions = INSTRUCTIONS
        self.handoff_description = HANDOFF_DESCRIPTION
        
        super().__init__(name, model_name, self.instructions)
        
        # Configuration and components (following dependency injection)
        self.config = config or GraphConfig.from_environment()
        self.pdf_path = pdf_path
        self.auto_initialize = auto_initialize
        
        # Core components (will be initialized in start())
        self._graph_store = graph_store
        self._index = index
        self._query_engine = query_engine
        self._llm: Optional[OpenAI] = None
        
        # Set class-level query engine if provided
        if query_engine:
            GraphAgent.set_query_engine(query_engine)
        
        # State management
        self._started: bool = False
        
        # Create agent with graph query tool
        self._agent: Optional[Agent] = None

    @staticmethod
    @function_tool()
    def query_knowledge_graph(query: str) -> AgentResponse:
        """Query the NCCN breast cancer guidelines knowledge graph.
        
        This tool searches the knowledge graph for information related to
        breast cancer treatment guidelines, providing evidence-based responses
        with citations and provenance.
        
        Args:
            query: The question or search query about breast cancer guidelines
            
        Returns:
            AgentResponse with answer and citations
        """
        try:
            logger.info(f"Querying knowledge graph: {query}")
            
            # Query the graph engine (now returns clean answer without citations)
            response_text = GraphAgent._query_engine.query(query)
            
            # Extract structured citations from the query engine
            citations = []
            try:
                raw_citations = GraphAgent._query_engine.get_last_citations()
                for citation in raw_citations:
                    # Skip None citations
                    if citation is None:
                        continue
                    
                    # Skip non-dict citations
                    if not isinstance(citation, dict):
                        continue
                        
                    try:
                        source = Source(
                            title=citation.get("title", "NCCN Guidelines"),
                            snippet=citation.get("snippet", ""),
                            source="knowledge_graph",
                            file_name=citation.get("file_name", "nccn_breast_cancer.pdf"),
                            start_page_label=citation.get("start_page", ""),
                            end_page_label=citation.get("end_page", ""),
                            score=citation.get("score", 0.0),
                            text=citation.get("text", ""),
                        )
                        citations.append(source)
                    except Exception as source_error:
                        # Log individual citation processing errors but continue
                        logger.debug(f"Skipping malformed citation: {source_error}")
                        continue
            except Exception as citation_error:
                logger.warning(f"Could not extract citations: {citation_error}")
            
            return AgentResponse(
                response_str=f"[Graph] {response_text}",
                citations=citations,
                route=AgentRouteType.GRAPH,
            )
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return AgentResponse(
                response_str=f"[Graph] Sorry, I encountered an error querying the knowledge graph: {str(e)}",
                route=AgentRouteType.GRAPH,
            )

    def _create_agent(self) -> Agent:
        """Create the core Agent with graph query tool."""
        if not GraphAgent._query_engine:
            raise ValueError("Query engine must be initialized before creating agent")
        
        return Agent(
            name=self.name,
            model=self.model_name,
            instructions=self.instructions,
            handoff_description=self.handoff_description,
            tools=[GraphAgent.query_knowledge_graph],
            model_settings=ModelSettings(
                tool_choice="required",
                temperature=0.01,
                top_p=1.0,
            ),
            tool_use_behavior="stop_on_first_tool",
            output_type=AgentResponse,
        )

    async def _initialize_components(self) -> None:
        """Initialize graph components if not provided via dependency injection."""
            
        try:
            # Initialize LLM
            if not self._llm:
                self._llm = OpenAI(model=self.config.api.openai_model)
            
            # Initialize graph store
            if not self._graph_store:
                with Spinner("Initializing graph store"):
                    self._graph_store = GraphRAGStore(
                        username=self.config.database.neo4j_username,
                        password=self.config.database.neo4j_password,
                        url=self.config.database.neo4j_url,
                        max_cluster_size=self.config.performance.max_cluster_size,
                        lazy_connection=True,
                    )
            
            # Initialize index - try loading first, then process if needed
            if not self._index:
                self._index = await self._load_or_create_index()
            
            # Initialize query engine
            if not self._query_engine:
                with Spinner("Setting up query engine"):
                    self._query_engine = GraphRAGQueryEngine(
                        graph_store=self._graph_store,
                        llm=self._llm,
                        index=self._index,
                        similarity_top_k=self.config.query.similarity_top_k,
                        max_summaries_to_use=self.config.query.max_summaries_to_use,
                        max_triplets_to_use=self.config.query.max_triplets_to_use,
                    )
                    # Set the class-level query engine
                    GraphAgent.set_query_engine(self._query_engine)
            
            logger.info("Graph agent components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize graph components: {e}")
            raise

    async def _load_or_create_index(self) -> PropertyGraphIndex:
        """Load existing index or create new one from PDF."""
        try:
            # Try loading existing index first
            with Spinner("Loading graph index"):
                index = PropertyGraphIndex(
                    nodes=[],
                    kg_extractors=[],
                    property_graph_store=self._graph_store,
                    show_progress=False,
                )
            
            # Load communities from cache
            persist_path = os.path.join(
                os.path.dirname(__file__), "..", "..", self.config.cache.communities_cache_file
            )
            
            if os.path.exists(persist_path):
                with Spinner("Loading communities from cache"):
                    self._graph_store.ensure_communities(
                        persist_path=persist_path,
                        validate_signature=False,
                        skip_summaries=True,
                    )
                logger.info("Loaded existing graph index and communities")
                return index
            else:
                logger.info("No existing communities found, will process PDF")
                return await self._process_pdf_and_build_index()
                
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            return await self._process_pdf_and_build_index()

    async def _process_pdf_and_build_index(self) -> PropertyGraphIndex:
        """Process PDF document and build knowledge graph index."""
        # Initialize document processor
        with Spinner("Initializing document processor"):
            if not self.config.api.llamacloud_api_key:
                raise ValueError("LLAMACLOUD_API_KEY is required for document processing")

            processor = DocumentProcessor(
                api_key=self.config.api.llamacloud_api_key,
                num_workers=self.config.performance.pdf_parse_workers,
                verbose=False,
                language="en",
            )

        # Process document
        with Spinner("Processing PDF document"):
            nodes = await processor.process_document(self.pdf_path)

        if not nodes:
            raise ValueError("Failed to process PDF or no content extracted")

        # Initialize knowledge extractor
        with Spinner("Initializing knowledge extractor"):
            extractor = GraphRAGExtractor(
                llm=self._llm,
                max_paths_per_chunk=self.config.performance.max_paths_per_chunk,
                num_workers=self.config.performance.kg_extraction_workers,
            )

        # Build knowledge graph
        with Spinner("Building knowledge graph"):
            index = PropertyGraphIndex(
                nodes=nodes,
                kg_extractors=[extractor],
                property_graph_store=self._graph_store,
                show_progress=False,
            )

        # Build communities
        persist_path = os.path.join(
            os.path.dirname(__file__), "..", "..", self.config.cache.communities_cache_file
        )
        with Spinner("Building communities"):
            self._graph_store.ensure_communities(persist_path=persist_path)

        logger.info("Successfully built knowledge graph from PDF")
        return index

    async def start(self) -> None:
        """Start the graph agent and initialize all components."""
        if self._started:
            return
            
        try:
            if self.auto_initialize:
                await self._initialize_components()
            
            # Create agent with initialized components
            if not self._agent:
                self._agent = self._create_agent()
                
            self._started = True
            logger.info(f"GraphAgent '{self.name}' started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start GraphAgent: {e}")
            raise

    async def aclose(self) -> None:
        """Cleanup resources."""
        self._started = False
        logger.info(f"GraphAgent '{self.name}' closed")

    async def stop(self) -> None:
        """Stop the graph agent."""
        await self.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit."""
        await self.stop()

    @override
    async def achat(self, query_str: str) -> AgentResponse:
        """Process a chat query using the knowledge graph.
        
        Args:
            query_str: The user's question about breast cancer guidelines
            
        Returns:
            AgentResponse with answer and citations from knowledge graph
        """
        logger.info(f"=> graph: {self.name}: {query_str}")
        
        try:
            # Ensure agent is started
            if not self._started:
                await self.start()
            
            if self._agent is None:
                raise ValueError("Agent not properly initialized")

            # Use parent's achat which will call our tool
            response = await super().achat(query_str)
            response.route = AgentRouteType.GRAPH
            return response
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return AgentResponse(
                response_str=f"[Graph] Sorry, I encountered an error: {str(e)}",
                route=AgentRouteType.GRAPH,
            )

    # Properties for accessing components (useful for testing and debugging)
    @property
    def config(self) -> GraphConfig:
        """Get the graph configuration."""
        return self._config
    
    @config.setter
    def config(self, value: GraphConfig) -> None:
        """Set the graph configuration."""
        self._config = value
    
    @property
    def graph_store(self) -> Optional[GraphRAGStore]:
        """Get the graph store instance."""
        return self._graph_store
    
    @property
    def index(self) -> Optional[PropertyGraphIndex]:
        """Get the property graph index."""
        return self._index
    
    @property
    def query_engine(self) -> Optional[GraphRAGQueryEngine]:
        """Get the query engine instance."""
        return self._query_engine

# ------------------------------------------------
# Example usage and smoke tests
# ------------------------------------------------
async def smoke_tests() -> None:
    """Comprehensive smoke tests for GraphAgent functionality."""
    print("🧪 Starting GraphAgent smoke tests...")
    
    try:
        print("==> 1. Creating GraphAgent")
        agent = GraphAgent(name="Graph Agent Test")
        
        print("==> 2. Starting agent (this may take a while for first run)")
        await agent.start()
        
        print("==> 3. Testing basic medical query")
        response1 = await agent.achat("What is ICD-10?")
        print(f"Response 1: {response1.response_str[:200]}...")
        print(f"Citations: {len(response1.citations)}")
        
        print("==> 4. Testing breast cancer treatment query")
        response2 = await agent.achat(
            "What is the recommended treatment for breast cancer?"
        )
        print(f"Response 2: {response2.response_str[:200]}...")
        print(f"Citations: {len(response2.citations)}")
        
        print("==> 5. Testing HER2-specific query")
        response3 = await agent.achat(
            "How should HER2-positive breast cancer be treated?"
        )
        print(f"Response 3: {response3.response_str[:200]}...")
        print(f"Citations: {len(response3.citations)}")
        
        print("==> 6. Testing agent properties")
        print(f"Config loaded: {agent.config is not None}")
        print(f"Graph store initialized: {agent.graph_store is not None}")
        print(f"Index loaded: {agent.index is not None}")
        print(f"Query engine ready: {agent.query_engine is not None}")
        
        print("✅ All smoke tests passed!")
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        raise
    finally:
        print("==> 7. Cleaning up")
        await agent.stop()


if __name__ == "__main__":
    import nest_asyncio
    
    # Allow nested event loops for Jupyter compatibility
    nest_asyncio.apply()
    
    try:
        asyncio.run(smoke_tests())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # Handle Jupyter/existing event loop
            loop = asyncio.get_event_loop()
            loop.run_until_complete(smoke_tests())
        else:
            raise
