# ------------------------------------------------------------------------------
# llamarag_agent.py
#
# This agent is an LlamaCloud RAG agent that can query the LlamaCloud index.
#
# Author: Theodore Mui
# Date: 2025-08-16
# ------------------------------------------------------------------------------

import asyncio
import hashlib
import os
from datetime import datetime
from typing import Any, List, Optional, override

from agents import Agent, Runner, function_tool
from agents.model_settings import ModelSettings
from agents.tracing import set_tracing_disabled
from llama_index.core.base.response.schema import Response
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from loguru import logger
from pydantic import BaseModel

from llama_index.postprocessor.cohere_rerank import CohereRerank

from bioagents.commons import classproperty
from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM
from bioagents.models.source import Source
from bioagents.utils.text_utils import make_title_and_snippet

set_tracing_disabled(disabled=True)

# Cohere reranker settings (model alt: rerank-english-v3.0)
COHERE_RERANKER_MODEL = os.getenv("COHERE_RERANKER_MODEL", "rerank-v3.5")
COHERE_RERANK_TOP_N = int(os.getenv("COHERE_RERANK_TOP_N", "6"))
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

DOCMCP_URL = os.getenv("DOCMCP_SERVER_URL", "http://localhost:8130/mcp/")

LLAMACLOUD_PROJECT_NAME = os.getenv("LLAMACLOUD_PROJECT_NAME")
LLAMACLOUD_ORG_ID = os.getenv("LLAMACLOUD_ORG_ID")
LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")
LLAMACLOUD_INDEX_NAME = os.getenv("LLAMACLOUD_INDEX_NAME")


INSTRUCTIONS = f"""\
You are an LlamaCloud RAG agent that can query documents and knowledge about Breast Cancer guidelines, 
including:

- National Comprehensive Cancer Network (NCCN) clinical practice guildelines in oncology
- Ductal Carcinoma In Situ (DCIS) diagnosis, workup, primary treatment, postsurgical treatment, and follow-up
- Invasive Breast Cancer diagnosis, workup, clinical stage assessment, surgery,
   histology, HR status, HER2 status, systemic adjuvant treatment, and follow-up

You should always directly answer the user's question thoroughly in a well structured manner.
You do not need to ask for permission, nor should you include any preambles.
Your response should include relevant citation information from the source documents.\n

## Response Instructions:
- Prepend the response with '[RAG]'
- Respond in a well structured Markdown format with proper headings and subheadings.
- Use bold text for important terms and phrases.

Today's date: {datetime.now().strftime("%Y-%m-%d")}
"""

HANDOFF_DESCRIPTION = f"""\
You are an LlamaCloud RAG agent that can query documents and knowledge about NCCN Breast Cancer guidelines
"""


class LlamaRAGAgent(BaseAgent):
    """
    This agent is an LlamaCloud RAG agent that can query the LlamaCloud index.
    """

    _index: Optional[LlamaCloudIndex] = None
    _query_engine: Optional[Any] = None
    _reranker: Optional[CohereRerank] = None

    @classproperty
    def reranker(cls) -> Optional[CohereRerank]:
        if cls._reranker is None and CohereRerank is not None:
            try:
                if COHERE_API_KEY:
                    cls._reranker = CohereRerank(
                        model=COHERE_RERANKER_MODEL, top_n=COHERE_RERANK_TOP_N
                    )
            except Exception as e:
                logger.warning(f"Reranker initialization skipped: {e}")
        return cls._reranker

    @classproperty
    def index(cls) -> LlamaCloudIndex:
        if cls._index is None:
            if not LLAMACLOUD_INDEX_NAME or not LLAMACLOUD_API_KEY:
                raise ValueError(
                    "LlamaCloud index issue: check LLAMACLOUD_INDEX_NAME LLAMACLOUD_API_KEY."
                )
            cls._index = LlamaCloudIndex(
                name=LLAMACLOUD_INDEX_NAME,
                project_name=LLAMACLOUD_PROJECT_NAME,
                organization_id=LLAMACLOUD_ORG_ID,
                api_key=LLAMACLOUD_API_KEY,
            )
        return cls._index

    @classproperty
    def query_engine(cls):
        if cls._query_engine is None:
            post_processors = [cls.reranker] if cls.reranker is not None else []
            cls._query_engine = cls.index.as_query_engine(
                similarity_top_k=10,
                node_postprocessors=post_processors,
            )
        return cls._query_engine

    @staticmethod
    @function_tool()
    def query_index(query: str) -> AgentResponse:
        """Query the documents and knowledge in a LlamaCloud index."""
        try:
            response = LlamaRAGAgent.query_engine.query(query)

            sources = []
            seen_text_hashes = set()
            for source in response.source_nodes:
                # Deduplicate by exact text content
                text_value = source.text or ""
                text_hash = hashlib.sha256(
                    text_value.encode("utf-8", errors="ignore")
                ).hexdigest()
                if text_hash in seen_text_hashes:
                    continue
                seen_text_hashes.add(text_hash)
                title, snippet = make_title_and_snippet(
                    text=source.text,
                    query=query,
                    max_length=300,
                )
                src = Source(
                    title=title,
                    snippet=snippet,
                    source=(
                        source.metadata["file_name"]
                        if "file_name" in source.metadata
                        else ""
                    ),
                    file_name=(
                        source.metadata["file_name"]
                        if "file_name" in source.metadata
                        else ""
                    ),
                    start_page_label=(
                        str(source.metadata["start_page_label"])
                        if "start_page_label" in source.metadata
                        else ""
                    ),
                    end_page_label=(
                        str(source.metadata["end_page_label"])
                        if "end_page_label" in source.metadata
                        else ""
                    ),
                    score=source.score,
                    text=source.text,
                )
                sources.append(src)

            if isinstance(response, Response):
                return AgentResponse(
                    response_str=response.response,
                    citations=sources,
                    judgement="",
                    route=AgentRouteType.LLAMARAG,
                )
            return AgentResponse(
                response_str=str(response),
                citations=[],
                judgement="",
                route=AgentRouteType.LLAMARAG,
            )
        except Exception as e:
            logger.error(f"Error querying LlamaCloud index: {e}")
            return f"Error querying LlamaCloud index: {e}"

    def __init__(
        self,
        name: str,
        model_name: str = LLM.GPT_4_1_MINI,
    ):
        self.instructions = INSTRUCTIONS
        self.handoff_description = HANDOFF_DESCRIPTION

        super().__init__(name, model_name, self.instructions)
        self._agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create the core RAG Agent."""
        return Agent(
            name=self.name,
            model=self.model_name,
            instructions=self.instructions,
            handoff_description=self.handoff_description,
            tools=[LlamaRAGAgent.query_index],
            model_settings=ModelSettings(
                tool_choice="required",
            ),
            tool_use_behavior="stop_on_first_tool",
            output_type=AgentResponse,
        )

    @override
    async def achat(self, query_str: str) -> AgentResponse:
        logger.info(f"=> llamarag: {self.name}: {query_str}")

        response = await super().achat(query_str)
        response.route = AgentRouteType.LLAMARAG
        return response


# ------------------------------------------------
# Example usage
# ------------------------------------------------
async def smoke_tests() -> None:
    try:
        print("==> 1")
        agent = LlamaRAGAgent(name="LlamaCloud RAG Agent")
        print("==> 2")
        # print(str(await agent.achat("What are the best treatment for patients with HER2 genes?")))
        print(str(await agent.achat("What are the key features in ICD-10?")))
        print("==> 3")
        print(
            str(
                await agent.achat(
                    "What are the top 10 United Nations climate mandates?"
                )
            )
        )
        print("==> 4")
    finally:
        print("==> 5")


if __name__ == "__main__":
    asyncio.run(smoke_tests())
