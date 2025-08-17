#------------------------------------------------------------------------------
# llamarag_agent.py
# 
# This agent is an LlamaCloud RAG agent that can query the LlamaCloud index.
# 
# Author: Theodore Mui
# Date: 2025-08-16
#------------------------------------------------------------------------------

import asyncio
import os
from typing import List, override
from loguru import logger

from agents import Agent, Runner, function_tool
from agents.model_settings import ModelSettings
from agents.tracing import set_tracing_disabled
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore

from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM
from bioagents.agents.base_agent import BaseAgent
from datetime import timedelta

set_tracing_disabled(disabled=True)

INSTRUCTIONS = f"""\
You are an LlamaCloud RAG agent that can query documents and knowledge about Breast Cancer guidelines, 
including:

- National Comprehensive Cancer Network (NCCN) clinical practice guildelines in oncology
- Ductal Carcinoma In Situ (DCIS) diagnosis, workup, primary treatment, postsurgical treatment, and follow-up
- Invasive Breast Cancer diagnosis, workup, clinical stage assessment, surgery,
   histology, HR status, HER2 status, systemic adjuvant treatment, and follow-up

You should always directly answer the user's question, without asking for permission, any preambles.
Your response should include relevant citation information from the source documents.\n

## Response Instructions:
- Prepend the response with '[RAG]'
"""

HANDOFF_DESCRIPTION = f"""\
You are an LlamaCloud RAG agent that can query documents and knowledge about NCCN Breast Cancer guidelines
"""

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8131/mcp/")

LLAMACLOUD_PROJECT_NAME = os.getenv("LLAMACLOUD_PROJECT_NAME")
LLAMACLOUD_ORGANIZATION_ID = os.getenv("LLAMACLOUD_ORGANIZATION_ID")
LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")
LLAMACLOUD_INDEX_NAME = os.getenv("LLAMACLOUD_INDEX_NAME")
# LLAMACLOUD_PIPELINE_ID = os.getenv("LLAMACLOUD_PIPELINE_ID")

@function_tool()
def query_index(query: str) -> str:
    """Query the documents and knowledge in an LlamaCloud index."""
    
    try:
        index = LlamaCloudIndex(
            name=LLAMACLOUD_INDEX_NAME,
            project_name=LLAMACLOUD_PROJECT_NAME,
            organization_id=LLAMACLOUD_ORGANIZATION_ID,
            # pipeline_id=LLAMACLOUD_PIPELINE_ID,
            api_key=LLAMACLOUD_API_KEY,
        )

        # nodes: List[NodeWithScore] = index.as_retriever().retrieve(query)
        response = index.as_query_engine().query(query)
        if isinstance(response, Response):
            return response.response
        else:
            return str(response)
    except Exception as e:
        logger.error(f"Error querying LlamaCloud index: {e}")
        return f"Error querying LlamaCloud index: {e}"

class LlamaRAGAgent(BaseAgent):
    """
    This agent is an LlamaCloud RAG agent that can query the LlamaCloud index.
    """
    def __init__(
        self, name: str, 
        model_name: str=LLM.GPT_4_1_MINI, 
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
            tools=[query_index],
            model_settings=ModelSettings(tool_choice="required"),
        )
    
    @override
    async def achat(self, query_str: str) -> AgentResponse:
        logger.info(f"=> llamamcp: {self.name}: {query_str}")

        try:
            result = await Runner.run(starting_agent=self._agent, input=query_str)
            return self._construct_response(result, "", AgentRouteType.LLAMARAG)
        except Exception as e:
            logger.error(f"RAG agent failed: {e}")
            return AgentResponse(
                response_str=f"[LlamaCloud RAG] RAG agent not working.",
                route=AgentRouteType.LLAMARAG,
            )

    
#------------------------------------------------
# Example usage
#------------------------------------------------
async def smoke_tests() -> None:
    try:
        print("==> 1")
        agent = LlamaRAGAgent(name="LlamaCloud RAG Agent")
        print("==> 2")
        # print(str(await agent.achat("What are the best treatment for patients with HER2 genes?")))
        print(str(await agent.achat("What are the key features in ICD-10?")))
        print("==> 3")
        print(str(await agent.achat("What are the top 10 United Nations climate mandates?")))
        print("==> 4")
    finally:
        print("==> 5")

if __name__ == "__main__":
    asyncio.run(smoke_tests())
