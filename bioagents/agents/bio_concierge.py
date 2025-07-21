#------------------------------------------------------------------------------
# bio_concierge.py
# 
# This is a "Bio Reasoning Concierge" that triage across multiple agents to answer
# a user's question.  This agent orchestrates across the following subagents:
# 
# 1. Chit Chat Agent
# 2. Web Reasoning Agent
# 3. Bio MCP Agent
# 
# Author: Theodore Mui
# Date: 2025-04-26
#------------------------------------------------------------------------------

from agents import (
    Agent,
    Runner
)
from loguru import logger
from typing import override

from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.biomcp_agent import BioMCPAgent
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM
from bioagents.agents.web_agent import WebReasoningAgent

class BioConciergeAgent(BaseAgent):
    def __init__(
        self, name: str, 
        model_name: str=LLM.GPT_4_1_NANO, 
    ):
        instructions = (
            "You are a bio-reasoning agent that routes queries to appropriate specialists. "
            "You analyze the user's query and determine the best way to respond."
            "\n## Response Instructions:\n"
            "- Prepend the response with '[Concierge]'\n"
        )

        super().__init__(name, model_name, instructions)

        self._agent = self._create_agent()

    def _create_agent(self):
        chit_chat_agent = ChitChatAgent(name="Chit Chat Agent")
        web_agent = WebReasoningAgent(name="Web Reasoning Agent")
        biomcp_agent = BioMCPAgent(name="Bio MCP Agent")
        
        return Agent(
            name="Bio Concierge",
            model=self.model_name,
            instructions=self.instructions,
            handoffs=[biomcp_agent._agent, web_agent._agent, chit_chat_agent._agent]
        )

    def _construct_response_with_agent_info(self, run_result, response_text: str, route: AgentRouteType) -> AgentResponse:
        """Construct response with agent-specific information and routing."""
        # Extract citations from run_result
        citations = []
        for item in run_result.new_items:
            if item.type == 'message_output_item':
                for content in item.raw_item.content:
                    if hasattr(content, 'annotations'):
                        for annotation in content.annotations:
                            if annotation.type == 'url_citation':
                                from bioagents.models.citation import Citation
                                citations.append(Citation(
                                    url=annotation.url,
                                    title=annotation.title,
                                    snippet=content.text[annotation.start_index:annotation.end_index],
                                    source="web"
                                ))
        
        return AgentResponse(
            response_str=response_text,
            citations=citations,
            judge_response="",
            route=route
        )

    @override
    async def achat(self, query_str: str) -> AgentResponse:
        logger.info(f"=> concierge: {self.name}: {query_str}")

        response = await super().achat(query_str)
        response.route = AgentRouteType.CONCIERGE
        return response

#------------------------------------------------
# Example usage
#------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import time
    
    start_time = time.time()
    agent = BioConciergeAgent(name="Bio")
    response = asyncio.run(agent.achat("How are you?"))
    print(f"{str(response)} ({time.time() - start_time:.1f}s)")
    
    start_time = time.time()
    response = asyncio.run(agent.achat("What is the latest news in the field of genetics?"))
    print(f"{str(response)} ({time.time() - start_time:.1f}s)")

    start_time = time.time()
    response = asyncio.run(agent.achat("How is the weather in Tokyo?"))
    print(f"{str(response)} ({time.time() - start_time:.1f}s)")