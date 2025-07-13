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

import asyncio
from agents import (
    Agent,
    Runner
)
from loguru import logger

from bioagents.agents.base_agent import ReasoningAgent
from bioagents.agents.biomcp_agent import BioMCPAgent
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.common import AgentResponse, AgentRoute
from bioagents.models.llms import LLM
from bioagents.agents.web_agent import WebReasoningAgent

class BioConciergeAgent(ReasoningAgent):
    def __init__(
        self, name: str, 
        model_name: str=LLM.GPT_4_1_NANO, 
    ):
        instructions = (
            "You are a bio-reasoning agent that routes queries to appropriate specialists. "
            "You analyze the user's query and determine the best sub-agent to respond."
        )

        super().__init__(name, model_name, instructions)

        self._agent = self._create_agent()
        
        # Mapping from agent names to display names and routes
        self._agent_name_mapping = {
            "Chit Chat Agent": ("Chit Chat Agent", AgentRoute.CHIT_CHAT),
            "Web Reasoning Agent": ("Web Reasoning Agent", AgentRoute.WEB),
            "Bio MCP Agent": ("Bio MCP Agent", AgentRoute.BIO_MCP),
            "BiomedicalAssistant": ("Bio MCP Agent", AgentRoute.BIO_MCP),  # BioMCPAgent internal name
            "Bio Concierge": ("Bio Concierge Agent", AgentRoute.UNKNOWN),
        }

    def _create_agent(self):
        chit_chat_agent = ChitChatAgent(name="Chit Chat Agent")
        web_agent = WebReasoningAgent(name="Web Reasoning Agent")
        biomcp_agent = BioMCPAgent(name="Bio MCP Agent")
        
        return Agent(
            name="Bio Concierge",
            model=self.model_name,
            instructions=self.instructions,
            handoff_description=(
                "You are a bio-reasoning agent that routes queries to appropriate specialists.",
                "You analyze the user's query and determine the best sub-agent to respond.",
                "You will handoff to the appropriate sub-agent based on the user's query.",
                "You will then return the response from the sub-agent to the user.",
            ),
            handoffs=[biomcp_agent._agent, web_agent._agent, chit_chat_agent._agent]
        )

    async def achat(self, query_str: str) -> AgentResponse:
        print("[AGENT DEBUG] Entered achat")
        if self._agent is None:
            print("[AGENT DEBUG] Agent not initialized, raising ValueError")
            raise ValueError("Agent not initialized")
        
        try:
            from agents import gen_trace_id, trace
            trace_id = gen_trace_id()
            with trace(workflow_name="BioConciergeAgent", trace_id=trace_id):
                result = await asyncio.wait_for(
                    Runner.run(
                        starting_agent=self._agent,
                        input=query_str,
                        max_turns=3,
                    ),
                    timeout=self.timeout
                )
                print("[AGENT DEBUG] Runner.run completed")
                logger.info(f"{self.name}: {query_str} -> {trace_id}")
                # Determine which agent provided the final response
                responding_agent_name = "Bio Concierge"  # Default
                route = AgentRoute.UNKNOWN
                # Look through the conversation items to find the last message source
                for item in reversed(result.new_items):
                    if (item.type == 'message_output_item' and 
                        hasattr(item.raw_item, 'source') and 
                        item.raw_item.source):
                        responding_agent_name = item.raw_item.source
                        break
                # Map agent name to display name and route
                display_name, route = self._agent_name_mapping.get(
                    responding_agent_name, 
                    ("Bio Concierge Agent", AgentRoute.UNKNOWN)
                )
                logger.debug(f"[DEBUG] responding_agent_name={responding_agent_name}, display_name={display_name}, route={route}")
                print(f"[AGENT DEBUG] About to return AgentResponse with display_name={display_name}")
                # Prepend agent name to response if it's from a sub-agent
                response_text = result.final_output
                if not response_text.startswith(f"{display_name}:"):
                    response_text = f"{display_name}: {response_text}"
                # Construct response with citations and route information
                return self._construct_response_with_agent_info(
                    result, response_text, route
                )
            print("[AGENT DEBUG] Exiting try block")
        except Exception as e:
            logger.error(f"achat: {str(e)}")
            print(f"[AGENT EXCEPTION] {e}")
            raise e
        print("[AGENT DEBUG] Exiting achat with None")

    def _construct_response_with_agent_info(self, run_result, response_text, route):
        """
        Construct AgentResponse with agent-specific information and citations.
        """
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