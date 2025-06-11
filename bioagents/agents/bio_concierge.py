#------------------------------------------------------------------------------
# bioreasoner.py
# 
# This is a "Bio Reasoning Agent" that triage across multiple agents to answer
# a user's question.  This agent orchestrates across the following subagents:
# 
# 1. Chit Chat Agent
# 2. Web Reasoning Agent
# 
# Author: Theodore Mui
# Date: 2025-04-26
#------------------------------------------------------------------------------

from agents import (
    Agent,
    Runner
)
from loguru import logger

from bioagents.agents.base_agent import ReasoningAgent
from bioagents.agents.biomcp_agent import BioMCPAgent
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.common import AgentResponse
from bioagents.models.llms import LLM
from bioagents.agents.web_agent import WebReasoningAgent

class BioConciergeAgent(ReasoningAgent):
    def __init__(
        self, name: str, 
        model_name: str=LLM.GPT_4_1_NANO, 
    ):
        instructions = (
            "You are a bio-reasoning agent that routes queries to appropriate specialists. "
            "You analyze the user's query and determine the best way to respond."
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