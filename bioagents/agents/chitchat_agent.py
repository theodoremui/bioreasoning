#------------------------------------------------------------------------------
# chitchat_agent.py
# 
# This agent is a friendly conversational assistant that informally chit chats with the user.
# 
# Author: Theodore Mui
# Date: 2025-04-26
#------------------------------------------------------------------------------

from agents import Agent
from typing import override
from loguru import logger

from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM
from bioagents.agents.base_agent import BaseAgent

class ChitChatAgent(BaseAgent):
    """
    This agent is a friendly conversational assistant that informally chit chats with the user.
    """
    def __init__(
        self, name: str, 
        model_name: str=LLM.GPT_4_1_NANO, 
    ):
        instructions = (
            "You are a friendly conversational assistant but you should be very brief and to the point."
            "You should AVOID asking the user any question."
            "\n## Response Instructions:\n"
            "- Prepend the response with '[Chit Chat]'\n"
        )

        super().__init__(name, model_name, instructions)
        self._agent = self._create_agent(name, model_name)

    def _create_agent(self, agent_name: str, model_name: str=LLM.GPT_4_1_NANO):
        agent = Agent(
            name=agent_name,
            model=model_name,
            instructions=self.instructions,
            handoff_description=(
                "You are a friendly conversational assistant to chit chat with the user "
                "and your responses should be very brief and to the point."
            ),
            handoffs=[],
            tools=[],
        )
        return agent
    
    @override
    async def achat(self, query_str: str) -> AgentResponse:
        logger.info(f"=> chitchat: {self.name}: {query_str}")

        response = await super().achat(query_str)
        response.route = AgentRouteType.CHITCHAT
        return response
    
#------------------------------------------------
# Example usage
#------------------------------------------------
if __name__ == "__main__":
    import asyncio
    
    agent = ChitChatAgent(name="Chit Chat Agent")
    response = asyncio.run(agent.achat("How are you?"))
    print(str(response))
    