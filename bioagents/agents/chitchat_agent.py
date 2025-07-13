#------------------------------------------------------------------------------
# chitchat_agent.py
# 
# This agent is a friendly conversational assistant that informally chit chats with the user.
# 
# Author: Theodore Mui
# Date: 2025-04-26
#------------------------------------------------------------------------------

from agents import Agent, Runner
from bioagents.models.llms import LLM
from bioagents.agents.base_agent import ReasoningAgent
from bioagents.agents.common import AgentResponse, AgentRoute
from loguru import logger

class ChitChatAgent(ReasoningAgent):
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
                "about informal topics and your responses should be very brief and to the point."
            ),
            handoffs=[],
            tools=[],
        )
        return agent
    
    async def achat(self, query_str: str) -> AgentResponse:
        logger.info(f"-> {self.name}: {query_str}")

        run_result = await Runner.run(
            starting_agent=self._agent,
            input=query_str,
            max_turns=1,
        )
        
        return self._construct_response(
            run_result=run_result, 
            judge_response="", 
            route=AgentRoute.CHIT_CHAT
        )
#------------------------------------------------
# Example usage
#------------------------------------------------
if __name__ == "__main__":
    import asyncio
    
    agent = ChitChatAgent(name="Chit Chat Agent")
    response = asyncio.run(agent.achat("How are you?"))
    print(str(response))
    