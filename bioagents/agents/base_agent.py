#------------------------------------------------------------------------------
# base_agent.py
# 
# This is the base class for all reasoning agents. It provides a common interface
# for all reasoning agents.
# 
# Author: Theodore Mui
# Date: 2025-04-26
#------------------------------------------------------------------------------

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from abc import ABC, abstractmethod
import asyncio
from agents import Runner, RunResult, gen_trace_id, trace
from loguru import logger

from bioagents.models.citation import Citation
from bioagents.models.llms import LLM
from bioagents.agents.common import AgentResponse, AgentRouteType

class BaseAgent(ABC):
    def __init__(
        self, name: str, 
        model_name: str=LLM.GPT_4_1_MINI, 
        instructions: str="You are a reasoning conversational agent.",
        timeout: int=60
    ):
        self.name = name
        self.model_name = model_name
        self.instructions = instructions
        self.timeout = timeout
        self._agent = None

    def _construct_response(
        self, 
        run_result: RunResult,
        judge_response: str = "",
        route: AgentRouteType = AgentRouteType.REASONING
    ) -> AgentResponse:
        citations = []
        for item in run_result.new_items:
            if item.type == 'message_output_item':
                for content in item.raw_item.content:
                    if hasattr(content, 'annotations'):
                        for annotation in content.annotations:
                            if annotation.type == 'url_citation':
                                citations.append(Citation(
                                    url=annotation.url,
                                    title=annotation.title,
                                    snippet=content.text[annotation.start_index:annotation.end_index],
                                    source="web"
                                ))
        
        return AgentResponse(
            response_str=run_result.final_output,
            citations=citations,
            judge_response=judge_response,
            route=route
        )
        
    async def achat(self, query_str: str) -> AgentResponse:
        if self._agent is None:
            raise ValueError("Agent not initialized")
        
        try:
            trace_id = gen_trace_id()
            with trace(workflow_name="BaseAgent", trace_id=trace_id):
                result = await asyncio.wait_for(
                    Runner.run(
                        starting_agent=self._agent,
                        input=query_str,
                        max_turns=3,
                    ),
                    timeout=self.timeout
                )
                    
                logger.info(f"{self.name}: {query_str} -> {trace_id}")
                return self._construct_response(result, "", AgentRouteType.REASONING)
                            
        except Exception as e:
            logger.error(f"achat: {str(e)}")
            raise e
#------------------------------------------------
# Example usage
#------------------------------------------------
if __name__ == "__main__":
    import asyncio
    
    agent = BaseAgent(name="ReasoningAgent")
    response = asyncio.run(agent.achat("What is the capital of the moon?"))
    print(str(response))
    