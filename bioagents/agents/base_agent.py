# ------------------------------------------------------------------------------
# base_agent.py
#
# This is the base class for all reasoning agents. It provides a common interface
# for all reasoning agents.
#
# Author: Theodore Mui
# Date: 2025-04-26
# ------------------------------------------------------------------------------

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import asyncio
from abc import ABC, abstractmethod

from agents import Runner, RunResult, gen_trace_id, trace

try:
    from agents.tracing import set_tracing_disabled

    set_tracing_disabled(
        True
    )  # Disable external tracing by default to avoid runtime teardown issues
except Exception:
    pass
from loguru import logger

from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM
from bioagents.models.source import Source
from bioagents.utils.text_utils import make_contextual_snippet


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        model_name: str = LLM.GPT_4_1_MINI,
        instructions: str = "You are a reasoning conversational agent.",
        timeout: int = 60,
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
        route: AgentRouteType = AgentRouteType.REASONING,
    ) -> AgentResponse:
        sources = []
        seen_urls = set()
        for item in run_result.new_items:
            if item.type == "message_output_item":
                for content in item.raw_item.content:
                    if hasattr(content, "annotations"):
                        for annotation in content.annotations:
                            if annotation.type == "url_citation":
                                url = getattr(annotation, "url", None)
                                if url and url not in seen_urls:
                                    sources.append(
                                        Source(
                                            url=url,
                                            title=annotation.title,
                                            snippet=content.text[
                                                annotation.start_index : annotation.end_index
                                            ],
                                            source="web",
                                        )
                                    )
                                    seen_urls.add(url)

        return AgentResponse(
            response_str=run_result.final_output,
            citations=sources,
            judge_response=judge_response,
            route=route,
        )

    async def achat(self, query_str: str) -> AgentResponse:
        if self._agent is None:
            raise ValueError("Agent not initialized")

        try:
            trace_id = gen_trace_id()
            # Avoid context managers that may hook task groups on close; log trace id explicitly
            result = await Runner.run(
                starting_agent=self._agent,
                input=query_str,
                max_turns=3,
            )
            logger.info(f"\t{self.name}: {query_str} -> {trace_id}")
            # If a tool returned an AgentResponse directly, surface it as-is
            if isinstance(result.final_output, AgentResponse):
                return result.final_output
            return self._construct_response(result, "", AgentRouteType.REASONING)
        except Exception as e:
            logger.error(f"achat: {str(e)}")
            raise e


# ------------------------------------------------
# Example usage
# ------------------------------------------------
if __name__ == "__main__":
    import asyncio

    agent = BaseAgent(name="ReasoningAgent")
    response = asyncio.run(agent.achat("What is the capital of the moon?"))
    print(str(response))
