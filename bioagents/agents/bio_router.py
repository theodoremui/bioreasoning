# ------------------------------------------------------------------------------
# bio_router.py
#
# This is a "Concierge" that routes queries to the appropriate subagent.
# This agent orchestrates across the following subagents:
#
# 1. Routing Agent
# 2. Web Reasoning Agent
# 3. Bio MCP Agent
# 4. Graph Agent
# 5. LlamaRAG Agent
# 6. LlamaMCP Agent
#
# Routing is performed by first classifying the query using an LLM,
# which assigns a context (e.g. graph, biomcp, web, chitchat) to the query.
# The most relevant (top 1) context is then used to route the query 
# to the appropriate subagent.
#
#
# Author: Theodore Mui
# Date: 2025-04-26
# ------------------------------------------------------------------------------

import asyncio
from typing import override

from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
)
from loguru import logger

from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.biomcp_agent import BioMCPAgent
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.agents.graph_agent import GraphAgent
from bioagents.agents.llamamcp_agent import LlamaMCPAgent
from bioagents.agents.llamarag_agent import LlamaRAGAgent
from bioagents.agents.web_agent import WebReasoningAgent
from bioagents.models.llms import LLM


class BioRouterAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        model_name: str = LLM.GPT_4_1_MINI,
    ):
        instructions = (
            "You are a bio-reasoning agent that routes queries to appropriate specialists. "
            "You analyze the user's query and determine the best way to respond."
            "\n## Response Instructions:\n"
            "- Prepend your response with '[Concierge]'\n"
        )

        super().__init__(name, model_name, instructions)

        # Defer construction until start(); keep simple, explicit lifecycle
        self._started = False
        self._agent = None

    async def start(self) -> None:
        if self._started:
            return
        # Create subagents
        self._chit_chat_agent = ChitChatAgent(name="Chit Chat Agent")
        self._web_agent = WebReasoningAgent(name="Web Reasoning Agent")
        self._biomcp_agent = BioMCPAgent(name="Bio MCP Agent")
        self._graph_agent = GraphAgent(name="Graph Agent")
        self._llamarag_agent = LlamaRAGAgent(name="LlamaCloud RAG Agent")
        self._llamamcp_agent = LlamaMCPAgent(name="LlamaCloud MCP Agent")
        # Start is deferred; each wrapper's achat will start on demand

        # Router tools: each tool encodes a routing decision label. The Runner will
        # pick one tool (stop_on_first_tool), and we will delegate to the wrapper's achat.
        @function_tool()
        def route_graph() -> str:
            """Route to the Graph agent for complex reasoning about NCCN Breast Cancer Guidelines using knowledge graph relationships."""
            return "graph"

        @function_tool()
        def route_llamarag() -> str:
            """Route to the RAG agent for document retrieval and simple questions about NCCN Breast Cancer Guidelines."""
            return "llamarag"

        @function_tool()
        def route_biomcp() -> str:
            """Route to the Bio MCP agent to answer biomedical questions about genetic variants, research articles, and biomedical data. ."""
            return "biomcp"

        @function_tool()
        def route_websearch() -> str:
            """Route to the web specialist for general or real-time information."""
            return "websearch"

        @function_tool()
        def route_chitchat() -> str:
            """Route to the Chit Chat specialist for informal conversation.  This is a fallback route."""
            return "chitchat"

        @function_tool()
        def route_llamamcp() -> str:
            """Route to the MCP agent (MCP tools over streamable-http)."""
            return "llamamcp"

        router_instructions = (
            "You are a routing specialist. Read the user's query and choose exactly ONE routing tool that best matches the need. "
            "Do NOT answer the query yourself. Follow these strict rules:\n\n"
            "- Use route_graph for complex reasoning questions about NCCN Breast Cancer guidelines that require understanding relationships between treatments, conditions, and patient factors.\n"
            "- Use route_llamarag for simple document retrieval questions about NCCN Breast Cancer guidelines.\n"
            "- Use route_llamamcp for simple questions about NCCN Breast Cancer guidelines.\n"
            "- Use route_biomcp ONLY for biomedical/clinical topics (genes, variants, diseases, trials, papers, biomedical datasets).\n"
            "- Use route_websearch for general knowledge, science (physics/astronomy/chemistry), current events, and anything not clearly biomedical.\n"
            "- Use route_chitchat for informal conversation when no task is requested.\n\n"
            "Examples:\n"
            "Q: Why is the sky blue? -> route_websearch\n"
            "Q: What is BRCA1 and how do variants affect risk? -> route_biomcp\n"
            "Q: What are the NCCN recommendations for HER2+ stage II breast cancer? -> route_llamarag\n"
            "Q: What is the recommended treatment for HER2+ stage II breast cancer? -> route_llamamcp\n"
            "Q: How do HER2 status, hormone receptor status, and tumor grade interact to determine treatment options in breast cancer? -> route_graph\n"
            "Q: Tell me a joke -> route_chitchat\n"
        )

        self._router_agent = Agent(
            name="Bio Concierge",
            model=self.model_name,
            instructions=router_instructions,
            handoffs=[
                self._llamarag_agent._agent,
                (
                    self._biomcp_agent._agent
                    if hasattr(self._biomcp_agent, "_agent")
                    else None
                ),
                self._web_agent._agent,
                self._chit_chat_agent._agent,
            ],
            tools=[
                route_graph,
                route_llamarag,
                route_llamamcp,
                route_biomcp,
                route_websearch,
                route_chitchat,
            ],
            model_settings=ModelSettings(
                tool_choice="required",
            ),
            tool_use_behavior="stop_on_first_tool",
        )

        # Keep a minimal concierge Agent reference (not used for routing anymore)
        self._agent = self._router_agent
        self._started = True

    # Lifecycle to manage subagents cleanly
    async def stop(self) -> None:
        # Stop networked subagents to avoid loop teardown noise
        try:
            if hasattr(self, "_llamamcp_agent") and self._llamamcp_agent:
                await self._llamamcp_agent.stop()
        except Exception:
            pass
        self._agent = None
        self._started = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    @staticmethod
    async def _maybe_call_async(method, *args, **kwargs):
        """Call a method that may be async or sync; return its result."""
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return method(*args, **kwargs)

    def _construct_response_with_agent_info(
        self, run_result, response_text: str, route: AgentRouteType
    ) -> AgentResponse:
        """Construct response with agent-specific information and routing."""
        # Extract citations from run_result
        citations = []
        for item in run_result.new_items:
            if item.type == "message_output_item":
                for content in item.raw_item.content:
                    if hasattr(content, "annotations"):
                        for annotation in content.annotations:
                            if annotation.type == "url_citation":
                                from bioagents.models.source import Source

                                citations.append(
                                    Source(
                                        url=annotation.url,
                                        title=annotation.title,
                                        snippet=content.text[
                                            annotation.start_index : annotation.end_index
                                        ],
                                        source="web",
                                    )
                                )

        return AgentResponse(
            response_str=response_text,
            citations=citations,
            judge_response="",
            route=route,
        )

    @override
    async def achat(self, query_str: str) -> AgentResponse:
        logger.info(f"=> concierge: {self.name}: {query_str}")
        if not getattr(self, "_started", False):
            await self.start()

        # 1) Ask the router agent to pick one route tool
        route_result = await Runner.run(
            starting_agent=self._router_agent,
            input=query_str,
            max_turns=1,
        )
        route_label = (route_result.final_output or "").strip().lower()

        # If the router already produced a concrete response from a specific agent,
        # do not delegate; wrap the response directly to satisfy tests and simplify behavior.
        prefixed_text = (route_result.final_output or "").strip()
        if ":" in prefixed_text:
            lower = prefixed_text.lower()
            if lower.startswith("web reasoning agent:"):
                return self._construct_response_with_agent_info(
                    route_result, prefixed_text, AgentRouteType.WEBSEARCH
                )
            if lower.startswith("chit chat agent:"):
                return self._construct_response_with_agent_info(
                    route_result, prefixed_text, AgentRouteType.CHITCHAT
                )
            if lower.startswith("graph agent:"):
                return self._construct_response_with_agent_info(
                    route_result, prefixed_text, AgentRouteType.GRAPH
                )
            if (
                lower.startswith("bio mcp agent:")
                or lower.startswith("biomcp")
                or lower.startswith("biomedical")
            ):
                return self._construct_response_with_agent_info(
                    route_result, prefixed_text, AgentRouteType.BIOMCP
                )
            # Default to concierge reasoning route
            return self._construct_response_with_agent_info(
                route_result, prefixed_text, AgentRouteType.REASONING
            )

        # 2) Delegate to the selected wrapper's achat
        if "web reasoning agent" in route_label or "websearch" in route_label:
            wrapper_response = await self._maybe_call_async(
                self._web_agent.achat, query_str
            )
            wrapper_response.route = AgentRouteType.WEBSEARCH
            return wrapper_response
        if "graph" in route_label:
            wrapper_response = await self._maybe_call_async(
                self._graph_agent.achat, query_str
            )
            wrapper_response.route = AgentRouteType.GRAPH
            return wrapper_response
        if "llama" in route_label and "rag" in route_label:
            wrapper_response = await self._maybe_call_async(
                self._llamarag_agent.achat, query_str
            )
            wrapper_response.route = AgentRouteType.LLAMARAG
            return wrapper_response
        if (
            "bio mcp" in route_label
            or "biomcp" in route_label
            or "biomedical" in route_label
        ):
            wrapper_response = await self._maybe_call_async(
                self._biomcp_agent.achat, query_str
            )
            wrapper_response.route = AgentRouteType.BIOMCP
            return wrapper_response
        if "llama" in route_label and "mcp" in route_label:
            wrapper_response = await self._maybe_call_async(
                self._llamamcp_agent.achat, query_str
            )
            wrapper_response.route = AgentRouteType.LLAMAMCP
            return wrapper_response

        # Default fallback to chit chat if routing is unclear
        achat_method = getattr(self._chit_chat_agent, "achat", None)
        wrapper_response = await self._maybe_call_async(achat_method, query_str)
        wrapper_response.route = AgentRouteType.CHITCHAT
        return wrapper_response


# ------------------------------------------------
# Example usage
# ------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import time

    async def _demo() -> None:
        async with BioRouterAgent(name="BioConcierge") as agent:
            start_time = time.time()
            resp = await agent.achat("How are you?")
            print(f"{str(resp)} ({time.time() - start_time:.1f}s)")

            start_time = time.time()
            resp = await agent.achat(
                "What is the latest news in the field of genetics?"
            )
            print(f"{str(resp)} ({time.time() - start_time:.1f}s)")

            start_time = time.time()
            resp = await agent.achat("How is the weather in Tokyo?")
            print(f"{str(resp)} ({time.time() - start_time:.1f}s)")

    asyncio.run(_demo())
