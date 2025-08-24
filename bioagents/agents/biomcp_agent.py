# ------------------------------------------------------------------------------
# biomcp_agent.py
#
# A high-level interface for interacting with biomedical research tools via the MCP protocol.
# Provides a simple async chat interface while handling all the complexity of server management,
# connection handling, and resource cleanup.
#
# Features:
# - Simple async chat interface via achat() method
# - Automatic server lifecycle management with lazy initialization
# - Per-operation connections for proper async resource management
# - Robust error handling and diagnostics
# - Environment-based configuration
# - Context manager support for proper cleanup
# - Auto-start capability for interactive environments
#
# Example:
#     # Simple usage (auto-starts server on first call)
#     agent = BioMCPAgent()
#     result = await agent.achat("Find articles about Alzheimer's disease")
#     print(result)
#
#     # Explicit lifecycle management (recommended for production)
#     async with BioMCPAgent() as agent:
#         result = await agent.achat("Get details for variant rs113488022")
#         print(result)
#
# Author: Theodore Mui
# Date: 2025-08-16
# ------------------------------------------------------------------------------

from agents import Agent
import asyncio
import os
from typing import override, Optional
from loguru import logger

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from agents.tracing import set_tracing_disabled

set_tracing_disabled(disabled=True)

from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM
from bioagents.agents.base_agent import BaseAgent
from datetime import timedelta

BIOMCP_URL = os.getenv("BIOMCP_SERVER_URL", "http://localhost:8132/mcp/")

INSTRUCTIONS = f"""\
You are an expert that can answer biomedical, genetic, pubmed, and general medical questions
using BioMCP research tools via the MCP protocol.
You should always directly answer the user's question, without asking for permission, any preambles.
Your response should include relevant citation information from the source documents.\n

## Response Instructions:
- Prepend the response with '[BioMCP]'
"""


class BioMCPAgent(BaseAgent):
    """
    This is an expert agent that can answer biomedical, genetic, pubmed, and general medical questions
    using BioMCP research tools via the MCP protocol.
    """

    def __init__(
        self,
        name: str,
        model_name: str = LLM.GPT_4_1_NANO,
    ):
        self.instructions = INSTRUCTIONS
        self.handoff_description = (
            "You are an expert that can answer biomedical, genetic, pubmed, and general medical questions "
            "using BioMCP research tools via the MCP protocol."
        )

        super().__init__(name, model_name, self.instructions)
        self._mcp_server: Optional[MCPServerStreamableHttp] = None
        self._started: bool = False
        self._agent: Optional[Agent] = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create the core Agent; MCP transport will be created per operation."""
        return Agent(
            name=self.name,
            model=self.model_name,
            instructions=self.instructions,
            handoff_description=self.handoff_description,
            model_settings=ModelSettings(
                tool_choice="required",
                temperature=0.01,
                top_p=1.0,
            ),
            output_type=AgentResponse,
        )

    async def start(self) -> None:
        """No-op: server is created per operation in achat()."""
        if self._agent is None:
            self._agent = self._create_agent()
        self._started = True

    async def aclose(self) -> None:
        # Nothing persistent to close; ensure agent detaches MCP servers.
        if self._agent is not None:
            try:
                self._agent.mcp_servers = []  # type: ignore[attr-defined]
            except Exception:
                pass

    async def stop(self) -> None:
        await self.aclose()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    @override
    async def achat(self, query_str: str) -> AgentResponse:
        logger.info(f"=> biomcp: {self.name}: {query_str}")
        # Per-operation MCP server lifecycle to avoid stale event loop bindings
        try:
            if self._agent is None:
                self._agent = self._create_agent()

            server = MCPServerStreamableHttp(
                name="LlamaCloud MCP Server",
                params={
                    "url": BIOMCP_URL,
                    "timeout": timedelta(seconds=30),
                    "sse_read_timeout": timedelta(seconds=120),
                    "terminate_on_close": False,
                },
                client_session_timeout_seconds=120,
            )

            await server.__aenter__()
            try:
                try:
                    tool_list = await server.list_tools()
                    print(
                        f"\t# BioMCPAgent tools: {len(tool_list)}: {[tool.name for tool in tool_list]}"
                    )
                except Exception:
                    pass

                try:
                    self._agent.mcp_servers = [server]  # type: ignore[attr-defined]
                except Exception:
                    pass

                response = await super().achat(query_str)
                response.route = AgentRouteType.BIOMCP
                return response
            finally:
                try:
                    await server.__aexit__(None, None, None)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"BioMCP connection failed to {BIOMCP_URL}: {e}")
            return AgentResponse(
                response_str=f"[BioMCP] MCP server not reachable at {BIOMCP_URL}.",
                route=AgentRouteType.BIOMCP,
            )

        response = await super().achat(query_str)
        response.route = AgentRouteType.LLAMARAG
        return response


# ------------------------------------------------
# Example usage
# ------------------------------------------------
async def smoke_tests() -> None:
    try:
        print("==> 1")
        agent = BioMCPAgent(name="Bio MCP Agent")
        print("==> 2")
        await agent.start()
        print("==> 3")
        print(str(await agent.achat("What is ICD-10?")))
        print("==> 4")
        print(
            str(
                await agent.achat(
                    "Give us a few pubmed articles about the spread of COVID in 2025."
                )
            )
        )
        print("==> 5")
    finally:
        print("==> 6")
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(smoke_tests())
