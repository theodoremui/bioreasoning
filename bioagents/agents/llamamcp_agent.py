# ------------------------------------------------------------------------------
# llamamcp_agent.py
#
# This agent is an LlamaCloud MCP agent that can query the LlamaCloud index.
#
# Author: Theodore Mui
# Date: 2025-08-16
# ------------------------------------------------------------------------------

import asyncio
import os
from typing import Optional, override

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from agents.tracing import set_tracing_disabled
from loguru import logger

set_tracing_disabled(disabled=True)

from datetime import timedelta

from bioagents.agents.base_agent import BaseAgent
from bioagents.agents.common import AgentResponse, AgentRouteType
from bioagents.models.llms import LLM

DOCMCP_URL = os.getenv("DOCMCP_SERVER_URL", "http://localhost:8130/mcp/")

INSTRUCTIONS = f"""
You are an LlamaCloud MCP agent that can query documents and knowledge about ICD-10 and medical codes.
You should always directly answer the user's question, without asking for permission, any preambles.
Your response should include relevant citation information from the source documents.\n

## Response Instructions:
- Prepend the response with '[MCP]'
"""


class LlamaMCPAgent(BaseAgent):
    """
    This agent is an LlamaCloud MCP agent that can query the LlamaCloud index.
    """

    def __init__(
        self,
        name: str,
        model_name: str = LLM.GPT_4_1_NANO,
    ):
        self.instructions = INSTRUCTIONS
        self.handoff_description = "You are an LlamaCloud MCP agent that can query documents and knowledge about ICD-10 and medical codes."

        super().__init__(name, model_name, self.instructions)
        self._mcp_server: Optional[MCPServerStreamableHttp] = None
        self._started: bool = False
        self._agent: Optional[Agent] = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create the core Agent and instantiate the MCP transport (no connect)."""
        # Instantiate transport with robust timeouts; connect later in start()
        self._mcp_server = MCPServerStreamableHttp(
            name="LlamaCloud MCP Server",
            params={
                "url": DOCMCP_URL,
                "timeout": timedelta(seconds=20),
                "sse_read_timeout": timedelta(seconds=120),
                "terminate_on_close": False,
            },
            client_session_timeout_seconds=120,
        )

        return Agent(
            name=self.name,
            model=self.model_name,
            instructions=self.instructions,
            handoff_description=self.handoff_description,
            model_settings=ModelSettings(tool_choice="required"),
            tool_use_behavior="stop_on_first_tool",
        )

    async def start(self) -> None:
        if self._started:
            return
        try:
            if self._mcp_server is None:
                self._agent = self._create_agent()
            await self._mcp_server.__aenter__()  # type: ignore[arg-type]
            try:
                tool_list = await self._mcp_server.list_tools()  # type: ignore[union-attr]
                print(
                    f"\t# LlamaMCPAgent tools: {len(tool_list)}: {[tool.name for tool in tool_list]}"
                )
            except Exception:
                pass
            if self._agent is not None:
                try:
                    self._agent.mcp_servers = [self._mcp_server]  # type: ignore[attr-defined]
                except Exception:
                    pass
            self._started = True
        except Exception:
            if self._mcp_server is not None:
                try:
                    await self._mcp_server.__aexit__(None, None, None)
                except Exception:
                    pass
            self._mcp_server = None
            self._started = False
            raise

    async def aclose(self) -> None:
        if self._mcp_server is not None:
            try:
                await self._mcp_server.__aexit__(None, None, None)
            except Exception:
                pass
            finally:
                self._mcp_server = None
        # Keep the Agent instance available; just detach MCP servers
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
        logger.info(f"=> llamamcp: {self.name}: {query_str}")

        try:
            if not self._started:
                await self.start()
            result = await Runner.run(starting_agent=self._agent, input=query_str)
            return self._construct_response(result, "", AgentRouteType.LLAMAMCP)
        except Exception as e:
            logger.error(f"MCP connection failed to {DOCMCP_URL}: {e}")
            return AgentResponse(
                response_str=f"[LlamaCloud MCP] MCP server not reachable at {DOCMCP_URL}. Start it first, then retry.",
                route=AgentRouteType.LLAMAMCP,
            )


# ------------------------------------------------
# Example usage
# ------------------------------------------------
async def smoke_tests() -> None:
    try:
        print("==> 1")
        agent = LlamaMCPAgent(name="LlamaCloud MCP Agent")
        print("==> 2")
        await agent.start()
        print("==> 3")
        print(str(await agent.achat("What is ICD-10?")))
        print("==> 4")
        print(
            str(
                await agent.achat(
                    "What are the top 10 United Nations climate mandates?"
                )
            )
        )
        print("==> 5")
    finally:
        print("==> 6")
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(smoke_tests())
