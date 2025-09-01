from __future__ import annotations

from typing import Dict

from loguru import logger

from bioagents.agents.common import AgentResponse
from server.registry import AgentRegistry
from server.retry import default_retry


class ChatService:
    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    @default_retry
    async def achat(self, agent_name: str, query: str) -> AgentResponse:
        agent = self._registry.get(agent_name)
        if hasattr(agent, "start"):
            try:
                await agent.start()  # type: ignore[attr-defined]
            except Exception:
                pass
        logger.info(f"ChatService achat to agent '{agent_name}'")
        return await agent.achat(query)

    async def simple_achat(self, agent_name: str, query: str) -> str:
        agent = self._registry.get(agent_name)
        if hasattr(agent, "start"):
            try:
                await agent.start()  # type: ignore[attr-defined]
            except Exception:
                pass
        logger.info(f"ChatService simplechat to agent '{agent_name}'")
        response_str = await agent.simple_achat(query)
        return response_str

