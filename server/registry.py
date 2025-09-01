from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol

from loguru import logger

from bioagents.agents.base_agent import BaseAgent


class AsyncAgentFactory(Protocol):
    def __call__(self) -> BaseAgent: ...


@dataclass
class AgentDescriptor:
    name: str
    factory: AsyncAgentFactory


DEFAULT_AGENTS: Dict[str, AgentDescriptor] = {
    "halo": AgentDescriptor("halo", lambda: _resolve_dotted("bioagents.agents.bio_halo:BioHALOAgent")),
    "router": AgentDescriptor("router", lambda: _resolve_dotted("bioagents.agents.bio_router:BioRouterAgent")),
    "graph": AgentDescriptor("graph", lambda: _resolve_dotted("bioagents.agents.graph_agent:GraphAgent")),
    "llamamcp": AgentDescriptor("llamamcp", lambda: _resolve_dotted("bioagents.agents.llamamcp_agent:LlamaMCPAgent")),
    "llamarag": AgentDescriptor("llamarag", lambda: _resolve_dotted("bioagents.agents.llamarag_agent:LlamaRAGAgent")),
    "biomcp": AgentDescriptor("biomcp", lambda: _resolve_dotted("bioagents.agents.biomcp_agent:BioMCPAgent")),
    "web": AgentDescriptor("web", lambda: _resolve_dotted("bioagents.agents.web_agent:WebReasoningAgent")),
    "chitchat": AgentDescriptor("chitchat", lambda: _resolve_dotted("bioagents.agents.chitchat_agent:ChitChatAgent")),
}


def _resolve_dotted(path: str) -> BaseAgent:
    module_name, _, class_name = path.replace("::", ":").partition(":")
    if not module_name or not class_name:
        raise ValueError(f"Invalid agent dotted path '{path}'. Use 'module:Class'")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    obj = cls(name=getattr(cls, "__name__", class_name))
    if not isinstance(obj, BaseAgent):
        raise TypeError(f"Resolved '{path}' is not a BaseAgent subclass")
    return obj


class AgentRegistry:
    """Lazy singleton-like registry for agents.

    - Allows YAML overrides via dotted paths
    - Maintains a single instance per agent key
    - Ensures async start/stop as needed by callers
    """

    def __init__(self, overrides: Optional[Dict[str, str]] = None) -> None:
        self._descriptors: Dict[str, AgentDescriptor] = dict(DEFAULT_AGENTS)
        if overrides:
            for key, dotted in overrides.items():
                try:
                    # Build a factory that resolves at first use
                    self._descriptors[key] = AgentDescriptor(
                        key,
                        lambda dotted_path=dotted: _resolve_dotted(dotted_path),
                    )
                except Exception as e:
                    logger.warning(f"Failed to register override for '{key}': {e}")
        self._instances: Dict[str, BaseAgent] = {}

    def available(self) -> Dict[str, str]:
        return {k: d.factory.__name__ if hasattr(d.factory, "__name__") else "factory" for k, d in self._descriptors.items()}

    def get(self, key: str) -> BaseAgent:
        if key not in self._descriptors:
            raise KeyError(f"Unknown agent '{key}'")
        if key in self._instances:
            return self._instances[key]
        desc = self._descriptors[key]
        agent = desc.factory()
        self._instances[key] = agent
        return agent


