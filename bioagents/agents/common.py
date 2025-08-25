# ------------------------------------------------------------------------------
# common.py
#
# This file provides base classes for agent interactions. It provides a common
# interface for all agent responses.
#
# Author: Theodore Mui
# Date: 2025-04-26
# ------------------------------------------------------------------------------


from dataclasses import dataclass, field
from enum import Enum
from typing import List

from bioagents.models.source import Source


class AgentRouteType(Enum):
    """
    The type of agent response.
    """

    BIOMCP = "biomcp"
    CHITCHAT = "chitchat"
    GRAPH = "graph"
    LLAMAMCP = "llamamcp"
    LLAMARAG = "llamarag"
    WEBSEARCH = "websearch"

    CONCIERGE = "concierge"
    REASONING = "reasoning"


@dataclass
class AgentResponse:
    response_str: str
    citations: List[Source] = field(default_factory=list)
    judge_response: str = ""
    route: AgentRouteType = AgentRouteType.REASONING
