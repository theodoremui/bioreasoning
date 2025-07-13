#------------------------------------------------------------------------------
# common.py
# 
# This file provides base classes for agent interactions. It provides a common
# interface for all agent responses.
# 
# Author: Theodore Mui
# Date: 2025-04-26
#------------------------------------------------------------------------------


from dataclasses import dataclass, field
from typing import List
from bioagents.models.citation import Citation
from enum import Enum

class AgentRoute(Enum):
    BIO_MCP = "biomcp"
    WEB = "web"
    CHIT_CHAT = "chit_chat"
    UNKNOWN = "unknown"

@dataclass
class AgentResponse:
    response_str: str
    citations: List[Citation] = field(default_factory=list)
    judge_response: str = ""
    route: AgentRoute = AgentRoute.UNKNOWN
