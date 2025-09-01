from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query text")


class Citation(BaseModel):
    url: Optional[str] = ""
    title: Optional[str] = ""
    snippet: Optional[str] = ""
    source: Optional[str] = ""
    file_name: Optional[str] = ""
    start_page_label: Optional[str] = ""
    end_page_label: Optional[str] = ""
    score: Optional[float] = 0.0
    text: Optional[str] = ""


class ChatResponse(BaseModel):
    response: str
    citations: List[Citation] = Field(default_factory=list)
    judgement: str = ""
    route: str = "reasoning"


