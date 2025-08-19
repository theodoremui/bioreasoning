#------------------------------------------------------------------------------
# source.py
# 
# This is a data class for sources. It provides a common interface for all sources.
# 
# Author: Theodore Mui
# Date: 2025-04-26
#------------------------------------------------------------------------------

from typing import List, Optional
from pydantic import BaseModel, Field

class Source(BaseModel):
    url: Optional[str] = Field(description="The URL of the source text if any", default="")
    title: Optional[str] = Field(description="Succinct title representing the full text", default="")
    snippet: Optional[str] = Field(description="A snippet of the text", default="")
    source: Optional[str] = Field(description="The source of the citation", default="")
    file_name: Optional[str] = Field(description="The file name of the source", default="")
    start_page_label: Optional[str] = Field(description="The start page label of the source", default="")
    end_page_label: Optional[str] = Field(description="The end page label of the source", default="")
    score: Optional[float] = Field(description="The score of the source", default=0.0)
    text: str = Field(description="Original full text -- without edits or any change")

class SourceList(BaseModel):
    sources: List[Source] = Field(description="A list of sources")
    
