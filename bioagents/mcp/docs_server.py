import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from mcp.server.fastmcp import FastMCP
from typing import List, Union, Literal

from bioagents.docs.querying import query_index
from bioagents.docs.processing import process_file
from bioagents.docs.mindmap import get_mind_map

mcp: FastMCP = FastMCP(name="MCP For NotebookLM")
mcp.settings.port = int(os.getenv("BIOMCP_PORT", "8131"))
mcp.settings.host = "localhost"

@mcp.tool(
    name="process_file_tool",
    description="This tool is useful to process files and produce summaries, question-answers and highlights.",
)
async def process_file_tool(
    filename: str,
) -> Union[str, Literal["Sorry, your file could not be processed."]]:
    try:
        notebook_model, text = await process_file(filename=filename)
        if notebook_model is None:
            return "Sorry, your file could not be processed."
        if text is None:
            text = ""
        return notebook_model + "\n%separator%\n" + text
    except Exception as e:
        return f"Sorry, your file could not be processed. Reason: {e}"


@mcp.tool(name="get_mind_map_tool", description="This tool is useful to get a mind ")
async def get_mind_map_tool(
    summary: str, highlights: List[str]
) -> Union[str, Literal["Sorry, mind map creation failed."]]:
    try:
        mind_map_fl = await get_mind_map(summary=summary, highlights=highlights)
        if mind_map_fl is None:
            return "Sorry, mind map creation failed."
        return mind_map_fl
    except Exception as e:
        return f"Sorry, mind map creation failed. Reason: {e}"


@mcp.tool(name="query_index_tool", description="Get knowledge from ingested documents in LlamaCloud index.")
async def query_index_tool(question: str) -> str:
    try:
        response = await query_index(question=question)
        if response is None:
            return "Sorry, I was unable to find an answer to your question."
        return response
    except Exception as e:
        return f"Sorry, I was unable to find an answer to your question. Reason: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
