import json
import os

from workflows import Workflow, step, Context
from workflows.events import StartEvent, StopEvent, Event
from workflows.resource import Resource
from llama_index.tools.mcp import BasicMCPClient
from typing import Annotated, List, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Allow long-running tools (e.g., LlamaCloud extraction) to complete without client timeout.
# Configurable via MCP_CLIENT_TIMEOUT env var; default to 900s (15 minutes).
MCP_CLIENT_TIMEOUT = int(os.getenv("MCP_CLIENT_TIMEOUT", "900"))
DOCMCP_SERVER_URL = os.getenv("DOCMCP_SERVER_URL", "http://localhost:8130/mcp")
DOCMCP_CLIENT = BasicMCPClient(command_or_url=DOCMCP_SERVER_URL, timeout=MCP_CLIENT_TIMEOUT)

# Retry configuration for MCP tool calls
MCP_CALL_MAX_ATTEMPTS = int(os.getenv("MCP_CALL_MAX_ATTEMPTS", "3"))
MCP_CALL_BACKOFF_MULTIPLIER = float(os.getenv("MCP_CALL_BACKOFF_MULTIPLIER", "2.0"))


@retry(
    reraise=True,
    stop=stop_after_attempt(MCP_CALL_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=MCP_CALL_BACKOFF_MULTIPLIER),
    retry=retry_if_exception_type(Exception),
)
async def call_mcp_tool_with_retry(
    mcp_client: BasicMCPClient, *, tool_name: str, arguments: dict
):
    return await mcp_client.call_tool(tool_name=tool_name, arguments=arguments)


class FileInputEvent(StartEvent):
    file: str


class NotebookOutputEvent(StopEvent):
    mind_map: str
    md_content: str
    summary: str
    highlights: List[str]
    questions: List[str]
    answers: List[str]


class MindMapCreationEvent(Event):
    summary: str
    highlights: List[str]
    questions: List[str]
    answers: List[str]
    md_content: str


def get_mcp_client(*args, **kwargs) -> BasicMCPClient:
    return DOCMCP_CLIENT


class NotebookLMWorkflow(Workflow):
    @step
    async def extract_file_data(
        self,
        ev: FileInputEvent,
        mcp_client: Annotated[BasicMCPClient, Resource(get_mcp_client)],
        ctx: Context,
    ) -> Union[MindMapCreationEvent, NotebookOutputEvent]:
        ctx.write_event_to_stream(
            ev=ev,
        )
        try:
            result = await call_mcp_tool_with_retry(
                mcp_client, tool_name="process_file_tool", arguments={"filename": ev.file}
            )
            split_result = result.content[0].text.split("\n%separator%\n")
            if len(split_result) > 1:
                json_data = split_result[0]
                if json_data.startswith("Sorry, your file"):
                    return NotebookOutputEvent(
                        mind_map="Unprocessable file, sorry😭",
                        md_content="",
                        summary=f"{json_data}",
                        highlights=[],
                        questions=[],
                        answers=[],
                    )
                md_text = split_result[1]
                json_rep = json.loads(json_data)
                return MindMapCreationEvent(
                    md_content=md_text,
                    **json_rep,
                )
        except Exception as e:
            print(f"⚠️  MCP client error: {e}")
            
        return NotebookOutputEvent(
            mind_map="Ensure MCP server - process_file_tool - is running on port 8131.😭",
            md_content="",
            summary="",
            highlights=[],
            questions=[],
            answers=[],
        )

    @step
    async def generate_mind_map(
        self,
        ev: MindMapCreationEvent,
        mcp_client: Annotated[BasicMCPClient, Resource(get_mcp_client)],
        ctx: Context,
    ) -> NotebookOutputEvent:
        ctx.write_event_to_stream(
            ev=ev,
        )
        try:
            result = await call_mcp_tool_with_retry(
                mcp_client,
                tool_name="get_mind_map_tool",
                arguments={"summary": ev.summary, "highlights": ev.highlights},
            )
            if result is not None:
                return NotebookOutputEvent(
                    mind_map=result.content[0].text,
                    **ev.model_dump(
                        include={
                            "summary",
                            "highlights",
                            "questions",
                            "answers",
                            "md_content",
                        }
                    ),
                )
            return NotebookOutputEvent(
                mind_map="Sorry, mind map creation failed😭",
                **ev.model_dump(
                    include={
                        "summary",
                        "highlights",
                        "questions",
                        "answers",
                        "md_content",
                    }
                ),
            )
        except Exception as e:
            print(f"⚠️  MCP client error in mind map generation: {e}")
            return NotebookOutputEvent(
                mind_map="MCP server connection failed during mind map generation. Please ensure the server is running on port 8131.😭",
                **ev.model_dump(
                    include={
                        "summary",
                        "highlights",
                        "questions",
                        "answers",
                        "md_content",
                    }
                ),
            )
