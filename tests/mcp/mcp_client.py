from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import asyncio
import os
import shutil
import subprocess
import time
from typing import Any

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from agents.tracing import set_tracing_disabled

set_tracing_disabled(disabled=True)

async def run(mcp_server: MCPServer):
    agent = Agent(
        name="mcp-agent",
        instructions=(
            "You are a helpful assistant that only use the tools provided to you."
            "You should NOT use your own knowledge to answer the question."
        ),
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(
            tool_choice="required",
        ),
    )
    user_input = input("Enter a question: ")
    while user_input.strip() != "":
        result = await Runner.run(starting_agent=agent, input=user_input)
        print(result.final_output)
        user_input = input("Enter a question: ")

async def main():
    print("starting MCP server... ", flush=True)
    async with MCPServerStreamableHttp(
        name="local-basic",
        params={
            "url": "http://localhost:8131/mcp"
        }
    ) as server:
        print("\ttools: ", await server.list_tools())
        await run(server)

if __name__ == "__main__":
    # Let's make sure the user has uv installed
    if not shutil.which("uv"):
        raise RuntimeError(
            "uv is not installed. Please install it: https://docs.astral.sh/uv/getting-started/installation/"
        )

    # We'll run the Streamable HTTP server in a subprocess. Usually this would be a remote server, but for this
    # demo, we'll run it locally at http://localhost:8131/mcp
    process: subprocess.Popen[Any] | None = None
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        server_file = os.path.join(this_dir, "mcp_server.py")

        print("Starting Streamable HTTP server at http://localhost:8131/mcp ...")

        # Run `uv run server.py` to start the Streamable HTTP server
        process = subprocess.Popen(["uv", "run", server_file])
        # Give it 3 seconds to start
        time.sleep(3)

        print("Streamable HTTP server started. Running example...\n\n")
    except Exception as e:
        print(f"Error starting Streamable HTTP server: {e}")
        exit(1)

    try:
        asyncio.run(main())
    finally:
        if process:
            process.terminate()