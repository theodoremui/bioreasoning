import random
import requests
from mcp.server.fastmcp import FastMCP

# Create the MCP server instance
mcp = FastMCP("local-basic")


# A simple tool the agent can call
@mcp.tool()
def greet(name: str) -> str:
    """Return a friendly greeting."""
    try:
        return f"Hello, {name}! 👋"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b
