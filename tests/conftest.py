"""
Pytest configuration and fixtures for bioagents tests.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator

# Ensure project root is on sys.path for imports like `bioagents.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env for testing
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
except ImportError:
    pass

# Set up environment for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")
os.environ.setdefault("BIOMCP_PORT", "8132")
os.environ.setdefault("DOCMCP_PORT", "8130")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = AsyncMock()

    # Mock a typical chat completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"

    client.chat.completions.create = AsyncMock(return_value=mock_response)

    return client


@pytest.fixture
async def mock_runner():
    """Mock the OpenAI agents Runner for testing."""
    from agents import RunResult
    from unittest.mock import MagicMock

    # Create a mock RunResult
    mock_result = MagicMock(spec=RunResult)
    mock_result.final_output = "Test response"
    mock_result.new_items = []

    # Mock the Runner.run method
    original_run = None
    try:
        from agents import Runner

        original_run = Runner.run
        Runner.run = AsyncMock(return_value=mock_result)
        yield mock_result
    finally:
        if original_run:
            Runner.run = original_run


@pytest.fixture
def sample_citations():
    """Sample citations for testing."""
    from bioagents.models.source import Source

    return [
        Source(
            url="https://example.com/article1",
            title="Sample Article 1",
            snippet="This is a sample snippet",
            source="web",
        ),
        Source(
            url="https://example.com/article2",
            title="Sample Article 2",
            snippet="Another sample snippet",
            source="pubmed",
        ),
    ]


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing BioMCP functionality."""
    server = MagicMock()
    server.__aenter__ = AsyncMock(return_value=server)
    server.__aexit__ = AsyncMock(return_value=None)
    return server


# Disable tracing for tests
@pytest.fixture(autouse=True)
def disable_tracing():
    """Disable agent tracing during tests."""
    try:
        from agents.tracing import set_tracing_disabled

        set_tracing_disabled(True)
        yield
    except ImportError:
        # If tracing module doesn't exist, continue
        yield


# Mock the biomcp server startup for tests
@pytest.fixture(autouse=True)
def mock_biomcp_server():
    """Mock the BioMCP server to avoid actual server startup during tests."""
    import subprocess
    from unittest.mock import patch

    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Process is running
    mock_process.communicate.return_value = ("Server started", "")

    with patch("subprocess.Popen", return_value=mock_process):
        yield mock_process


def make_mock_subagent(agent_name, response_text):
    """Create a mock sub-agent with a specific name and output."""
    from unittest.mock import MagicMock

    mock_agent = MagicMock()
    mock_agent._agent = MagicMock()
    mock_agent._agent.name = agent_name
    mock_agent._agent.model = "mock-model"
    mock_agent._agent.instructions = f"Instructions for {agent_name}"
    mock_agent._agent.handoffs = []
    # The agent framework expects a ._agent attribute for handoffs
    return mock_agent
