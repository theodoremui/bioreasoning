import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bioagents.mcp import biomcp_server as srv


@pytest.mark.asyncio
async def test_list_tools_happy_path():
    with (
        patch("bioagents.mcp.biomcp_server.stdio_client") as mock_stdio,
        patch("bioagents.mcp.biomcp_server.ClientSession") as MockSession,
    ):
        # Mock stdio_client context
        mock_read, mock_write = object(), object()
        mock_stdio.return_value.__aenter__.return_value = (mock_read, mock_write)
        mock_stdio.return_value.__aexit__.return_value = None

        # Mock session
        session = AsyncMock()
        session.initialize = AsyncMock()
        tool = MagicMock()
        tool.name = "tool_a"
        session.list_tools = AsyncMock(return_value=MagicMock(tools=[tool]))
        MockSession.return_value.__aenter__.return_value = session
        MockSession.return_value.__aexit__.return_value = None

        result = await srv.list_tools()
        assert "tool_a" in result


@pytest.mark.asyncio
async def test_get_variant_details_success():
    with (
        patch("bioagents.mcp.biomcp_server.stdio_client") as mock_stdio,
        patch("bioagents.mcp.biomcp_server.ClientSession") as MockSession,
    ):
        mock_stdio.return_value.__aenter__.return_value = (object(), object())
        mock_stdio.return_value.__aexit__.return_value = None

        session = AsyncMock()
        session.initialize = AsyncMock()
        content_block = MagicMock()
        content_block.text = "Variant details..."
        call_result = MagicMock(isError=False, content=[content_block])
        session.call_tool = AsyncMock(return_value=call_result)
        MockSession.return_value.__aenter__.return_value = session
        MockSession.return_value.__aexit__.return_value = None

        text = await srv.get_variant_details("rs123")
        assert "Variant details" in text


@pytest.mark.asyncio
async def test_article_searcher_and_parse_articles():
    with (
        patch("bioagents.mcp.biomcp_server.stdio_client") as mock_stdio,
        patch("bioagents.mcp.biomcp_server.ClientSession") as MockSession,
    ):
        mock_stdio.return_value.__aenter__.return_value = (object(), object())
        mock_stdio.return_value.__aexit__.return_value = None

        session = AsyncMock()
        session.initialize = AsyncMock()
        # Simulate MCP text response for search
        mcp_text = (
            "# Record 1\n"
            "Pmid: 123\nTitle: T\nJournal: J\nDate: 2024\nDoi: d\nPubmed Url: u\nAuthors: A, B\n"
        )
        content_block = MagicMock()
        content_block.text = mcp_text
        call_result = MagicMock(isError=False, content=[content_block])
        session.call_tool = AsyncMock(return_value=call_result)
        MockSession.return_value.__aenter__.return_value = session
        MockSession.return_value.__aexit__.return_value = None

        text = await srv.article_searcher(diseases=["x"])  # returns raw text
        articles = await srv.parse_articles(text)  # use exported async parser
        assert len(articles) == 1
        assert articles[0].pmid == "123"


@pytest.mark.asyncio
async def test_get_article_details_and_parse():
    with (
        patch("bioagents.mcp.biomcp_server.stdio_client") as mock_stdio,
        patch("bioagents.mcp.biomcp_server.ClientSession") as MockSession,
    ):
        mock_stdio.return_value.__aenter__.return_value = (object(), object())
        mock_stdio.return_value.__aexit__.return_value = None

        session = AsyncMock()
        session.initialize = AsyncMock()
        mcp_text = (
            "Pmid: 999\n"
            "Title: Long title\n"
            "Abstract: A\n\nB\n"
            "Full Text: FT\n\nMore\n"
            "Pubmed Url: https://pubmed.ncbi.nlm.nih.gov/999\n"
            "Pmc Url: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC999\n"
            "Authors: X, Y\n"
            "Date: 2020 Jan\n"
        )
        content_block = MagicMock()
        content_block.text = mcp_text
        call_result = MagicMock(isError=False, content=[content_block])
        session.call_tool = AsyncMock(return_value=call_result)
        MockSession.return_value.__aenter__.return_value = session
        MockSession.return_value.__aexit__.return_value = None

        raw = await srv.get_article_details("999")
        details = await srv.parse_article_details(raw)
        assert details.pmid == "999"
        assert details.year == 2020
