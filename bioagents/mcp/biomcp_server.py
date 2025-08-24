# ------------------------------------------------------------------------------
# biomcp_server.py
#
# This file provides the MCP server for the BioMCP agent.
#
# Author: Theodore Mui
# Date: 2025-06-11
# ------------------------------------------------------------------------------

import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from datetime import datetime
import os
import time
from typing import List
from loguru import logger
from bioagents.models.pubmed import (
    PubMedArticle,
    PubMedArticleDetails,
    parse_articles,
    parse_article_details,
)

import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# Create server
mcp = FastMCP("BioMCP Server")
mcp.settings.port = int(os.environ.get("BIOMCP_PORT", "8132"))

# --------------------------------
# Tools
# --------------------------------


@mcp.tool()
async def list_tools() -> list[str]:
    """
    List all available tools.
    """

    logger.info("Listing available tools")

    server_params = StdioServerParameters(
        command="uv", args=["run", "--with", "biomcp-python", "biomcp", "run"]
    )
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        response_str = ""
        try:
            tool_result = await session.list_tools()
            response_str = f"Available tools: {[t.name for t in tool_result.tools]}"
        except Exception as e:
            logger.error(f"Error: {e}")
            response_str = f"list_tools: {e}"
        return response_str


@mcp.tool()
async def get_variant_details(variant_id: str) -> str:
    """
    Get details of a variant.

    Args:
        variant_id (str): The ID of the variant
    """

    logger.info(f"Getting details of variant with ID: {variant_id}")

    server_params = StdioServerParameters(
        command="uv", args=["run", "--with", "biomcp-python", "biomcp", "run"]
    )
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        # Initialize the session
        await session.initialize()

        response_str = ""
        try:
            result = await session.call_tool(
                "variant_details",
                {
                    "call_benefit": "Understand the variant details",
                    "variant_id": variant_id,
                },
            )
            if not result.isError and result.content:
                # Access the text content from the first content block
                response_str = result.content[0].text
        except Exception as e:
            logger.error(f"Error: {e}")
            response_str = f"get_variant_details: {e}"
        return response_str


@mcp.tool()
async def get_article_details(pmid: str) -> str:
    """
    Get details of a PubMed article.

    Args:
        pmid (str): The PubMed ID of the article
    """

    logger.info(f"Getting details of article with PMID: {pmid}")

    server_params = StdioServerParameters(
        command="uv", args=["run", "--with", "biomcp-python", "biomcp", "run"]
    )
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        response_str = ""
        try:
            result = await session.call_tool(
                "article_details",
                {"call_benefit": "SGet details of a PubMed article", "pmid": pmid},
            )
            if not result.isError and result.content:
                response_str = result.content[0].text
        except Exception as e:
            logger.error(f"Error: {e}")
            response_str = f"get_article_details: {e}"
        return response_str


@mcp.tool()
async def article_searcher(
    diseases: List[str] = [], keywords: List[str] = [], genes: List[str] = []
) -> str:
    """
    Search PubMed articles using structured criteria.

    Args:
        diseases (List[str]): List of diseases to search for
        keywords (List[str]): List of keywords to search for
        genes (List[str]): List of genes to search for

    """
    server_params = StdioServerParameters(
        command="uv", args=["run", "--with", "biomcp-python", "biomcp", "run"]
    )
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        logger.info(
            f"Searching for articles with diseases: {diseases}, keywords: {keywords}, genes: {genes}"
        )

        await session.initialize()
        response_str = ""
        try:
            result = await session.call_tool(
                "article_searcher",
                {
                    "call_benefit": "Search PubMed articles using structured criteria",
                    "diseases": diseases,
                    "keywords": keywords,
                    "genes": genes,
                },
            )
            if not result.isError and result.content:
                response_str = result.content[0].text
        except Exception as e:
            logger.error(f"Error: {e}")
            response_str = f"get_articles: {e}"
        return response_str


@mcp.tool()
async def trial_searcher(
    conditions: List[str] = [],
    terms: List[str] = [],
    interventions: List[str] = [],
    recruitment_status: str = "",
    study_type: str = "",
    nct_ids: List[str] = [],
    lat: float = 0.0,
    long: float = 0.0,
    distance: float = 0.0,
    min_date: str = "",
    max_date: str = "",
    date_field: str = "",
    phase: str = "",
    age_group: str = "",
    primary_purpose: str = "",
    intervention_type: str = "",
    study_design: str = "",
) -> str:
    """
    Search clinical trials using structured criteria.

    Args:
        conditions (List[str]): List of conditions to search for
        terms (List[str]): List of terms to search for
        interventions (List[str]): List of interventions to search for
        recruitment_status (str): The recruitment status of the trial
        study_type (str): The type of the trial
        nct_ids (List[str]): List of NCT IDs to search for
        lat (float): The latitude of the trial
        long (float): The longitude of the trial
        distance (float): The distance from the latitude and longitude
        min_date (str): The minimum date of the trial
        max_date (str): The maximum date of the trial
        date_field (str): The field of the date
        phase (str): The phase of the trial
        age_group (str): The age group of the trial
        primary_purpose (str): The primary purpose of the trial
        intervention_type (str): The type of the intervention
        study_design (str): The design of the trial
    """

    server_params = StdioServerParameters(
        command="uv", args=["run", "--with", "biomcp-python", "biomcp", "run"]
    )
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        logger.info(
            f"Searching for clinical trials with conditions: {conditions}, terms: {terms}, phase: {phase}, age_group: {age_group}"
        )

        await session.initialize()
        response_str = ""
        try:
            result = await session.call_tool(
                "trial_searcher",
                {
                    "call_benefit": "Get information about clinical trials and information based on specified criteria.",
                    "conditions": conditions,
                    "terms": terms,
                    "interventions": interventions,
                    "recruitment_status": recruitment_status,
                    "study_type": study_type,
                    "nct_ids": nct_ids,
                    "lat": lat,
                    "long": long,
                    "distance": distance,
                    "min_date": min_date,
                    "max_date": max_date,
                    "date_field": date_field,
                    "phase": phase,
                    "age_group": age_group,
                    "primary_purpose": primary_purpose,
                    "intervention_type": intervention_type,
                    "study_design": study_design,
                    "primary_purpose": primary_purpose,
                    "intervention_type": intervention_type,
                    "study_design": study_design,
                },
            )
            if not result.isError and result.content:
                response_str = result.content[0].text
        except Exception as e:
            logger.error(f"Error: {e}")
            response_str = f"get_articles: {e}"
        return response_str


## PubMed models and parsing utilities are imported from bioagents.models.pubmed
# --------------------------------
# Streamable HTTPMCP Server
# --------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
