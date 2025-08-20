"""
PubMed data models and parsing utilities.

This module defines Pydantic models representing PubMed article metadata and
provides robust parsing utilities to extract structured data from semi-structured
text outputs produced by external tools or LLMs. Parsing logic is implemented
with careful handling of edge cases such as multi-line fields, optional values,
and varying formatting.

Public API:
- Classes: PubMedArticle, PubMedArticleDetails
- Functions: parse_articles, parse_article_details (async)

Design notes:
- Parsing is implemented in a dedicated PubMedParser class for testability and
  separation of concerns. Top-level async functions are thin wrappers that
  preserve the existing asynchronous API used elsewhere in the codebase.
"""

from __future__ import annotations

import re
from typing import List

from loguru import logger
from pydantic import BaseModel, Field


class PubMedArticle(BaseModel):
    """Lightweight summary of a PubMed article suitable for search listings."""

    pmid: str = Field(description="The PubMed ID of the article")
    pmcid: str = Field(default="", description="The PubMed Central ID of the article")
    title: str = Field(default="", description="The title of the article")
    journal: str = Field(default="", description="The journal of the article")
    conference: str = Field(default="", description="The conference of the article")
    year: int = Field(default=0, description="The year of the article")
    authors: List[str] = Field(default=[], description="The authors of the article")
    abstract: str = Field(default="", description="The abstract of the article")
    url: str = Field(default="", description="The URL of the article")
    doi: str = Field(default="", description="The DOI of the article")


class PubMedArticleDetails(BaseModel):
    """Detailed view of a PubMed article suitable for reading context."""

    pmid: str = Field(description="The PubMed ID of the article")
    title: str = Field(default="", description="The title of the article")
    journal: str = Field(default="", description="The journal of the article")
    conference: str = Field(default="", description="The conference of the article")
    year: int = Field(default=0, description="The year of the article")
    authors: List[str] = Field(default=[], description="The authors of the article")
    abstract: str = Field(default="", description="The abstract of the article")
    full_text: str = Field(default="", description="The full text of the article")
    pubmed_url: str = Field(default="", description="The URL of the article on PubMed")
    pmc_url: str = Field(default="", description="The URL of the article on PubMed Central")


class PubMedParser:
    """Utility class for parsing PubMed-related text into structured models.

    Methods accept free-form text produced by external systems and extract
    fields using conservative regular expressions. Where possible, whitespace is
    normalized while preserving the structure of multi-paragraph text.
    """

    RECORD_SPLIT_PATTERN = r"# Record \d+\s*\n"

    @staticmethod
    def _extract_year_from_date(date_str: str) -> int:
        match = re.search(r"(\d{4})", date_str)
        if not match:
            return 0
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Invalid year format in date: {date_str}")
            return 0

    @staticmethod
    def _split_authors(authors_str: str) -> List[str]:
        if not authors_str:
            return []
        authors: List[str] = []
        for author in authors_str.split(','):
            author = author.strip()
            if author and author != '...':
                authors.append(author)
        return authors

    @classmethod
    def parse_articles(cls, text: str) -> List[PubMedArticle]:
        """Parse a search result listing into PubMedArticle items.

        The input is expected to contain one or more records starting with
        a line formatted like "# Record N" followed by key/value fields
        (Pmid, Pmcid, Title, Journal/Conference, Date, Doi, Abstract,
        Pubmed Url, Authors). Missing fields are tolerated.
        """
        if not text:
            return []

        articles: List[PubMedArticle] = []
        records = re.split(cls.RECORD_SPLIT_PATTERN, text)

        for record in records:
            if not record or 'Pmid:' not in record:
                continue
            try:
                pmid_match = re.search(r"Pmid:\s*(\d+)", record)
                pmid = pmid_match.group(1) if pmid_match else ""

                pmcid_match = re.search(r"Pmcid:\s*(\w+)", record)
                pmcid = pmcid_match.group(1) if pmcid_match else ""

                title_match = re.search(r"Title:\s*\n?(.*?)(?=\n(?:Journal|Conference):)", record, re.DOTALL)
                title = title_match.group(1).strip().replace('\n', ' ') if title_match else ""

                journal_match = re.search(r"Journal:\s*(.*?)(?=\n)", record)
                conference_match = re.search(r"Conference:\s*(.*?)(?=\n)", record)
                journal = journal_match.group(1).strip() if journal_match else ""
                conference = conference_match.group(1).strip() if conference_match else ""

                date_match = re.search(r"Date:\s*(.*?)(?=\n|$)", record)
                year = cls._extract_year_from_date(date_match.group(1).strip()) if date_match else 0

                doi_match = re.search(r"Doi:\s*(.*?)(?=\n)", record)
                doi = doi_match.group(1).strip() if doi_match else ""

                abstract_match = re.search(r"Abstract:\s*\n?(.*?)(?=\nPubmed Url:|$)", record, re.DOTALL)
                abstract = abstract_match.group(1).strip() if abstract_match else ""

                url_match = re.search(r"Pubmed Url:\s*(.*?)(?=\n)", record)
                url = url_match.group(1).strip() if url_match else ""

                authors_match = re.search(r"Authors:\s*(.*?)(?=\n|$)", record)
                authors = cls._split_authors(authors_match.group(1).strip()) if authors_match else []

                article = PubMedArticle(
                    pmid=pmid,
                    pmcid=pmcid,
                    title=title,
                    journal=journal,
                    conference=conference,
                    year=year,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    doi=doi,
                )
                articles.append(article)
            except Exception as e:
                logger.error(f"Error parsing article record: {e}")
                continue
        return articles

    @classmethod
    def parse_article_details(cls, text: str) -> PubMedArticleDetails:
        """Parse a detailed article view into PubMedArticleDetails.

        Raises ValueError if the required Pmid field is not present.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("Input text is empty")

        article_data = {
            "pmid": "",
            "title": "",
            "journal": "",
            "conference": "",
            "year": 0,
            "authors": [],
            "abstract": "",
            "full_text": "",
            "pubmed_url": "",
            "pmc_url": "",
        }

        pmid_match = re.search(r"Pmid:\s*(\d+)", text, re.IGNORECASE)
        if not pmid_match:
            raise ValueError("Required PMID field not found in text")
        article_data["pmid"] = pmid_match.group(1)

        title_match = re.search(
            r"Title:\s*\n?(.*?)(?=\n(?:Abstract|Journal|Conference|Date|Pmid|Authors|Full Text|$))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1).strip())
            article_data["title"] = title

        journal_match = re.search(r"Journal:\s*(.*?)(?=\n)", text, re.IGNORECASE)
        if journal_match:
            article_data["journal"] = journal_match.group(1).strip()
        else:
            conference_match = re.search(r"Conference:\s*(.*?)(?=\n)", text, re.IGNORECASE)
            if conference_match:
                article_data["conference"] = conference_match.group(1).strip()

        date_match = re.search(r"Date:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
        if date_match:
            article_data["year"] = cls._extract_year_from_date(date_match.group(1).strip())

        abstract_match = re.search(
            r"Abstract:\s*\n?(.*?)(?=\n(?:Full Text|Pubmed Url|Pmc Url|Authors|$))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            abstract = re.sub(r"\n\s*\n", "\n\n", abstract)
            article_data["abstract"] = abstract

        full_text_match = re.search(
            r"Full Text:\s*\n?(.*?)(?=\n(?:Pubmed Url|Pmc Url|Authors|$))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if full_text_match:
            full_text = full_text_match.group(1).strip()
            full_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", full_text)
            article_data["full_text"] = full_text

        pubmed_url_match = re.search(r"Pubmed Url:\s*(https?://[^\s\n]+)", text, re.IGNORECASE)
        if pubmed_url_match:
            article_data["pubmed_url"] = pubmed_url_match.group(1).strip()

        pmc_url_match = re.search(r"Pmc Url:\s*(https?://[^\s\n]+)", text, re.IGNORECASE)
        if pmc_url_match:
            article_data["pmc_url"] = pmc_url_match.group(1).strip()

        authors_match = re.search(r"Authors:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
        if authors_match:
            article_data["authors"] = cls._split_authors(authors_match.group(1).strip())

        return PubMedArticleDetails(**article_data)


async def parse_articles(text: str) -> List[PubMedArticle]:
    """Async wrapper for PubMedParser.parse_articles.

    Provided to maintain backward compatibility with existing async call sites.
    """

    return PubMedParser.parse_articles(text)


async def parse_article_details(text: str) -> PubMedArticleDetails:
    """Async wrapper for PubMedParser.parse_article_details.

    Provided to maintain backward compatibility with existing async call sites.
    """

    try:
        return PubMedParser.parse_article_details(text)
    except ValueError as ve:
        logger.error(f"Validation error in parse_article_details: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in parse_article_details: {e}")
        raise Exception(f"Failed to parse article details: {str(e)}") from e


__all__ = [
    "PubMedArticle",
    "PubMedArticleDetails",
    "PubMedParser",
    "parse_articles",
    "parse_article_details",
]


