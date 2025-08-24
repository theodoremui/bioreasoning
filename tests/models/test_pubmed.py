import pytest

from bioagents.models.pubmed import (
    PubMedArticle,
    PubMedArticleDetails,
    PubMedParser,
    parse_articles,
    parse_article_details,
)


def test_parse_articles_empty_input():
    articles = PubMedParser.parse_articles("")
    assert articles == []


def test_parse_articles_basic_record():
    text = (
        "# Record 1\n"
        "Pmid: 123456\n"
        "Pmcid: PMC123\n"
        "Title: Example Title\n"
        "Journal: J Test\n"
        "Date: 2024 May\n"
        "Doi: 10.1000/example\n"
        "Abstract: This is a short abstract.\n"
        "Pubmed Url: https://pubmed.ncbi.nlm.nih.gov/123456\n"
        "Authors: Alice A., Bob B.\n"
    )
    articles = PubMedParser.parse_articles(text)
    assert len(articles) == 1
    a = articles[0]
    assert a.pmid == "123456"
    assert a.pmcid == "PMC123"
    assert a.title == "Example Title"
    assert a.journal == "J Test"
    assert a.conference == ""
    assert a.year == 2024
    assert a.doi == "10.1000/example"
    assert a.url.endswith("/123456")
    assert a.authors == ["Alice A.", "Bob B."]


@pytest.mark.asyncio
async def test_parse_articles_async_wrapper():
    text = "# Record 1\n" "Pmid: 1\n" "Title: T\n" "Journal: J\n"
    articles = await parse_articles(text)
    assert len(articles) == 1
    assert isinstance(articles[0], PubMedArticle)


def test_parse_article_details_requires_pmid():
    with pytest.raises(ValueError):
        PubMedParser.parse_article_details("Title: Missing PMID")


def test_parse_article_details_basic():
    text = (
        "Pmid: 999\n"
        "Title: Long title wrapping\n"
        "Abstract: Para1.\n\nPara2.\n"
        "Full Text: Body\n\nMore body\n"
        "Pubmed Url: https://pubmed.ncbi.nlm.nih.gov/999\n"
        "Pmc Url: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC999\n"
        "Authors: X, Y, ...\n"
        "Date: 2020 Jan\n"
    )
    details = PubMedParser.parse_article_details(text)
    assert details.pmid == "999"
    assert details.title.startswith("Long title")
    assert details.year == 2020
    assert details.pubmed_url.endswith("/999")
    assert details.pmc_url.endswith("PMC999")
    assert details.authors == ["X", "Y"]


@pytest.mark.asyncio
async def test_parse_article_details_async_wrapper():
    text = "Pmid: 42\n" "Title: T\n" "Journal: J\n" "Date: 2023\n"
    details = await parse_article_details(text)
    assert isinstance(details, PubMedArticleDetails)
    assert details.pmid == "42"
