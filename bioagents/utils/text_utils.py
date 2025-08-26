###############################################################################
# text_utils.py
#
# Text utilities for snippet generation and related helpers.
#
# This module provides utilities to extract readable, contextually-relevant
# snippets from longer bodies of text. These are useful for previews and
# citations where we want to show a short excerpt that helps the user
# understand why a given document was retrieved.
#
# Functions
# ---------
# - make_contextual_snippet(text, query, max_length=100, boundary_window=15)
#
# Examples
# --------
# Basic usage with a long text and a query term:
#     >>> long_text = "The sky appears blue due to Rayleigh scattering... molecules ... wavelength ..."
#     >>> make_contextual_snippet(long_text, query="why is the sky blue", max_length=50)
#     '… sky appears blue due to Rayleigh scattering …'
#
# If no query token is found, the snippet defaults to the start of the text:
#     >>> make_contextual_snippet("First sentence. Second sentence.", query="unrelated", max_length=20)
#     'First sentence. …'
#
# If the text is shorter than max_length, it is returned as is:
#     >>> make_contextual_snippet("short text", query="anything", max_length=50)
#     'short text'
###############################################################################

import re
from typing import List, Tuple


def clean_text(text_to_clean: str) -> str:
    """
    Clean text by removing Markdown headings and collapsing whitespace.
    This is used to clean up the text before generating a snippet.

    Args:
        text_to_clean: The text to clean.

    Returns:
        The cleaned text.
    """
    # Remove Markdown heading markers at line starts (e.g., '#', '##', ...)
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", text_to_clean)

    # Normalize Markdown tables into comma-separated cells per line, end row with period.
    # Strategy:
    # - Remove header separators lines like |---|---| or ---|---
    # - For lines that contain '|' treat as table rows: split on '|', trim cells, join with ', '
    # - Remove leading/trailing pipes
    lines = cleaned.splitlines()
    normalized_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip markdown table separator rows (e.g., |---|---| or --- | ---)
        if re.fullmatch(r"\|?\s*:?[-]{3,}[:]?\s*(\|\s*:?[-]{3,}[:]?\s*)*\|?", stripped):
            continue
        if '|' in stripped:
            # Table row: split and join with commas
            parts = [p.strip() for p in stripped.strip('|').split('|') if p.strip()]
            if parts:
                normalized_lines.append(', '.join(parts) + '.')
            continue
        normalized_lines.append(stripped)

    cleaned = " ".join([ln for ln in normalized_lines if ln])
    # Collapse whitespace/newlines into single spaces
    return " ".join(cleaned.split())


def make_contextual_snippet(
    text: str,
    query: str,
    max_length: int = 100,
    boundary_window: int = 15,
) -> str:
    """Return a readable snippet up to ``max_length`` characters.

    The function tries to center the snippet around the first occurrence of a
    meaningful token derived from ``query`` (tokens of length >= 3). If no such
    token is found in ``text``, the snippet is taken from the beginning.

    The snippet is adjusted to nearest word boundaries (within ``boundary_window``)
    to avoid chopping words in half. Leading/trailing ellipses are added when
    the snippet is truncated from the start or the end respectively.

    Parameters
    ----------
    text : str
        The full text to extract a snippet from.
    query : str
        The user query used to locate a relevant region in the text.
    max_length : int, optional
        Target maximum length of the returned snippet, by default 100.
    boundary_window : int, optional
        Number of characters to look around the start/end for whitespace to
        align to word boundaries, by default 15.

    Returns
    -------
    str
        A snippet of up to ``max_length`` characters that is readable and
        contextually relevant to ``query`` when possible.
    """

    if not text:
        return ""

    text_stripped = text.strip()

    if len(text_stripped) <= max_length:
        return clean_text(text_stripped)

    # Tokenize query and search for first meaningful token (length >= 3)
    query_tokens: List[str] = [tok.lower() for tok in query.split() if len(tok) >= 3]
    lower_text = text_stripped.lower()
    hit_pos = -1
    for token in query_tokens:
        hit_pos = lower_text.find(token)
        if hit_pos != -1:
            break

    if hit_pos == -1:
        start = 0
    else:
        start = max(0, hit_pos - max_length // 2)

    end = min(len(text_stripped), start + max_length)

    # Adjust start to the previous space within boundary_window
    space_before = text_stripped.rfind(" ", max(0, start - boundary_window), start)
    if space_before != -1:
        start = space_before + 1

    # Adjust end to the last space within boundary_window after the desired end
    space_after = text_stripped.rfind(
        " ", start, min(len(text_stripped), end + boundary_window)
    )
    if space_after != -1 and space_after > start:
        end = min(space_after, start + max_length)

    snippet = text_stripped[start:end].strip()
    snippet = clean_text(snippet)

    # Add ellipses if we trimmed the original text
    if start > 0:
        snippet = "… " + snippet
    if end < len(text_stripped):
        snippet = snippet + " …"

    return snippet


def make_title_and_snippet(
    text: str,
    query: str,
    *,
    max_length: int = 100,
    title_max_length: int = 80,
    boundary_window: int = 15,
) -> Tuple[str, str]:
    """Return a representative (title, snippet) tuple for the given text.

    Title selection strategy (fast, heuristic, robust):
    - If the text begins with a Markdown-style heading ("# ", "## ", etc.) on the
      first line, use it (stripped of leading hashes) as the title.
    - Else, choose the sentence containing the first meaningful query token
      (>=3 chars) as the title; if none found, use the first sentence.
    - Trim the title at a reasonable length (``title_max_length``), respecting word
      boundaries, and add ellipsis if trimmed.

    Snippet is produced by ``make_contextual_snippet`` with the same query and
    boundary settings.

    Parameters are analogous to ``make_contextual_snippet`` with the addition of
    ``title_max_length``.
    """

    if not text:
        return "", ""

    # 1) Attempt to use a markdown-style heading as a title
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    heading_match = re.match(r"^(#{1,6})\s+(.*)$", first_line)
    if heading_match:
        raw_title = heading_match.group(2).strip()
    else:
        # 2) Otherwise pick sentence containing first query token (or first sentence)
        query_tokens: List[str] = [
            tok.lower() for tok in query.split() if len(tok) >= 3
        ]
        lower_text = text.lower()
        hit_pos = -1
        for token in query_tokens:
            hit_pos = lower_text.find(token)
            if hit_pos != -1:
                break

        # Find sentence boundaries around the hit (or from start)
        # Simple sentence boundary: . ! ? or newline
        boundary_regex = re.compile(r"[.!?]\s|\n")
        if hit_pos == -1:
            # First sentence until boundary
            match = boundary_regex.search(text)
            raw_title = text[: match.start()].strip() if match else text.strip()
        else:
            # Find start boundary back from hit_pos
            start_search = list(boundary_regex.finditer(text[:hit_pos]))
            start = start_search[-1].end() if start_search else 0
            # Find end boundary forward from hit_pos
            end_match = boundary_regex.search(text, pos=hit_pos)
            end = end_match.start() if end_match else len(text)
            raw_title = text[start:end].strip()

    # Normalize title: remove leading '#' and collapse repeated whitespace/newlines
    raw_title = re.sub(r"^\s*#+\s*", "", raw_title)
    raw_title = re.sub(r"\s+", " ", raw_title).strip()
    # Ensure title starts at a whole word:
    #  - drop leading non-alphanumeric punctuation
    raw_title = re.sub(r"^[^A-Za-z0-9]+", "", raw_title)
    #  - drop leading single-letter artifact (except valid one-letter words 'a'/'I')
    m_lead = re.match(r"^([A-Za-z])\b\s+(.*)$", raw_title)
    if m_lead and m_lead.group(1).lower() not in ("a", "i"):
        raw_title = m_lead.group(2).lstrip()

    # 3) Trim title to max length with word boundary preference
    title = raw_title
    if len(title) > title_max_length:
        # Cut then backtrack to last space
        cut = title[:title_max_length]
        sp = cut.rfind(" ")
        if sp > 0:
            cut = cut[:sp]
        title = cut.rstrip() + " …"

    # 4) Build snippet using existing helper
    snippet = make_contextual_snippet(
        text=text,
        query=query,
        max_length=max_length,
        boundary_window=boundary_window,
    )

    return title, snippet
