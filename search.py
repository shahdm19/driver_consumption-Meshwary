"""Tavily web-search wrapper.

Replaces the original ``_tavily_search`` + ``search_web`` + ``multi_query_search``
trio.  ``search_web`` was a one-line wrapper around ``_tavily_search`` — both
are consolidated into :meth:`WebSearcher.search`.  :meth:`multi_query_search`
preserves the original multi-query fan-out and early-stop logic.
"""
from __future__ import annotations

import traceback
from typing import List, Optional

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE: bool = True
except ImportError:
    TAVILY_AVAILABLE = False

from config import (
    MULTI_QUERY_MIN_CONTEXTS,
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
    TAVILY_SEARCH_DEPTH,
)
from logging_config import get_logger

logger = get_logger(__name__)


class WebSearcher:
    """Wraps the Tavily client.  Falls back to empty context if unavailable."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.client = None
        if TAVILY_AVAILABLE and api_key:
            self.client = TavilyClient(api_key=api_key)
            logger.info("Tavily search initialized")
        elif not TAVILY_AVAILABLE:
            logger.warning(
                "Tavily not installed. Install with: pip install tavily-python"
            )
        else:
            logger.warning("TAVILY_API_KEY not set. Search will not work.")

    @property
    def available(self) -> bool:
        """True iff a Tavily client is wired up (used by /health)."""
        return self.client is not None

    # ---- single query ------------------------------------------------------

    def search(self, query: str) -> str:
        """Run a single Tavily search and return a formatted context string.

        Returns ``""`` if the client is unavailable or the call fails.  This
        method is the SINGLE entry point for raw Tavily access — there is no
        longer a separate ``_tavily_search`` + ``search_web`` pair.
        """
        if not self.client:
            return ""
        try:
            logger.info(" Tavily search: %s", query)
            response = self.client.search(
                query=query,
                search_depth=TAVILY_SEARCH_DEPTH,
                max_results=TAVILY_MAX_RESULTS,
                include_answer=True,
            )
            parts: List[str] = []
            if response.get("answer"):
                parts.append(f"[AI Summary]: {response['answer']}")
            for r in response.get("results", []):
                title = r.get("title", "")
                content = r.get("content", "")
                parts.append(f"Title: {title}\nContent: {content}")
            return "\n\n".join(parts)
        except Exception as e:
            logger.error("Tavily search failed: %s\n%s", e, traceback.format_exc())
            return ""

    # ---- multi query -------------------------------------------------------

    def multi_query_search(self, make: str, model: str, year: int) -> str:
        """Run several targeted queries and stop early once enough context.

        The query list and the early-stop threshold (``MULTI_QUERY_MIN_CONTEXTS``)
        are preserved exactly from the original ``multi_query_search``.
        """
        queries = [
            f'"{make} {model}" {year} engine displacement cc Egypt specifications',
            f"{make} {model} {year} مواصفات المحرك سعة سي سي مصر",
            f"{make} {model} {year} سعة الموتور سلندرات توربو مصر",
            f"{make} {model} {year} fuel consumption L/100km specs",
        ]

        contexts: List[str] = []
        for q in queries:
            ctx = self.search(q)
            if ctx:
                contexts.append(ctx)
            if len(contexts) >= MULTI_QUERY_MIN_CONTEXTS:
                break
        return "\n\n=====\n\n".join(contexts) if contexts else ""
