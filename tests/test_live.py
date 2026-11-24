"""Smoke tests against the REAL services. Skipped unless RUN_LIVE=1 (they need network and keys).

    RUN_LIVE=1 python -m pytest tests/test_live.py        (PowerShell: $env:RUN_LIVE=1)
"""
import os

import pytest

import evidence as ev
import find_papers as fp

pytestmark = pytest.mark.live


def test_pubmed_returns_papers_with_titles():
    papers = fp.search_pubmed("insulin resistance intermittent fasting", 3)
    assert papers and all(p["title"] for p in papers)


def test_semantic_scholar_search():
    try:
        papers = fp.search_semantic_scholar("transformer machine translation", 3)
    except fp.SearchError as e:  # free tier is often rate-limited; that is not a code bug
        pytest.skip(f"Semantic Scholar unavailable: {e}")
    assert isinstance(papers, list)


def test_claude_can_decompose_a_claim():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    info = ev.decompose_claim(ev._client(key), "claude-sonnet-5-5", "VR training improves surgical skills.")
    assert 1 <= info["required_outcome_level"] <= 5
