"""Renders the Streamlit page headlessly (no browser, no network, no API calls)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import evidence as ev  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")

CI = {"intervention": "VR", "comparator": "traditional", "outcome": "skills",
      "required_outcome_level": 3, "population": "residents"}


def _paper(title, **kw):
    return {"title": title, "abstract": "VR improved skills. It worked.", "year": 2024, "authors": ["A B"],
            "venue": "J", "sources": ["pubmed"], "citations": None, "doi": "10.1/x", "url": "http://x",
            "pdf_url": None, **kw}


def _result(label, **kw):
    return {"label": label, "strength": "strong", "directness": "direct", "max_outcome_level": 3,
            "study_design": "rct", "sample_size": 20, "population": "res", "intervention": "VR",
            "intervention_match": "exact", "comparator": "x", "comparator_match": "matches",
            "outcomes": [{"name": "OSATS", "level": 3, "result": "improved"}], "rationale": "because",
            "limitations": ["small"], "quotes_verified": ["VR improved skills."], "quotes_unverified": ["fake"],
            "guardrail_notes": ["A -> B: why"], **kw}


def test_start_page_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception


def test_results_page_renders_papers_summary_and_error_rows():
    papers = [_paper("T1"), _paper("T2", round=2)]
    results = {0: _result("SUPPORTS"),
               1: {"label": "ERROR", "strength": "n/a", "directness": "n/a", "max_outcome_level": 0,
                   "rationale": "boom", "guardrail_notes": [], "quotes_verified": [], "quotes_unverified": [],
                   "limitations": [], "outcomes": []}}
    deeper = {"query": '("vr"[tiab])', "explanation": "targets outcomes", "papers": [papers[1]],
              "results": {1: results[1]}, "errors": ["PubMed: down"], "found": 3}
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["papers"] = papers
    at.session_state["claim"] = "c"
    at.session_state["analysis"] = {"claim_info": CI, "results": results,
                                    "summary": ev.summarize(CI, results), "deeper": deeper}
    at.run()
    assert not at.exception
    assert any("failed to analyze" in e.value for e in at.error)       # ERROR paper is surfaced
    assert any("Second search" in e.value for e in at.error)           # failed second search shown
    assert len(at.metric) == len(ev.LABELS)
