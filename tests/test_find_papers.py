"""Offline tests for search, retry/error handling, parsing, merging and routing (no network)."""
import os

import pytest
import requests

import find_papers as fp


class FakeResp:
    def __init__(self, status=200, json_data=None, content=b"", headers=None, text=""):
        self.status_code = status
        self._json = json_data
        self.content = content
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fp.time, "sleep", lambda s: None)


def _sequence(monkeypatch, items):
    """Make requests.get return/raise the given items in order; returns the call log."""
    calls, it = [], iter(items)

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(fp.requests, "get", fake_get)
    return calls


# ------------------------------------------------------------------ http_get
def test_http_get_retries_429_then_succeeds(monkeypatch):
    calls = _sequence(monkeypatch, [FakeResp(429, headers={"Retry-After": "0"}), FakeResp(200)])
    assert fp.http_get("X", "u", {}).status_code == 200
    assert len(calls) == 2


def test_http_get_does_not_retry_client_errors(monkeypatch):
    calls = _sequence(monkeypatch, [FakeResp(403, text="forbidden")])
    with pytest.raises(fp.SearchError, match="HTTP 403"):
        fp.http_get("X", "u", {})
    assert len(calls) == 1


def test_http_get_gives_up_after_timeouts(monkeypatch):
    calls = _sequence(monkeypatch, [requests.Timeout()] * 3)
    with pytest.raises(fp.SearchError, match="gave up"):
        fp.http_get("X", "u", {})
    assert len(calls) == 3


def test_http_get_gives_up_on_connection_errors(monkeypatch):
    _sequence(monkeypatch, [requests.ConnectionError()] * 3)
    with pytest.raises(fp.SearchError, match="connection error"):
        fp.http_get("X", "u", {})


def test_http_get_gives_up_on_persistent_500(monkeypatch):
    _sequence(monkeypatch, [FakeResp(500)] * 3)
    with pytest.raises(fp.SearchError, match="gave up"):
        fp.http_get("X", "u", {})


def test_http_get_unretryable_request_exception(monkeypatch):
    calls = _sequence(monkeypatch, [requests.exceptions.InvalidURL("bad")])
    with pytest.raises(fp.SearchError, match="request failed"):
        fp.http_get("X", "u", {})
    assert len(calls) == 1


# ---------------------------------------------------------------- Semantic Scholar
S2_REC = {"title": "T", "abstract": "A", "year": 2024, "authors": [{"name": "X Y"}], "venue": "V",
          "citationCount": 3, "externalIds": {"DOI": "10.1/ABC", "PubMed": "123"},
          "openAccessPdf": {"url": "http://pdf"}, "url": "http://s2"}


def test_s2_parses_records_and_skips_malformed(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "k")
    null_fields = {"title": "T2", "authors": None, "externalIds": None, "openAccessPdf": None}
    monkeypatch.setattr(fp, "http_get", lambda *a, **k: FakeResp(200, {"data": [S2_REC, null_fields, "garbage"]}))
    papers = fp.search_semantic_scholar("q", 5)
    assert [p["title"] for p in papers] == ["T", "T2"]  # "garbage" skipped, nulls tolerated
    assert papers[0]["doi"] == "10.1/ABC" and papers[0]["pdf_url"] == "http://pdf" and papers[0]["citations"] == 3


def test_s2_invalid_json_raises_search_error(monkeypatch):
    monkeypatch.setattr(fp, "http_get", lambda *a, **k: FakeResp(200, None))
    with pytest.raises(fp.SearchError, match="invalid JSON"):
        fp.search_semantic_scholar("q", 5)


def test_s2_rejected_key_falls_back_to_no_key(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "bad-key")
    seen = []

    def fake(name, url, params, headers=None, **k):
        seen.append(dict(headers or {}))
        if len(seen) == 1:
            raise fp.SearchError("SemanticScholar: HTTP 403: 'bad key'")
        return FakeResp(200, {"data": [S2_REC]})

    monkeypatch.setattr(fp, "http_get", fake)
    assert len(fp.search_semantic_scholar("q", 5)) == 1
    assert "x-api-key" in seen[0] and "x-api-key" not in seen[1]


def test_s2_other_errors_propagate(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "k")
    monkeypatch.setattr(fp, "http_get", lambda *a, **k: (_ for _ in ()).throw(fp.SearchError("HTTP 500 boom")))
    with pytest.raises(fp.SearchError):
        fp.search_semantic_scholar("q", 5)


# ------------------------------------------------------------------------ PubMed
PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
   <Journal><Title>Journal One</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
   <ArticleTitle>First <i>paper</i> title</ArticleTitle>
   <Abstract><AbstractText Label="BACKGROUND">Bg text.</AbstractText><AbstractText Label="RESULTS">Result text.</AbstractText></Abstract>
   <AuthorList><Author><LastName>Smith</LastName><ForeName>Ann</ForeName></Author></AuthorList>
 </Article></MedlineCitation>
 <PubmedData><ArticleIdList><ArticleId IdType="pubmed">111</ArticleId><ArticleId IdType="doi">10.1/one</ArticleId></ArticleIdList></PubmedData>
 </PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID>222</PMID><Article>
   <Journal><Title>Journal Two</Title><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
   <ArticleTitle>Second paper</ArticleTitle>
 </Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def _pubmed_http(idlist, content=PUBMED_XML, esearch_json=None):
    def fake(name, url, params, headers=None, **k):
        if name == "PubMed/esearch":
            return FakeResp(200, esearch_json if esearch_json is not None else {"esearchresult": {"idlist": idlist}})
        return FakeResp(200, content=content)
    return fake


def test_pubmed_parses_structured_abstract_year_doi_authors(monkeypatch):
    monkeypatch.setattr(fp, "http_get", _pubmed_http(["111", "222"]))
    p1, p2 = fp.search_pubmed("q", 5)
    assert p1["title"] == "First paper title"
    assert p1["abstract"] == "BACKGROUND: Bg text.\nRESULTS: Result text."
    assert (p1["doi"], p1["pmid"], p1["year"], p1["authors"]) == ("10.1/one", "111", 2020, ["Ann Smith"])
    assert p2["abstract"] is None and p2["year"] == 2019 and p2["doi"] is None
    assert p2["url"].endswith("/222/")


def test_pubmed_no_results_skips_efetch(monkeypatch):
    calls = []

    def fake(name, *a, **k):
        calls.append(name)
        return FakeResp(200, {"esearchresult": {"idlist": []}})

    monkeypatch.setattr(fp, "http_get", fake)
    assert fp.search_pubmed("q", 5) == []
    assert calls == ["PubMed/esearch"]


def test_pubmed_unexpected_esearch_response(monkeypatch):
    monkeypatch.setattr(fp, "http_get", _pubmed_http([], esearch_json={"oops": 1}))
    with pytest.raises(fp.SearchError, match="unexpected response"):
        fp.search_pubmed("q", 5)


def test_pubmed_invalid_xml(monkeypatch):
    monkeypatch.setattr(fp, "http_get", _pubmed_http(["1"], content=b"<not xml"))
    with pytest.raises(fp.SearchError, match="invalid XML"):
        fp.search_pubmed("q", 5)


# ------------------------------------------------------------- routing and merge
def test_is_medical_routing():
    assert fp.is_medical("intermittent fasting improves insulin sensitivity")
    assert not fp.is_medical("transformer models improve machine translation quality")


def test_key_prefers_doi_then_pmid_then_title():
    assert fp._key({"doi": "10.1/ABC", "pmid": "1"}) == "doi:10.1/abc"
    assert fp._key({"pmid": "5"}) == "pmid:5"
    assert fp._key({"title": "Hello, World!"}) == "title:helloworld"


def test_merge_dedupes_and_fills_missing_fields():
    a = {"source": "pubmed", "title": "X", "doi": "10.1/A", "abstract": None, "citations": None}
    b = {"source": "semanticscholar", "title": "X", "doi": "10.1/a", "abstract": "abs", "citations": 9}
    out = fp.merge([a], [b], [{"source": "pubmed", "title": "Other", "doi": None, "pmid": "7"}])
    assert len(out) == 2
    merged = out[0]
    assert merged["abstract"] == "abs" and merged["citations"] == 9
    assert merged["sources"] == ["pubmed", "semanticscholar"]


def _paper(src, title="P"):
    return {"source": src, "title": title, "doi": None, "pmid": None}


def test_run_search_medical_uses_pubmed_only(monkeypatch):
    s2_called = []
    monkeypatch.setattr(fp, "search_semantic_scholar", lambda q, n: s2_called.append(1) or [])
    monkeypatch.setattr(fp, "search_pubmed", lambda q, n: [_paper("pubmed")])
    papers, errors = fp.run_search("q", 5, medical=True, enrich=False)
    assert len(papers) == 1 and errors == [] and not s2_called


def test_run_search_one_source_failing_keeps_the_other(monkeypatch):
    def boom(q, n):
        raise fp.SearchError("S2 down")

    monkeypatch.setattr(fp, "search_semantic_scholar", boom)
    monkeypatch.setattr(fp, "search_pubmed", lambda q, n: [_paper("pubmed")])
    papers, errors = fp.run_search("q", 5, medical=False, enrich=False)
    assert len(papers) == 1 and len(errors) == 1 and "S2 down" in errors[0]


def test_run_search_unexpected_exception_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(fp, "search_semantic_scholar", lambda q, n: (_ for _ in ()).throw(ValueError("weird")))
    monkeypatch.setattr(fp, "search_pubmed", lambda q, n: [_paper("pubmed")])
    papers, errors = fp.run_search("q", 5, medical=False, enrich=False)
    assert len(papers) == 1 and "unexpected error" in errors[0]


def test_run_search_gives_each_source_its_own_query(monkeypatch):
    got = {}
    monkeypatch.setattr(fp, "search_semantic_scholar", lambda q, n: got.setdefault("s2", q) and [])
    monkeypatch.setattr(fp, "search_pubmed", lambda q, n: got.setdefault("pm", q) and [])
    fp.run_search({"pubmed": '"vr"[tiab]', "semanticscholar": "vr training"}, 5, medical=False, enrich=False)
    assert got == {"s2": "vr training", "pm": '"vr"[tiab]'}


def test_run_search_still_accepts_a_single_string(monkeypatch):
    got = []
    monkeypatch.setattr(fp, "search_semantic_scholar", lambda q, n: got.append(q) or [])
    monkeypatch.setattr(fp, "search_pubmed", lambda q, n: got.append(q) or [])
    fp.run_search("one query", 5, medical=False, enrich=False)
    assert got == ["one query", "one query"]


# ----------------------------------------------------------------------- .env
def test_load_env_reads_values_and_skips_empty(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("# comment\nTEST_FP_KEY='abc'\nTEST_FP_EMPTY=\n", encoding="utf-8")
    monkeypatch.setattr(fp, "BASE_DIR", str(tmp_path))
    try:
        fp.load_env()
        assert os.environ.get("TEST_FP_KEY") == "abc"
        assert "TEST_FP_EMPTY" not in os.environ
    finally:
        os.environ.pop("TEST_FP_KEY", None)


def test_load_env_missing_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(fp, "BASE_DIR", str(tmp_path))
    fp.load_env()  # must not raise


# ----------------------------------------------------------- citation enrichment
def _papers_for_enrich():
    return [{"doi": "10.1/a", "pmid": "1", "citations": None, "pdf_url": None},
            {"doi": None, "pmid": "2", "citations": None, "pdf_url": None},
            {"doi": "10.1/c", "citations": 7, "pdf_url": None}]


def test_enrich_fills_citations_and_pdf_only_where_missing(monkeypatch):
    captured = {}

    def fake_post(url, params=None, json=None, headers=None, timeout=None):
        captured["ids"] = json["ids"]
        return FakeResp(200, [{"citationCount": 5, "openAccessPdf": {"url": "http://p"}}, None])

    monkeypatch.setattr(fp.requests, "post", fake_post)
    papers = _papers_for_enrich()
    fp.enrich_citations(papers)
    assert captured["ids"] == ["DOI:10.1/a", "PMID:2"]
    assert papers[0]["citations"] == 5 and papers[0]["pdf_url"] == "http://p"
    assert papers[1]["citations"] is None  # not found upstream
    assert papers[2]["citations"] == 7  # already had a count: untouched


def test_enrich_never_raises_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(fp.requests, "post", boom)
    papers = _papers_for_enrich()
    fp.enrich_citations(papers)
    assert papers[0]["citations"] is None


def test_enrich_gives_up_quietly_on_rate_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(fp.requests, "post", lambda *a, **k: calls.append(1) or FakeResp(429))
    fp.enrich_citations(_papers_for_enrich())
    assert len(calls) == 3


def test_enrich_does_not_retry_client_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(fp.requests, "post", lambda *a, **k: calls.append(1) or FakeResp(400))
    fp.enrich_citations(_papers_for_enrich())
    assert len(calls) == 1
