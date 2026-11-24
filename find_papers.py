"""Step 1: find candidate papers for a claim using Semantic Scholar + PubMed.

Usage:
    python find_papers.py "claim or search query" [--limit 20] [--out papers.json]

Logs go to the console and to logs/research.log (rotating).
"""
import argparse
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from logging.handlers import RotatingFileHandler

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "research.log")


def setup_logging() -> logging.Logger:
    """Idempotent (Streamlit re-runs the script on every interaction)."""
    logger = logging.getLogger("research")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(funcName)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    os.makedirs(LOG_DIR, exist_ok=True)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


log = setup_logging()


class SearchError(Exception):
    """A source failed in a way the caller should report (after retries)."""


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); doesn't override real env vars."""
    p = os.path.join(BASE_DIR, path)
    if not os.path.exists(p):
        log.warning(".env not found at %s", p)
        return
    loaded = []
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
                    loaded.append(k.strip())
    log.info(".env loaded; keys present: %s", loaded or "none (file empty?)")


load_env()


def s2_key() -> str | None:
    return os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")


MEDICAL_TERMS = {
    "disease", "disorder", "syndrome", "cancer", "tumor", "tumour", "oncology", "diabetes", "insulin",
    "cardiac", "cardiovascular", "hypertension", "stroke", "infection", "virus", "viral", "bacterial",
    "antibiotic", "vaccine", "drug", "therapy", "treatment", "clinical", "trial", "patient", "patients",
    "surgery", "diagnosis", "symptom", "mortality", "morbidity", "health", "medical", "medicine",
    "dose", "dosage", "placebo", "randomized", "randomised", "pathology", "chronic", "acute",
    "obesity", "cholesterol", "blood", "kidney", "liver", "lung", "brain", "cognitive", "dementia",
    "alzheimer", "depression", "anxiety", "psychiatric", "pregnancy", "nutrition", "diet", "fasting",
    "supplement", "vitamin", "inflammation", "immune", "gene", "genetic", "protein", "cell", "covid",
    "pharmacology", "epidemiology", "biomarker", "glucose", "metabolic", "sleep", "exercise",
}


def is_medical(query: str) -> bool:
    hits = set(re.findall(r"[a-z]+", query.lower())) & MEDICAL_TERMS
    log.info("Medical check for %r -> %s (matched: %s)", query, bool(hits), sorted(hits))
    return bool(hits)


S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,authors,venue,citationCount,externalIds,openAccessPdf,url"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": "research-evidence-finder/0.1"}


def http_get(name: str, url: str, params: dict, headers: dict | None = None, retries: int = 3,
             timeout: int = 30) -> requests.Response:
    """GET with retry/backoff for timeouts, connection errors, 429 and 5xx. Never logs headers (API keys)."""
    last = "unknown error"
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            r = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
        except requests.Timeout:
            last = f"timeout after {timeout}s"
        except requests.ConnectionError as e:
            last = f"connection error: {e.__class__.__name__}"
        except requests.RequestException as e:
            raise SearchError(f"{name}: request failed: {e}") from e
        else:
            log.debug("%s attempt %d: HTTP %s in %.2fs", name, attempt, r.status_code, time.time() - t0)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                wait = float(r.headers.get("Retry-After", 0) or 0) or 2 * attempt
                if attempt < retries:
                    log.warning("%s: %s, retrying in %.0fs (attempt %d/%d)", name, last, wait, attempt, retries)
                    time.sleep(wait)
                    continue
            else:  # 400/401/403/404...: not retryable
                raise SearchError(f"{name}: HTTP {r.status_code}: {r.text[:200]!r}")
        if attempt < retries:
            log.warning("%s: %s, retrying (attempt %d/%d)", name, last, attempt, retries)
            time.sleep(2 * attempt)
    raise SearchError(f"{name}: gave up after {retries} attempts ({last})")


def search_semantic_scholar(query: str, limit: int) -> list[dict]:
    key = s2_key()
    headers = dict(HEADERS)
    if key:
        headers["x-api-key"] = key.strip()
    else:
        log.warning("No Semantic Scholar API key found; expect rate limiting.")
    params = {"query": query, "limit": limit, "fields": S2_FIELDS}
    log.info("Semantic Scholar search: %r (limit=%d, key=%s)", query, limit, "yes" if key else "no")

    try:
        r = http_get("SemanticScholar", S2_URL, params, headers)
    except SearchError as e:
        if key and ("HTTP 401" in str(e) or "HTTP 403" in str(e)):
            log.error("Semantic Scholar rejected the API key (%s); retrying without key.", e)
            r = http_get("SemanticScholar", S2_URL, params, HEADERS)
        else:
            raise

    try:
        data = r.json().get("data", [])
    except ValueError as e:
        raise SearchError(f"SemanticScholar: invalid JSON response: {e}") from e

    papers, skipped = [], 0
    for p in data:
        try:
            ids = p.get("externalIds") or {}
            papers.append({
                "source": "semanticscholar",
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "year": p.get("year"),
                "authors": [a.get("name", "") for a in (p.get("authors") or [])][:6],
                "venue": p.get("venue"),
                "citations": p.get("citationCount"),
                "doi": ids.get("DOI"),
                "pmid": ids.get("PubMed"),
                "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
                "url": p.get("url"),
            })
        except Exception:
            skipped += 1
            log.exception("Skipping malformed Semantic Scholar record: %.200s", p)
    log.info("Semantic Scholar: %d papers (%d skipped)", len(papers), skipped)
    return papers


def search_pubmed(query: str, limit: int) -> list[dict]:
    log.info("PubMed search: %r (limit=%d)", query, limit)
    r = http_get("PubMed/esearch", f"{EUTILS}/esearch.fcgi",
                 {"db": "pubmed", "term": query, "retmax": limit, "retmode": "json", "sort": "relevance"})
    try:
        pmids = r.json()["esearchresult"]["idlist"]
    except (ValueError, KeyError) as e:
        raise SearchError(f"PubMed/esearch: unexpected response: {e!r}") from e
    log.info("PubMed esearch returned %d ids", len(pmids))
    if not pmids:
        return []

    r = http_get("PubMed/efetch", f"{EUTILS}/efetch.fcgi",
                 {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError as e:
        raise SearchError(f"PubMed/efetch: invalid XML: {e}") from e

    papers, skipped = [], 0
    for art in root.findall(".//PubmedArticle"):
        try:
            pmid = art.findtext(".//PMID")
            parts = []
            for ab in art.findall(".//Abstract/AbstractText"):
                text = "".join(ab.itertext()).strip()
                label = ab.get("Label")
                parts.append(f"{label}: {text}" if label else text)
            doi = None
            for aid in art.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text
            authors = []
            for a in art.findall(".//Author")[:6]:
                name = f"{a.findtext('ForeName') or ''} {a.findtext('LastName') or ''}".strip()
                if name:
                    authors.append(name)
            title_el = art.find(".//ArticleTitle")
            year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
            papers.append({
                "source": "pubmed",
                "title": "".join(title_el.itertext()).strip() if title_el is not None else "(no title)",
                "abstract": "\n".join(parts) or None,
                "year": int(year) if str(year).isdigit() else None,
                "authors": authors,
                "venue": art.findtext(".//Journal/Title"),
                "citations": None,
                "doi": doi,
                "pmid": pmid,
                "pdf_url": None,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        except Exception:
            skipped += 1
            log.exception("Skipping malformed PubMed record")
    log.info("PubMed: %d papers (%d skipped)", len(papers), skipped)
    return papers


def _key(p: dict) -> str:
    if p.get("doi"):
        return "doi:" + p["doi"].lower()
    if p.get("pmid"):
        return "pmid:" + str(p["pmid"])
    return "title:" + re.sub(r"\W+", "", (p.get("title") or "").lower())


def merge(*lists: list[dict]) -> list[dict]:
    """Dedupe by DOI/PMID/title, keeping the richest record per paper."""
    merged: dict[str, dict] = {}
    total = 0
    for plist in lists:
        for p in plist:
            total += 1
            k = _key(p)
            if k not in merged:
                merged[k] = dict(p, sources=[p["source"]])
                continue
            m = merged[k]
            m["sources"].append(p["source"])
            for field, val in p.items():
                if field != "source" and not m.get(field) and val:
                    m[field] = val
    log.info("Merged %d records into %d unique papers", total, len(merged))
    return list(merged.values())


def enrich_citations(papers: list[dict]) -> None:
    """Fill citation counts (and open-access PDF links) via Semantic Scholar's batch lookup by DOI/PMID.
    Lookup only - never raises, never changes which papers were found."""
    targets, ids = [], []
    for p in papers:
        if p.get("citations") is None:
            ident = f"DOI:{p['doi']}" if p.get("doi") else (f"PMID:{p['pmid']}" if p.get("pmid") else None)
            if ident:
                targets.append(p)
                ids.append(ident)
    if not ids:
        return
    headers = dict(HEADERS)
    if s2_key():
        headers["x-api-key"] = s2_key().strip()
    log.info("Citation enrichment: looking up %d papers in Semantic Scholar", len(ids))
    for attempt in range(1, 4):
        try:
            r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                              params={"fields": "citationCount,openAccessPdf"},
                              json={"ids": ids[:500]}, headers=headers, timeout=30)
        except requests.RequestException as e:
            log.warning("Citation enrichment attempt %d failed: %s", attempt, e.__class__.__name__)
            time.sleep(2 * attempt)
            continue
        if r.status_code == 200:
            found = 0
            for p, rec in zip(targets, r.json()):
                if rec:
                    p["citations"] = rec.get("citationCount")
                    if not p.get("pdf_url"):
                        p["pdf_url"] = (rec.get("openAccessPdf") or {}).get("url")
                    found += 1
            log.info("Citation enrichment: %d/%d found", found, len(ids))
            return
        log.warning("Citation enrichment attempt %d: HTTP %s", attempt, r.status_code)
        if r.status_code not in (429, 500, 502, 503, 504):
            return
        time.sleep(2 * attempt)
    log.warning("Citation enrichment gave up; citation counts left empty.")


def run_search(query: "str | dict[str, str]", limit: int, medical: bool, enrich: bool = True) -> tuple[list[dict], list[str]]:
    """Run the routed search. Returns (papers, errors); one source failing doesn't kill the other."""
    errors, results = [], []
    sources = [("PubMed", search_pubmed)] if medical else [("Semantic Scholar", search_semantic_scholar),
                                                            ("PubMed", search_pubmed)]
    log.info("Routing: %s", "medical -> PubMed only" if medical else "general -> both sources")
    for name, fn in sources:
        try:
            # `query` may be one string for every source, or {"pubmed": ..., "semanticscholar": ...}
            q = query.get(name.lower().replace(" ", ""), "") if isinstance(query, dict) else query
            results.append(fn(q, limit))
        except SearchError as e:
            log.error("%s failed: %s", name, e)
            errors.append(f"{name}: {e}")
        except Exception as e:
            log.exception("%s crashed unexpectedly", name)
            errors.append(f"{name}: unexpected error {e.__class__.__name__}: {e}")
    papers = merge(*results)
    if enrich and papers:
        enrich_citations(papers)
    return papers, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--limit", type=int, default=20, help="results per source")
    ap.add_argument("--out", default="papers.json")
    args = ap.parse_args()

    papers, errors = run_search(args.query, args.limit, is_medical(args.query))
    with_abs = [p for p in papers if p.get("abstract")]
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"query": args.query, "papers": papers, "errors": errors}, f, indent=2, ensure_ascii=False)
    except OSError:
        log.exception("Could not write %s", args.out)
        raise SystemExit(1)

    log.info("Done: %d unique papers, %d with abstract, %d source errors. Saved to %s",
             len(papers), len(with_abs), len(errors), args.out)
    for p in papers[:10]:
        print(f"- [{p.get('year')}] {p['title']}  ({'+'.join(p['sources'])})")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
