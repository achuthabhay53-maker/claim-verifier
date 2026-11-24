"""Streamlit UI: claim -> find papers -> validate evidence (direction, directness, strength, outcome level).

Run:  .venv\\Scripts\\python.exe -m streamlit run app.py   (or double-click run.bat)
"""
import os

import pandas as pd
import streamlit as st

from evidence import LABEL_ICON, LABELS, LEVEL_SHORT, OUTCOME_LEVELS, analyze_papers, deeper_pass, summarize
from find_papers import LOG_FILE, is_medical, log, run_search, s2_key

st.set_page_config(page_title="Claim Evidence Finder", page_icon="🔬", layout="wide")
st.title("🔬 Claim Evidence Finder")
st.caption("Finds papers (PubMed / Semantic Scholar), then checks whether each one really supports your claim, "
           "how directly, and how strongly.")

with st.sidebar:
    st.header("Settings")
    limit = st.slider("Results per source", 5, 50, 15)
    topic = st.radio(
        "Topic type",
        ["Auto-detect", "Medical (PubMed only)", "General (PubMed + Semantic Scholar)"],
        help="Medical queries search PubMed only. Everything else searches both sources.",
    )
    st.caption("Semantic Scholar key: " + ("✅ loaded" if s2_key() else "❌ not found in .env"))
    enrich = st.checkbox("Add citation counts (Semantic Scholar lookup)", True,
                         help="Lookup by DOI/PMID only. It does not add papers to the search.")
    only_abstract = st.checkbox("Only papers with abstracts", True)
    auto_deeper = st.checkbox("Auto second search if evidence level is too low", True,
                              help="If no paper reaches the outcome level the claim needs, run one more search "
                                   "aimed at that outcome (e.g. complications, survival) and check the new papers.")
    st.divider()
    api_key = st.text_input("Anthropic API key", value=os.environ.get("ANTHROPIC_API_KEY", ""), type="password")
    model = st.text_input("Model", "claude-sonnet-5-5")
    with st.expander("Outcome levels used"):
        for k, v in OUTCOME_LEVELS.items():
            st.markdown(f"**{k}** – {v}")

claim = st.text_area("Your claim", height=80,
                     placeholder="e.g. Virtual reality simulation improves surgical technical skills compared with traditional training.")
query = st.text_input("Search query (optional — defaults to the claim)")

if st.button("1. Search papers", type="primary", disabled=not claim.strip()):
    q = query.strip() or claim.strip()
    medical = is_medical(claim + " " + q) if topic.startswith("Auto") else topic.startswith("Medical")
    log.info("UI search: claim=%r query=%r topic=%s medical=%s limit=%d", claim, q, topic, medical, limit)
    st.info("Medical topic → searching PubMed only." if medical else "General topic → searching PubMed + Semantic Scholar.")
    with st.spinner("Searching..."):
        papers, errors = run_search(q, limit, medical, enrich=enrich)
    for err in errors:
        st.error(f"⚠️ {err}  (details in the log below)")
    if not papers and not errors:
        st.warning("No papers found. Try a broader query.")
    if only_abstract:
        before = len(papers)
        papers = [p for p in papers if p.get("abstract")]
        log.info("Abstract filter: %d -> %d", before, len(papers))
    st.session_state.update(papers=papers, claim=claim.strip(), analysis=None, medical=medical)

papers = st.session_state.get("papers")
if papers is not None:
    st.subheader(f"{len(papers)} papers found")
    if not api_key:
        st.info("Add an Anthropic API key in the sidebar to enable evidence validation.")

    if st.button("2. Validate evidence for the claim", type="primary", disabled=not (api_key and papers)):
        bar = st.progress(0.0, text="Analyzing claim and papers...")
        try:
            claim_info, results = analyze_papers(
                api_key, model, st.session_state["claim"], papers,
                progress_cb=lambda d, n: bar.progress(d / n, text=f"Analyzed {d}/{n} papers"))
            summ = summarize(claim_info, results)
            deeper = None
            if auto_deeper and summ["needs_deeper"]:
                bar.progress(0.0, text="Evidence level too low - running a second, outcome-targeted search...")
                deeper = deeper_pass(
                    api_key, model, st.session_state["claim"], claim_info, papers, limit,
                    st.session_state.get("medical", True), enrich, only_abstract,
                    progress_cb=lambda d, n: bar.progress(d / n, text=f"Second search: analyzed {d}/{n} new papers"))
                results.update(deeper["results"])
                papers = papers + deeper["papers"]
                st.session_state["papers"] = papers
                summ = summarize(claim_info, results, deeper_query=deeper["query"])
            st.session_state["analysis"] = {"claim_info": claim_info, "results": results,
                                            "summary": summ, "deeper": deeper}
        except Exception as e:
            log.exception("Evidence validation failed")
            st.error(f"Evidence validation failed: {e.__class__.__name__}: {e}  (see log below)")
        bar.empty()

analysis = st.session_state.get("analysis")
if analysis and papers:
    ci, results, summ = analysis["claim_info"], analysis["results"], analysis["summary"]

    st.header("Result")
    (st.warning if summ["best_level"] < summ["required_level"] else st.success)(summ["headline"])

    dp = analysis.get("deeper")
    if dp:
        st.info(f"🔎 **Second search** (nothing from the first pass answered the claim at level {summ['required_level']}).  \n"
                f"Query: `{dp['query']}`  \n{dp['explanation']}  \n"
                f"Found {dp['found']} papers, **{len(dp['papers'])} new** after removing ones already checked.")
        for err in dp["errors"]:
            st.error(f"⚠️ Second search: {err}")

    with st.expander("How the claim was broken down", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Intervention**\n\n{ci['intervention']}")
        c2.markdown(f"**Comparator**\n\n{ci['comparator']}")
        c3.markdown(f"**Outcome**\n\n{ci['outcome']}")
        c4.markdown(f"**Population**\n\n{ci['population']}")
        st.markdown(f"**Evidence needed:** level {ci['required_outcome_level']} – "
                    f"{OUTCOME_LEVELS[ci['required_outcome_level']]}")

    cols = st.columns(len(LABELS))
    for col, label in zip(cols, LABELS):
        col.metric(f"{LABEL_ICON[label]} {label.replace('_', ' ').title()}", summ["counts"].get(label, 0))
    if summ["counts"].get("ERROR"):
        st.error(f"{summ['counts']['ERROR']} paper(s) failed to analyze — see the log.")

    rows = []
    for i, p in enumerate(papers):
        r = results.get(i)
        if not r:
            continue
        rows.append({
            "Label": f"{LABEL_ICON[r['label']]} {r['label']}", "Strength": r["strength"], "Directness": r["directness"],
            "Design": r.get("study_design", ""), "N": r.get("sample_size"),
            "Intervention match": r.get("intervention_match", ""), "Comparator match": r.get("comparator_match", ""),
            "Best outcome level": f"{r['max_outcome_level']} ({LEVEL_SHORT[r['max_outcome_level']]})",
            "Search": p.get("round", 1),
            "Year": p.get("year"), "Citations": p.get("citations"), "Title": p["title"],
            "Downgraded": "; ".join(r["guardrail_notes"]),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("⬇️ Download table (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "evidence.csv", "text/csv")

    st.subheader("Papers")
    pick = st.multiselect("Filter by label", LABELS + ["ERROR"], default=LABELS + ["ERROR"])
    order = {l: n for n, l in enumerate(["SUPPORTS", "PARTIALLY_SUPPORTS", "MIXED", "CONTRADICTS",
                                         "INSUFFICIENT_EVIDENCE", "IRRELEVANT", "ERROR"])}
    for i in sorted(results, key=lambda k: order.get(results[k]["label"], 9)):
        r, p = results[i], papers[i]
        if r["label"] not in pick:
            continue
        tag = " · 🔎 2nd search" if p.get("round") == 2 else ""
        head = f"{LABEL_ICON[r['label']]} {r['label']} · {r['strength']} · {p['title']} ({p.get('year')}){tag}"
        with st.expander(head):
            st.write(f"**{', '.join(p['authors'])}** — {p.get('venue') or ''}  \n"
                     f"Sources: {', '.join(p['sources'])} · Citations: "
                     f"{p['citations'] if p.get('citations') is not None else 'n/a'}")
            links = [f"[Page]({p['url']})"] if p.get("url") else []
            if p.get("pdf_url"):
                links.append(f"[Open-access PDF]({p['pdf_url']})")
            if p.get("doi"):
                links.append(f"[DOI](https://doi.org/{p['doi']})")
            st.markdown(" · ".join(links))

            st.markdown(f"**Why:** {r.get('rationale', '')}")
            for n in r["guardrail_notes"]:
                st.warning(f"Rule applied: {n}")
            if r["label"] != "ERROR":
                a, b = st.columns(2)
                a.markdown(f"**Design:** {r.get('study_design')} · **N:** {r.get('sample_size') or 'n/r'}  \n"
                           f"**Population:** {r.get('population')}  \n"
                           f"**Intervention:** {r.get('intervention')} _(match: {r.get('intervention_match')})_  \n"
                           f"**Comparator:** {r.get('comparator')} _(match: {r.get('comparator_match')})_")
                outs = "\n".join(f"- L{o['level']} {o['name']} → {o['result']}" for o in r.get("outcomes", [])) or "- none reported"
                b.markdown(f"**Outcomes with results:**\n{outs}  \n**Directness:** {r['directness']}")
                if r.get("limitations"):
                    st.markdown("**Limitations:** " + "; ".join(r["limitations"]))
                for q in r["quotes_verified"]:
                    st.markdown(f"> ✅ {q}")
                for q in r["quotes_unverified"]:
                    st.markdown(f"> ⚠️ _not found verbatim in abstract:_ {q}")
            st.markdown("**Abstract**")
            st.write(p["abstract"])

st.divider()
with st.expander("🪵 Activity log (latest 80 lines)"):
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            st.code("".join(f.readlines()[-80:]) or "(empty)", language="text")
    except FileNotFoundError:
        st.caption("No log yet.")
    st.caption(f"Full log: {LOG_FILE}  — press R to refresh")
