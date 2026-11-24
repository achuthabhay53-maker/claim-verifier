"""Evidence validation: does a paper support a claim, and how directly / how strongly?

Pipeline
  1. decompose_claim()   -> intervention, comparator, outcome, required outcome level
  2. extract_facts()     -> structured facts for one paper (design, population, intervention,
                            comparator, outcomes measured, limitations, verbatim quotes)
  3. apply_guardrails()  -> deterministic rules in code that downgrade over-generous labels
  4. verify_quotes()     -> every quote must really appear in the abstract
  5. summarize()         -> overall answer for the claim
"""
import concurrent.futures as cf
import re

import anthropic

from find_papers import _key as paper_key
from find_papers import log, run_search

LABELS = ["SUPPORTS", "PARTIALLY_SUPPORTS", "CONTRADICTS", "MIXED", "IRRELEVANT", "INSUFFICIENT_EVIDENCE"]
LABEL_ICON = {"SUPPORTS": "🟢", "PARTIALLY_SUPPORTS": "🟡", "CONTRADICTS": "🔴", "MIXED": "🟠",
              "IRRELEVANT": "⚪", "INSUFFICIENT_EVIDENCE": "🔘", "ERROR": "⚠️"}

# Outcome ladder: a claim about a higher level cannot be proven by evidence from a lower one.
OUTCOME_LEVELS = {
    1: "Knowledge, satisfaction, confidence, cognitive load",
    2: "Performance measured inside the simulator / training tool itself",
    3: "Skill assessed on a standard task or model (e.g. OSATS, GOALS, procedural checklist)",
    4: "Performance or errors in real operations / real clinical practice",
    5: "Patient outcomes (complications, survival, safety events)",
}
LEVEL_SHORT = {1: "knowledge/satisfaction", 2: "in-simulator metrics", 3: "skill assessment",
               4: "real-world performance", 5: "patient outcomes", 0: "none reported"}

DESIGNS = ["meta_analysis", "systematic_review", "rct", "non_randomized_trial", "cohort", "crossover",
           "narrative_review", "other"]

LEVELS_TEXT = "\n".join(f"  {k} = {v}" for k, v in OUTCOME_LEVELS.items())

CLAIM_TOOL = {
    "name": "decompose_claim",
    "description": "Break a research claim into the parts a paper must match.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intervention": {"type": "string", "description": "The specific intervention. Be precise about its type."},
            "comparator": {"type": "string", "description": "What it is compared against, or 'none stated'."},
            "outcome": {"type": "string", "description": "The outcome the claim says changes."},
            "required_outcome_level": {
                "type": "integer", "minimum": 1, "maximum": 5,
                "description": "Lowest outcome level that could genuinely prove the claim.\n" + LEVELS_TEXT},
            "population": {"type": "string", "description": "Who the claim is about, or 'unspecified'."},
        },
        "required": ["intervention", "comparator", "outcome", "required_outcome_level", "population"],
    },
}

PAPER_TOOL = {
    "name": "record_paper_facts",
    "description": "Record structured facts about one paper and how it relates to the claim.",
    "input_schema": {
        "type": "object",
        "properties": {
            "study_design": {"type": "string", "enum": DESIGNS},
            "population": {"type": "string"},
            "sample_size": {"type": ["integer", "null"], "description": "Participants (or trials for reviews if participants unknown)"},
            "intervention": {"type": "string", "description": "What was actually tested in this paper."},
            "intervention_match": {
                "type": "string", "enum": ["exact", "variant", "mixed", "different"],
                "description": ("exact = same type of intervention as the claim. "
                                "variant = related but not the same type (e.g. 360-degree video instead of interactive simulation). "
                                "mixed = the intervention is lumped with other modalities so its own effect cannot be isolated. "
                                "different = a different intervention (e.g. augmented reality for a VR claim).")},
            "comparator": {"type": "string"},
            "comparator_match": {
                "type": "string", "enum": ["matches", "partial", "not_matching", "none"],
                "description": ("matches = compared against what the claim names. partial = some comparison arms match, others do not. "
                                "not_matching = compared against something else. none = no comparison group.")},
            "outcomes": {
                "type": "array",
                "description": "Outcomes with RESULTS reported in the abstract. Do NOT list outcomes only mentioned as hopes, future work, or motivation.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "level": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Outcome ladder level:\n" + LEVELS_TEXT},
                        "result": {"type": "string", "enum": ["improved", "no_difference", "worse", "mixed"]},
                    },
                    "required": ["name", "level", "result"],
                },
            },
            "proposed_label": {"type": "string", "enum": LABELS},
            "directness": {"type": "string", "enum": ["direct", "indirect", "background"],
                           "description": "direct = tests the claim's intervention/comparator/outcome. indirect = covers it only as part of something broader or adjacent. background = only context."},
            "rationale": {"type": "string", "description": "1-3 plain sentences explaining the label."},
            "limitations": {"type": "array", "items": {"type": "string"},
                            "description": "Limitations stated or evident from the abstract (sample size, bias, heterogeneity, no real-world outcomes...)."},
            "quotes": {"type": "array", "items": {"type": "string"},
                       "description": "1-3 sentences copied EXACTLY from the abstract that are the key evidence."},
        },
        "required": ["study_design", "population", "intervention", "intervention_match", "comparator",
                     "comparator_match", "outcomes", "proposed_label", "directness", "rationale",
                     "limitations", "quotes"],
    },
}

SYSTEM = (
    "You are a careful evidence reviewer. You judge ONLY from the text given. Be conservative: "
    "an improvement in one outcome is not evidence for a different, higher-level outcome. "
    "Improved simulator or test scores do not prove fewer complications, fewer real-world errors, or better patient outcomes. "
    "A paper about a broader category (e.g. 'simulation' in general) is indirect evidence for a narrower claim (e.g. VR). "
    "Papers that mix several interventions without isolating the effect of the claim's intervention are indirect. "
    "Use INSUFFICIENT_EVIDENCE when the paper is on topic but does not measure what the claim needs. "
    "Use IRRELEVANT only when the paper is not about the claim's intervention at all."
)


def _client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, max_retries=4, timeout=90)


def _call_tool(client, model: str, tool: dict, user: str) -> dict:
    # This model rejects forced tool_choice, so ask for the tool in the prompt and retry once if it is skipped.
    prompt = user + f"\n\nRespond by calling the `{tool['name']}` tool. Do not write any other text."
    for attempt in (1, 2):
        msg = client.messages.create(
            model=model, max_tokens=2500, system=SYSTEM, tools=[tool],
            messages=[{"role": "user", "content": prompt}],
        )
        for block in msg.content:
            if block.type == "tool_use":
                return block.input
        log.warning("No tool call from model (attempt %d, stop_reason=%s)", attempt, msg.stop_reason)
    raise ValueError(f"model did not return structured output (stop_reason={msg.stop_reason})")


def decompose_claim(client, model: str, claim: str) -> dict:
    info = _call_tool(client, model, CLAIM_TOOL, f"Claim: {claim}")
    log.info("Claim decomposed: %s", info)
    return info


def extract_facts(client, model: str, claim: str, claim_info: dict, paper: dict) -> dict:
    user = (
        f"CLAIM: {claim}\n"
        f"Claim parts -> intervention: {claim_info['intervention']} | comparator: {claim_info['comparator']} | "
        f"outcome: {claim_info['outcome']} | population: {claim_info['population']}\n\n"
        f"PAPER TITLE: {paper['title']}\nABSTRACT:\n{paper['abstract']}"
    )
    return _call_tool(client, model, PAPER_TOOL, user)


# ---------------------------------------------------------------- quotes
def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip(" .\"'")


def verify_quotes(quotes: list[str], abstract: str) -> tuple[list[str], list[str]]:
    """Split quotes into (found verbatim in abstract, not found)."""
    hay = _norm(abstract)
    ok, bad = [], []
    for q in quotes or []:
        if len(_norm(q)) < 10:  # stray characters like "<" are model noise, not quotes
            continue
        frags = [f for f in re.split(r"\.\.\.|…|\[\.\.\.\]", q) if len(_norm(f)) >= 15] or [q]
        (ok if all(_norm(f) in hay for f in frags) else bad).append(q)
    return ok, bad


# ------------------------------------------------------------ guardrails
SUPPORTIVE = {"SUPPORTS", "PARTIALLY_SUPPORTS"}


def max_level(facts: dict) -> int:
    return max((o.get("level", 0) for o in facts.get("outcomes", [])), default=0)


def strength_of(facts: dict, label: str, directness: str) -> str:
    if label in ("IRRELEVANT", "INSUFFICIENT_EVIDENCE"):
        return "n/a"
    if directness != "direct":
        return "indirect"
    d, n = facts.get("study_design"), facts.get("sample_size") or 0
    if d in ("meta_analysis", "systematic_review"):
        s = "strong"
    elif d == "rct":
        s = "strong" if n >= 100 else "moderate"
    elif d in ("non_randomized_trial", "cohort", "crossover"):
        s = "moderate" if n >= 100 else "weak"
    else:  # narrative review, other
        s = "weak"
    if s == "strong" and label in ("PARTIALLY_SUPPORTS", "MIXED"):
        s = "moderate"
    return s


def apply_guardrails(claim_info: dict, facts: dict) -> dict:
    """Return final label/strength/directness plus the list of reasons any downgrade happened."""
    label = facts["proposed_label"]
    directness = facts.get("directness", "direct")
    notes: list[str] = []
    req = claim_info["required_outcome_level"]
    got = max_level(facts)
    im, cm = facts.get("intervention_match"), facts.get("comparator_match")

    def downgrade(new: str, why: str):
        nonlocal label
        if new != label:
            notes.append(f"{label} → {new}: {why}")
            label = new

    if im == "different" and label != "IRRELEVANT":
        downgrade("IRRELEVANT", "paper tests a different intervention than the claim")
    elif label != "IRRELEVANT":
        if got < req:
            downgrade("INSUFFICIENT_EVIDENCE",
                      f"claim needs level {req} ({LEVEL_SHORT[req]}) but the study only measured "
                      f"level {got} ({LEVEL_SHORT[got]})")
        elif label == "SUPPORTS":
            if im in ("variant", "mixed"):
                downgrade("PARTIALLY_SUPPORTS",
                          "intervention is a " + ("related variant" if im == "variant" else "mix of modalities")
                          + ", not exactly what the claim states")
            elif cm in ("partial", "not_matching", "none"):
                downgrade("PARTIALLY_SUPPORTS", f"comparator does not fully match the claim ({cm})")
    if im in ("variant", "mixed") and directness == "direct":
        directness = "indirect"
        notes.append("directness set to indirect: intervention only partly matches the claim")
    return {"label": label, "directness": directness, "strength": strength_of(facts, label, directness),
            "max_outcome_level": got, "guardrail_notes": notes}


def analyze_paper(client, model, claim, claim_info, paper) -> dict:
    facts = extract_facts(client, model, claim, claim_info, paper)
    final = apply_guardrails(claim_info, facts)
    ok, bad = verify_quotes(facts.get("quotes", []), paper["abstract"])
    if bad:
        log.warning("Unverified quote(s) in %r: %s", paper["title"][:60], bad)
    log.info("Analyzed %r: model=%s final=%s strength=%s level=%s", paper["title"][:60],
             facts["proposed_label"], final["label"], final["strength"], final["max_outcome_level"])
    return {**facts, **final, "model_label": facts["proposed_label"], "quotes_verified": ok, "quotes_unverified": bad}


def analyze_papers(api_key: str, model: str, claim: str, papers: list[dict], progress_cb=None, workers: int = 4,
                   claim_info: dict | None = None, indices=None):
    """Returns (claim_info, {paper_index: result}). A failure on one paper never stops the others.
    Pass claim_info + indices to analyze only some papers (used by the second search)."""
    client = _client(api_key)
    claim_info = claim_info or decompose_claim(client, model, claim)
    indices = list(range(len(papers)) if indices is None else indices)
    results: dict[int, dict] = {}

    def work(i):
        if not (papers[i].get("abstract") or "").strip():
            log.info("No abstract for %r; marking INSUFFICIENT_EVIDENCE without calling the model", papers[i]["title"])
            return i, {"label": "INSUFFICIENT_EVIDENCE", "strength": "n/a", "directness": "background",
                       "max_outcome_level": 0, "rationale": "No abstract available, so the paper cannot be judged.",
                       "guardrail_notes": [], "quotes_verified": [], "quotes_unverified": [], "limitations": [],
                       "outcomes": []}
        try:
            return i, analyze_paper(client, model, claim, claim_info, papers[i])
        except Exception as e:
            log.exception("Analysis failed for %r", papers[i]["title"])
            return i, {"label": "ERROR", "strength": "n/a", "directness": "n/a", "max_outcome_level": 0,
                       "rationale": f"{e.__class__.__name__}: {e}", "guardrail_notes": [], "quotes_verified": [],
                       "quotes_unverified": [], "limitations": [], "outcomes": []}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for done, fut in enumerate(cf.as_completed([ex.submit(work, i) for i in indices]), 1):
            i, r = fut.result()
            results[i] = r
            if progress_cb:
                progress_cb(done, len(indices))
    return claim_info, results


QUERY_TOOL = {
    "name": "make_outcome_terms",
    "description": "List search terms for finding studies that actually MEASURE the claim's outcome.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intervention_terms": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6,
                                   "description": "Synonyms for the intervention, e.g. 'virtual reality', 'VR simulator'."},
            "context_terms": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6,
                              "description": "Synonyms for the setting, e.g. 'surgical training', 'surgical education'."},
            "outcome_terms": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 8,
                              "description": "Synonyms for the outcome the claim needs, e.g. 'postoperative complications', 'mortality'."},
            "explanation": {"type": "string", "description": "One short sentence on what this search targets."},
        },
        "required": ["intervention_terms", "context_terms", "outcome_terms", "explanation"],
    },
}


def build_query(info: dict) -> str:
    """Build the boolean query in code so the parentheses are always right:
    (intervention OR ...) AND (context OR ...) AND (outcome OR ...), every term quoted and field-tagged."""
    def group(terms):
        clean = [t.strip().replace('"', "") for t in terms if t and t.strip()]
        return "(" + " OR ".join(f'"{t}"[tiab]' for t in clean) + ")"
    return " AND ".join(group(info[k]) for k in ("intervention_terms", "context_terms", "outcome_terms"))


def deeper_pass(api_key: str, model: str, claim: str, claim_info: dict, papers: list[dict], limit: int,
                medical: bool, enrich: bool, only_abstract: bool, progress_cb=None) -> dict:
    """Second search aimed at the outcome level the claim needs. Analyzes only papers not seen before.
    Returns {query, explanation, papers (new), results ({index in papers+new: result}), errors, found}."""
    client = _client(api_key)
    info = _call_tool(client, model, QUERY_TOOL,
                      f"Claim: {claim}\nIntervention: {claim_info['intervention']}\n"
                      f"Outcome needed: {claim_info['outcome']} (level {claim_info['required_outcome_level']}: "
                      f"{OUTCOME_LEVELS[claim_info['required_outcome_level']]})\nPopulation: {claim_info['population']}")
    query = build_query(info)
    log.info("Deeper search query: %s", query)
    found, errors = run_search(query, limit, medical, enrich)
    seen = {paper_key(p) for p in papers}
    new = [p for p in found if paper_key(p) not in seen and (p.get("abstract") or not only_abstract)]
    log.info("Deeper search: %d found, %d new after dedupe/abstract filter", len(found), len(new))
    for p in new:
        p["round"] = 2
    results: dict[int, dict] = {}
    if new:
        start = len(papers)
        _, results = analyze_papers(api_key, model, claim, papers + new, progress_cb, claim_info=claim_info,
                                    indices=range(start, start + len(new)))
    return {"query": query, "explanation": info.get("explanation", ""), "papers": new, "results": results,
            "errors": errors, "found": len(found)}


def summarize(claim_info: dict, results: dict, deeper_query: str | None = None) -> dict:
    counts = {l: 0 for l in LABELS + ["ERROR"]}
    for r in results.values():
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    req = claim_info["required_outcome_level"]
    # Papers judged IRRELEVANT (e.g. VR for patient rehab, not surgeon training) never count as evidence level.
    best = max((r["max_outcome_level"] for r in results.values() if r["label"] not in ("IRRELEVANT", "ERROR")),
               default=0)
    solid = sum(1 for r in results.values() if r["label"] == "SUPPORTS" and r["strength"] in ("strong", "moderate"))
    if best < req:
        head = (f"Insufficient evidence: the claim needs level {req} evidence ({LEVEL_SHORT[req]}), "
                f"but the best evidence found only reaches level {best} ({LEVEL_SHORT[best]}).")
        if deeper_query:
            head += " This stays true after a second search aimed at that outcome, though it only covers the top results of each search."
    elif solid:
        head = f"{solid} paper(s) give direct, moderate-to-strong support at the outcome level the claim needs."
    elif counts["SUPPORTS"] + counts["PARTIALLY_SUPPORTS"]:
        head = "Only partial, indirect or weak support was found at the outcome level the claim needs."
    else:
        head = "No paper supports the claim at the outcome level it needs."
    # Trigger for the second search: nothing on-topic backs (or contradicts) the claim at the level it needs.
    answered = any(r["label"] in ("SUPPORTS", "PARTIALLY_SUPPORTS", "CONTRADICTS", "MIXED")
                   and r["max_outcome_level"] >= req for r in results.values())
    return {"counts": counts, "best_level": best, "required_level": req, "headline": head,
            "needs_deeper": not answered}
