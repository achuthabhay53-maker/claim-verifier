"""Offline tests for claim/evidence logic: guardrails, strength, quotes, summary, query building,
error isolation and the second-search pass. The model is replaced by canned answers."""
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

import evidence as ev

CI3 = {"intervention": "VR training", "comparator": "traditional", "outcome": "technical skills",
       "required_outcome_level": 3, "population": "residents"}
CI5 = {**CI3, "outcome": "patient survival", "required_outcome_level": 5}


def facts(**over):
    base = dict(proposed_label="SUPPORTS", directness="direct", intervention_match="exact",
                comparator_match="matches", study_design="rct", sample_size=20,
                outcomes=[{"name": "OSATS", "level": 3, "result": "improved"}],
                population="residents", intervention="VR", comparator="textbook",
                rationale="r", limitations=[], quotes=[])
    base.update(over)
    return base


# ------------------------------------------------------------------ guardrails
def test_clean_support_is_kept():
    out = ev.apply_guardrails(CI3, facts())
    assert out["label"] == "SUPPORTS" and out["strength"] == "moderate" and out["guardrail_notes"] == []


def test_low_outcome_level_becomes_insufficient():
    out = ev.apply_guardrails(CI5, facts())  # only level 3 measured, claim needs 5
    assert out["label"] == "INSUFFICIENT_EVIDENCE" and out["strength"] == "n/a"
    assert "level 5" in out["guardrail_notes"][0]


def test_no_outcomes_at_all_is_insufficient():
    assert ev.apply_guardrails(CI3, facts(outcomes=[]))["label"] == "INSUFFICIENT_EVIDENCE"


def test_contradiction_at_too_low_a_level_is_insufficient():
    out = ev.apply_guardrails(CI5, facts(proposed_label="CONTRADICTS"))
    assert out["label"] == "INSUFFICIENT_EVIDENCE"


def test_different_intervention_is_irrelevant():
    out = ev.apply_guardrails(CI3, facts(intervention_match="different"))
    assert out["label"] == "IRRELEVANT"


@pytest.mark.parametrize("match", ["variant", "mixed"])
def test_variant_or_mixed_intervention_caps_at_partial_and_indirect(match):
    out = ev.apply_guardrails(CI3, facts(intervention_match=match))
    assert out["label"] == "PARTIALLY_SUPPORTS"
    assert out["directness"] == "indirect" and out["strength"] == "indirect"


@pytest.mark.parametrize("cmp_", ["partial", "not_matching", "none"])
def test_comparator_mismatch_caps_at_partial(cmp_):
    assert ev.apply_guardrails(CI3, facts(comparator_match=cmp_))["label"] == "PARTIALLY_SUPPORTS"


def test_irrelevant_stays_irrelevant_without_notes():
    out = ev.apply_guardrails(CI3, facts(proposed_label="IRRELEVANT", outcomes=[]))
    assert out["label"] == "IRRELEVANT" and out["guardrail_notes"] == []


def test_model_insufficient_label_is_respected():
    assert ev.apply_guardrails(CI3, facts(proposed_label="INSUFFICIENT_EVIDENCE"))["label"] == "INSUFFICIENT_EVIDENCE"


# -------------------------------------------------------------------- strength
@pytest.mark.parametrize("design,n,expected", [
    ("meta_analysis", None, "strong"), ("systematic_review", 0, "strong"),
    ("rct", 150, "strong"), ("rct", 20, "moderate"),
    ("cohort", 20, "weak"), ("crossover", 150, "moderate"), ("non_randomized_trial", 10, "weak"),
    ("narrative_review", None, "weak"), ("other", 500, "weak"),
])
def test_strength_by_design(design, n, expected):
    assert ev.strength_of({"study_design": design, "sample_size": n}, "SUPPORTS", "direct") == expected


def test_strength_edge_cases():
    f = {"study_design": "meta_analysis", "sample_size": None}
    assert ev.strength_of(f, "PARTIALLY_SUPPORTS", "direct") == "moderate"  # strong downgraded
    assert ev.strength_of(f, "SUPPORTS", "indirect") == "indirect"
    assert ev.strength_of(f, "IRRELEVANT", "direct") == "n/a"
    assert ev.strength_of(f, "INSUFFICIENT_EVIDENCE", "direct") == "n/a"


def test_max_level():
    assert ev.max_level({"outcomes": [{"level": 1}, {"level": 4}]}) == 4
    assert ev.max_level({}) == 0


# ---------------------------------------------------------------------- quotes
def test_verify_quotes_accepts_verbatim_despite_case_spacing_and_dashes():
    ok, bad = ev.verify_quotes(["vr  improved\nskills – a lot."], "Abstract: VR improved skills - a lot. More.")
    assert ok and not bad


def test_verify_quotes_treats_math_symbols_as_equal():
    ok, bad = ev.verify_quotes(["scores >= 270 and change <= 5 (-3)"], "Scores ≥ 270 and change ≤ 5 (−3).")
    assert ok and not bad


def test_verify_quotes_flags_invented_text():
    ok, bad = ev.verify_quotes(["VR cured cancer in every patient."], "VR improved skills.")
    assert bad == ["VR cured cancer in every patient."] and not ok


def test_verify_quotes_handles_ellipsis_fragments():
    abstract = "Residents trained with VR were faster and made fewer errors than the control group overall."
    ok, bad = ev.verify_quotes(["Residents trained with VR were faster ... fewer errors than the control group"], abstract)
    assert ok and not bad


def test_verify_quotes_ignores_junk_fragments():
    assert ev.verify_quotes(["<", ">", "/"], "anything") == ([], [])


# --------------------------------------------------------------------- summary
def res(label, level=3, strength="moderate"):
    return {"label": label, "max_outcome_level": level, "strength": strength}


def test_summary_insufficient_when_level_too_low():
    s = ev.summarize(CI5, {0: res("SUPPORTS", 3), 1: res("INSUFFICIENT_EVIDENCE", 1)})
    assert s["headline"].startswith("Insufficient evidence") and s["best_level"] == 3 and s["needs_deeper"]


def test_summary_mentions_second_search():
    s = ev.summarize(CI5, {0: res("SUPPORTS", 3)}, deeper_query="q")
    assert "second search" in s["headline"]


def test_summary_irrelevant_papers_do_not_raise_best_level():
    s = ev.summarize(CI5, {0: res("IRRELEVANT", 5), 1: res("SUPPORTS", 3)})
    assert s["best_level"] == 3


def test_summary_direct_support_headline_and_no_deeper_needed():
    s = ev.summarize(CI3, {0: res("SUPPORTS", 3, "strong")})
    assert "direct, moderate-to-strong" in s["headline"] and not s["needs_deeper"]


def test_summary_weak_support_headline():
    assert "partial, indirect or weak" in ev.summarize(CI3, {0: res("PARTIALLY_SUPPORTS", 3, "indirect")})["headline"]


def test_summary_no_support_at_needed_level():
    s = ev.summarize(CI3, {0: res("INSUFFICIENT_EVIDENCE", 3)})
    assert "No paper supports" in s["headline"] and s["needs_deeper"]


def test_needs_deeper_cases():
    assert ev.summarize(CI5, {0: res("INSUFFICIENT_EVIDENCE", 5)})["needs_deeper"]
    assert not ev.summarize(CI5, {0: res("PARTIALLY_SUPPORTS", 5)})["needs_deeper"]
    assert ev.summarize(CI5, {0: res("SUPPORTS", 3)})["needs_deeper"]


def test_summary_counts_labels():
    s = ev.summarize(CI3, {0: res("SUPPORTS"), 1: res("SUPPORTS"), 2: res("ERROR", 0)})
    assert s["counts"]["SUPPORTS"] == 2 and s["counts"]["ERROR"] == 1


# ----------------------------------------------------------------- query builder
def test_build_query_structure_and_cleaning():
    q = ev.build_query({"intervention_terms": ["virtual reality", 'VR "sim"'],
                        "context_terms": ["surgical training", "  "],
                        "outcome_terms": ["mortality", "survival", "complications"]})
    assert q == ('("virtual reality"[tiab] OR "VR sim"[tiab]) AND ("surgical training"[tiab]) '
                 'AND ("mortality"[tiab] OR "survival"[tiab] OR "complications"[tiab])')


# ------------------------------------------------------------------ _call_tool
class FakeModels:
    """Stands in for client.models: returns/raises the given items in order and records each call."""

    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


def _client_with(*items):
    models = FakeModels(*items)
    return SimpleNamespace(models=models), models


def _api_error(code):
    return genai_errors.ClientError(code, {"error": {"message": f"fake {code}", "status": "X"}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(ev.time, "sleep", lambda s: None)


def test_call_tool_returns_parsed_json_and_sends_schema():
    client, models = _client_with('{"ok": 1}')
    assert ev._call_tool(client, "gemini-x", ev.CLAIM_TOOL, "hello") == {"ok": 1}
    call = models.calls[0]
    assert call["model"] == "gemini-x" and call["contents"] == "hello"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == ev.CLAIM_TOOL["input_schema"]


def test_call_tool_retries_rate_limit_then_succeeds():
    client, models = _client_with(_api_error(429), _api_error(503), '{"ok": 1}')
    assert ev._call_tool(client, "m", ev.CLAIM_TOOL, "u") == {"ok": 1}
    assert len(models.calls) == 3


def test_call_tool_retries_bad_json_and_empty_text():
    client, models = _client_with("not json", None, "[1, 2]", '{"ok": 1}')
    assert ev._call_tool(client, "m", ev.CLAIM_TOOL, "u") == {"ok": 1}
    assert len(models.calls) == 4


def test_call_tool_gives_up_after_max_attempts():
    client, models = _client_with(*[_api_error(429)] * 3)
    with pytest.raises(ValueError, match="after 3 attempts"):
        ev._call_tool(client, "m", ev.CLAIM_TOOL, "u", max_attempts=3)
    assert len(models.calls) == 3


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_call_tool_does_not_retry_auth_billing_or_model_errors(code):
    client, models = _client_with(_api_error(code), '{"ok": 1}')
    with pytest.raises(genai_errors.APIError):
        ev._call_tool(client, "m", ev.CLAIM_TOOL, "u")
    assert len(models.calls) == 1


def test_schemas_are_plain_json_schema_objects():
    for tool in (ev.CLAIM_TOOL, ev.PAPER_TOOL, ev.QUERY_TOOL):
        schema = tool["input_schema"]
        assert schema["type"] == "object" and schema["required"]
        assert set(schema["required"]) <= set(schema["properties"])


# -------------------------------------------------------------- analyze_papers
@pytest.fixture
def fake_model(monkeypatch):
    """Replace the model calls; a paper titled 'bad' makes extraction raise."""
    calls = []
    monkeypatch.setattr(ev, "_client", lambda key: object())
    monkeypatch.setattr(ev, "decompose_claim", lambda c, m, claim: CI3)

    def fake_extract(client, model, claim, claim_info, paper):
        calls.append(paper["title"])
        if paper["title"] == "bad":
            raise RuntimeError("boom")
        return facts(quotes=["VR improved skills a lot."])

    monkeypatch.setattr(ev, "extract_facts", fake_extract)
    return calls


PAPERS = [{"title": "good", "abstract": "VR improved skills a lot."},
          {"title": "bad", "abstract": "x"},
          {"title": "noabs", "abstract": None}]


def test_one_failing_paper_does_not_stop_the_rest(fake_model):
    _, res_ = ev.analyze_papers("k", "m", "claim", PAPERS)
    assert res_[0]["label"] == "SUPPORTS" and res_[0]["quotes_verified"] == ["VR improved skills a lot."]
    assert res_[1]["label"] == "ERROR" and "boom" in res_[1]["rationale"]


def test_paper_without_abstract_is_insufficient_and_model_not_called(fake_model):
    _, res_ = ev.analyze_papers("k", "m", "claim", PAPERS)
    assert res_[2]["label"] == "INSUFFICIENT_EVIDENCE" and "noabs" not in fake_model


def test_analyze_subset_and_progress(fake_model):
    ticks = []
    _, res_ = ev.analyze_papers("k", "m", "claim", PAPERS, progress_cb=lambda d, n: ticks.append((d, n)),
                                claim_info=CI3, indices=[0])
    assert list(res_) == [0] and ticks == [(1, 1)]


# ----------------------------------------------------------------- deeper_pass
def test_deeper_pass_dedupes_filters_and_tags(monkeypatch, fake_model):
    existing = [{"title": "old", "abstract": "a", "doi": "10.1/old", "pmid": None}]
    found = [{"title": "old again", "abstract": "a", "doi": "10.1/OLD", "pmid": None},   # duplicate by DOI
             {"title": "good", "abstract": "VR improved skills a lot.", "doi": "10.1/new", "pmid": None},
             {"title": "noabs", "abstract": None, "doi": "10.1/na", "pmid": None}]
    monkeypatch.setattr(ev, "_call_tool", lambda *a, **k: {
        "intervention_terms": ["vr", "vr sim"], "context_terms": ["surgical training", "education"],
        "outcome_terms": ["mortality", "survival", "complications"], "explanation": "why"})
    monkeypatch.setattr(ev, "run_search", lambda q, n, medical, enrich: (found, []))

    out = ev.deeper_pass("k", "m", "claim", CI5, existing, 5, True, False, only_abstract=True)
    assert [p["title"] for p in out["papers"]] == ["good"]          # duplicate + no-abstract dropped
    assert out["papers"][0]["round"] == 2
    assert list(out["results"]) == [1]                              # indexed after the existing paper
    assert out["query"].startswith('("vr"[tiab]') and out["found"] == 3


def test_deeper_pass_keeps_abstractless_papers_when_allowed(monkeypatch, fake_model):
    found = [{"title": "noabs", "abstract": None, "doi": "10.1/na", "pmid": None}]
    monkeypatch.setattr(ev, "_call_tool", lambda *a, **k: {
        "intervention_terms": ["a", "b"], "context_terms": ["c", "d"], "outcome_terms": ["e", "f", "g"], "explanation": ""})
    monkeypatch.setattr(ev, "run_search", lambda q, n, medical, enrich: (found, []))
    out = ev.deeper_pass("k", "m", "claim", CI5, [], 5, True, False, only_abstract=False)
    assert out["results"][0]["label"] == "INSUFFICIENT_EVIDENCE"


# ======================================================================================
# Added with the search planner / population check / CONTRADICTS caps / thin-result flag
# ======================================================================================

# ------------------------------------------------------ CONTRADICTS needs care too
def contra(**over):
    return facts(proposed_label="CONTRADICTS",
                 outcomes=[{"name": "OSATS", "level": 3, "result": "no_difference"}], sample_size=80, **over)


def test_clean_contradiction_from_a_decent_study_is_kept():
    out = ev.apply_guardrails(CI3, contra())
    assert out["label"] == "CONTRADICTS" and out["guardrail_notes"] == []


@pytest.mark.parametrize("over", [
    {"intervention_match": "mixed"}, {"intervention_match": "variant"},
    {"population_match": "partial"}, {"population_match": "different"},
    {"comparator_match": "partial"}, {"comparator_match": "not_matching"}, {"comparator_match": "none"},
])
def test_contradiction_from_a_mismatched_study_is_capped_at_mixed(over):
    out = ev.apply_guardrails(CI3, contra(**over))
    assert out["label"] == "MIXED" and out["guardrail_notes"]


def test_contradiction_from_a_tiny_study_is_capped_at_mixed():
    out = ev.apply_guardrails(CI3, facts(proposed_label="CONTRADICTS", sample_size=26,
                                         outcomes=[{"name": "x", "level": 3, "result": "no_difference"}]))
    assert out["label"] == "MIXED" and "small study" in out["guardrail_notes"][0]


def test_small_sample_cap_does_not_apply_to_reviews_or_unknown_n():
    for over in ({"study_design": "meta_analysis", "sample_size": 10}, {"sample_size": None}):
        out = ev.apply_guardrails(CI3, contra(**{k: v for k, v in over.items() if k != "sample_size"}) | over)
        assert out["label"] == "CONTRADICTS"


# ------------------------------------------------------------ population check
@pytest.mark.parametrize("pm", ["partial", "different"])
def test_population_mismatch_caps_support_and_makes_it_indirect(pm):
    out = ev.apply_guardrails(CI3, facts(population_match=pm))
    assert out["label"] == "PARTIALLY_SUPPORTS" and out["directness"] == "indirect" and out["strength"] == "indirect"


def test_missing_population_match_is_treated_as_matching():
    assert ev.apply_guardrails(CI3, facts())["label"] == "SUPPORTS"  # older results have no such field


def test_paper_schema_requires_population_match():
    schema = ev.PAPER_TOOL["input_schema"]
    assert "population_match" in schema["required"]
    assert schema["properties"]["population_match"]["enum"] == ["matches", "partial", "different"]


# ------------------------------------------------------------ thin-result flag
def test_thin_result_flag():
    few = {i: res("SUPPORTS") for i in range(ev.THIN_RESULT_THRESHOLD - 1)}
    enough = {i: res("SUPPORTS") for i in range(ev.THIN_RESULT_THRESHOLD)}
    assert ev.summarize(CI3, few)["thin"] and ev.summarize(CI3, few)["n_checked"] == ev.THIN_RESULT_THRESHOLD - 1
    assert not ev.summarize(CI3, enough)["thin"]


def test_errors_do_not_count_as_checked_papers():
    results = {i: res("SUPPORTS") for i in range(ev.THIN_RESULT_THRESHOLD)}
    results[0] = res("ERROR", 0)
    assert ev.summarize(CI3, results)["thin"]


# --------------------------------------------------------------- query builders
TERMS = {"intervention_terms": ["virtual reality", "VR simulator"], "context_terms": ["surgical training"],
         "outcome_terms": ["technical skills", "OSATS"]}


def test_build_query_skips_empty_groups():
    q = ev.build_query({**TERMS, "outcome_terms": []})
    assert q == '("virtual reality"[tiab] OR "VR simulator"[tiab]) AND ("surgical training"[tiab])'
    assert ev.build_query({"intervention_terms": [], "context_terms": [" "], "outcome_terms": []}) == ""


def test_search_queries_give_each_source_its_own_syntax():
    q = ev.search_queries(TERMS)
    assert "[tiab]" in q["pubmed"] and "AND" in q["pubmed"]
    assert q["semanticscholar"] == "virtual reality surgical training technical skills"   # plain text, no tags
    relaxed = ev.search_queries(TERMS, relaxed=True)
    assert "technical skills" not in relaxed["pubmed"] and relaxed["semanticscholar"] == "virtual reality surgical training"


# ---------------------------------------------------------------- plan_and_search
def _papers(n, prefix="p"):
    return [{"title": f"{prefix}{i}", "doi": f"10.1/{prefix}{i}", "pmid": None, "abstract": "a"} for i in range(n)]


@pytest.fixture
def planner(monkeypatch):
    """Fake the model's search terms and the search backend; records every query used."""
    monkeypatch.setattr(ev, "_client", lambda key: object())
    monkeypatch.setattr(ev, "_call_tool", lambda *a, **k: {**TERMS, "explanation": "why"})
    monkeypatch.setattr(ev, "enrich_citations", lambda papers: None)
    seen, answers = [], {}

    def fake_run_search(query, limit, medical, enrich=True):
        seen.append(query)
        return answers.get(len(seen), []), []

    monkeypatch.setattr(ev, "run_search", fake_run_search)
    return seen, answers


def test_plan_uses_structured_query_not_the_raw_sentence(planner):
    seen, answers = planner
    answers[1] = _papers(12)
    papers, errors, plan = ev.plan_and_search("k", "m", "A long claim sentence that improves things.", 15, True, False)
    assert len(papers) == 12 and not plan["relaxed_used"] and plan["relaxed"] is None
    assert len(seen) == 1 and isinstance(seen[0], dict) and "claim sentence" not in seen[0]["pubmed"]


def test_plan_relaxes_when_precise_query_finds_too_few(planner):
    seen, answers = planner
    answers[1], answers[2] = _papers(2), _papers(14, "r")
    papers, _, plan = ev.plan_and_search("k", "m", "claim", 15, True, False)
    assert len(papers) == 14 and plan["relaxed_used"] and plan["precise_count"] == 2 and plan["relaxed_count"] == 14
    assert len(seen) == 2 and "technical skills" not in seen[1]["pubmed"]
    assert plan["queries"] == seen[1]


def test_plan_keeps_precise_results_if_relaxing_does_not_help(planner):
    seen, answers = planner
    answers[1], answers[2] = _papers(3), _papers(1, "r")
    papers, _, plan = ev.plan_and_search("k", "m", "claim", 15, True, False)
    assert len(papers) == 3 and not plan["relaxed_used"] and plan["relaxed_count"] == 1


def test_plan_enriches_only_the_final_list(monkeypatch, planner):
    seen, answers = planner
    answers[1], answers[2] = _papers(1), _papers(9, "r")
    enriched = []
    monkeypatch.setattr(ev, "enrich_citations", lambda papers: enriched.append(len(papers)))
    ev.plan_and_search("k", "m", "claim", 15, True, enrich=True)
    assert enriched == [9]


def test_plan_rejects_unusable_terms(monkeypatch):
    monkeypatch.setattr(ev, "_client", lambda key: object())
    monkeypatch.setattr(ev, "_call_tool", lambda *a, **k: {"intervention_terms": [], "context_terms": [],
                                                           "outcome_terms": [], "explanation": ""})
    with pytest.raises(ValueError, match="no usable search terms"):
        ev.plan_and_search("k", "m", "claim", 15, True, False)
