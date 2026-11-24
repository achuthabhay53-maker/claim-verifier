"""Offline tests for claim/evidence logic: guardrails, strength, quotes, summary, query building,
error isolation and the second-search pass. The model is replaced by canned answers."""
from types import SimpleNamespace

import pytest

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
def _client_returning(*messages):
    it = iter(messages)
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: next(it)))


def _msg(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason="end_turn")


TEXT = SimpleNamespace(type="text", text="hello")
TOOL = SimpleNamespace(type="tool_use", input={"ok": 1})


def test_call_tool_returns_tool_input():
    assert ev._call_tool(_client_returning(_msg(TOOL)), "m", ev.CLAIM_TOOL, "u") == {"ok": 1}


def test_call_tool_retries_once_when_tool_skipped():
    assert ev._call_tool(_client_returning(_msg(TEXT), _msg(TOOL)), "m", ev.CLAIM_TOOL, "u") == {"ok": 1}


def test_call_tool_raises_when_model_never_calls_tool():
    with pytest.raises(ValueError, match="structured output"):
        ev._call_tool(_client_returning(_msg(TEXT), _msg(TEXT)), "m", ev.CLAIM_TOOL, "u")


# -------------------------------------------------------------- analyze_papers
@pytest.fixture
def fake_model(monkeypatch):
    """Replace the Claude calls; a paper titled 'bad' makes extraction raise."""
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
