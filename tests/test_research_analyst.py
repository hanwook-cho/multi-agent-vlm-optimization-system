"""Mode B MVP — Research Analyst verifier + extraction contract (CI-safe).

No network, no LLM: the verifier is pure and the extraction takes an injected
chat_fn. Locks the HLD §6.4 anti-hallucination guard — a citation must be a real
registry paper and every verbatim excerpt must be an exact substring of the abstract.
"""

from __future__ import annotations

import json

from agents import research_analyst as ra

ABSTRACT = (
    "We propose Dynamic Token Merging, which merges redundant vision tokens by "
    "attention similarity. On a 0.5B VLM it cuts vision tokens by 50% with under "
    "1 point accuracy loss across POPE and MMBench."
)


def _valid_record(excerpt: str = "merges redundant vision tokens by attention similarity") -> dict:
    return {
        "title": "Dynamic Token Merging",
        "source_citation": {"title": "Dynamic Token Merging", "authors": ["A. B"],
                            "year": 2025, "arxiv_id": "2501.01234",
                            "url": "https://arxiv.org/abs/2501.01234",
                            "venue": None, "github_url": None},
        "claimed_effect": "Merges redundant vision tokens to cut compute.",
        "verbatim_excerpts": [{"text": excerpt, "location": None}],
        "reported_results": "50% fewer vision tokens, <1pt drop on POPE/MMBench.",
        "applicability_check": {"requirements": ["attention maps"], "verdict": "applicable",
                                "notes": "We have a SigLIP encoder with accessible attention."},
        "known_failure_modes": ["hurts dense OCR"],
        "implementation_difficulty": "new_module",
        "confidence_flags": {"claimed_effect": "high"},
    }


# ── verifier ──────────────────────────────────────────────────────────────────

def test_excerpt_substring_match_normalizes_ws_and_case():
    assert ra.excerpt_in_abstract("Merges  REDUNDANT vision tokens", ABSTRACT)
    assert not ra.excerpt_in_abstract("merges audio tokens", ABSTRACT)
    assert not ra.excerpt_in_abstract("", ABSTRACT)


def test_citation_in_registry():
    rec = _valid_record()
    assert ra.citation_in_registry(rec, {"2501.01234", "2412.00001"})
    assert not ra.citation_in_registry(rec, {"2412.00001"})        # not retrieved
    rec["source_citation"]["arxiv_id"] = None
    assert not ra.citation_in_registry(rec, {"2501.01234"})


def test_verify_record_accepts_grounded_record():
    v = ra.verify_record(_valid_record(), ABSTRACT, {"2501.01234"})
    assert v["ok"] and v["citation_ok"] and v["excerpts_ok"] and not v["schema_errors"]


def test_verify_record_rejects_hallucinated_excerpt():
    rec = _valid_record(excerpt="merges tokens via a learned gating network")  # not in abstract
    v = ra.verify_record(rec, ABSTRACT, {"2501.01234"})
    assert not v["ok"] and not v["excerpts_ok"] and v["citation_ok"]


def test_verify_record_rejects_unregistered_citation():
    v = ra.verify_record(_valid_record(), ABSTRACT, {"9999.99999"})
    assert not v["ok"] and not v["citation_ok"]


def test_verify_record_rejects_schema_violation():
    rec = _valid_record()
    rec["implementation_difficulty"] = "trivial"     # not in enum
    v = ra.verify_record(rec, ABSTRACT, {"2501.01234"})
    assert not v["ok"] and v["schema_errors"]


# ── extraction (injected chat_fn; no LLM) ─────────────────────────────────────

def test_extract_stamps_citation_from_paper_and_strips_extras():
    paper = {"id": "2501.01234", "title": "Dynamic Token Merging",
             "authors": ["A. B"], "year": 2025, "abstract": ABSTRACT,
             "url": "https://arxiv.org/abs/2501.01234"}
    # LLM output omits source_citation (per the prompt) and adds a stray key.
    llm_json = {**_valid_record(), "hallucinated_extra": "ignore me"}
    del llm_json["source_citation"]
    fake_chat = lambda system, user: "thinking...\n" + json.dumps(llm_json) + "\n done"

    rec = ra.extract_record(paper, "problem", fake_chat)
    assert rec is not None
    # citation is set authoritatively from the real paper, not the LLM
    assert rec["source_citation"]["arxiv_id"] == "2501.01234"
    assert "hallucinated_extra" not in rec                       # stripped (schema is closed)
    # and it verifies end-to-end
    assert ra.verify_record(rec, ABSTRACT, {"2501.01234"})["ok"]


def test_extract_returns_none_on_unparseable():
    assert ra.extract_record({"id": "x", "abstract": ""}, "p", lambda s, u: "no json here") is None


def test_clean_arxiv_id_strips_version():
    assert ra.clean_arxiv_id("2504.01690v2") == "2504.01690"
    assert ra.clean_arxiv_id("arXiv:2504.01690") == "2504.01690"
    assert ra.clean_arxiv_id("2504.01690") == "2504.01690"
    assert ra.clean_arxiv_id(None) is None
    assert ra.clean_arxiv_id("not-an-id") is None


def test_extract_cleans_versioned_id_for_schema():
    # a versioned arXiv id would fail the schema's arxiv_id pattern if not cleaned
    paper = {"id": "2501.01234v3", "title": "T", "authors": ["A"], "year": 2025,
             "abstract": ABSTRACT, "url": "https://arxiv.org/abs/2501.01234"}
    rec = _valid_record()
    del rec["source_citation"]
    rec = ra._stamp_citation(rec, paper)
    assert rec["source_citation"]["arxiv_id"] == "2501.01234"     # version stripped
    assert not ra.schema_errors(rec)                              # validates
