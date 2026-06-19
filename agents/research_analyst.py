"""agents/research_analyst.py — Mode B MVP (HLD §6.2; ADR-0009, ADR-0015).

The Research Analyst is Mode A's counterpart: instead of proposing configs over known
techniques, it reads the *literature* and surfaces new ones. MVP flow (one-shot,
human-gated, compute-light):

  1. retrieve — search arXiv (tools/fetch_papers) for an open problem; add to registry
  2. extract  — LLM turns each paper's ABSTRACT into a HypothesisRecord
                (schemas/hypothesis_record.schema.json)
  3. VERIFY   — deterministic, never trusting the LLM (HLD §6.4 anti-hallucination):
                · source_citation is set by US from the real paper (not the LLM)
                · its arxiv_id must be a real registry paper
                · every verbatim excerpt must be an exact substring of the abstract
                · the record must schema-validate
  4. route    — survivors → a hypothesis queue + the Mode A→B escalation gate
                (services.gates.gate_mode_b_escalation), which surfaces in the console

The verifier functions are pure (no network/LLM) so they are unit-testable; the LLM
call is injectable (`chat_fn`) so tests never need a backend.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "hypothesis_record.schema.json"
HYPOTHESIS_QUEUE = PROJECT_ROOT / "artifacts" / "hypothesis_queue.json"

# Top-level keys the schema allows (additionalProperties: false) — strip the rest.
_ALLOWED_KEYS = {
    "title", "source_citation", "claimed_effect", "verbatim_excerpts",
    "original_hyperparameters", "reported_results", "applicability_check",
    "known_failure_modes", "implementation_difficulty",
    "proposed_codebase_insertion_point", "confidence_flags",
}


# ── Verification (pure; the trust boundary) ──────────────────────────────────

def _norm(s: str) -> str:
    """Punctuation-insensitive normalizer for excerpt matching: lowercase, fold smart
    quotes/dashes and LaTeX artifacts, drop punctuation, collapse whitespace. The
    word SEQUENCE must still match (a strong anti-hallucination guard), but quote-style
    differences (parens, ', em-dashes, arXiv \\textemdash) don't cause false rejects."""
    s = (s or "").lower().replace("\\textemdash", " ").replace("\\textendash", " ")
    s = re.sub(r"[‘’“”]", "'", s)      # smart quotes → '
    s = re.sub(r"[^a-z0-9' ]+", " ", s)                    # drop other punctuation
    return re.sub(r"\s+", " ", s).strip()


_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,7})")


def clean_arxiv_id(s: str | None) -> str | None:
    """Strip the version suffix (e.g. '2504.01690v2' → '2504.01690') so the id matches
    the schema's arxiv_id pattern and the registry consistently."""
    if not s:
        return None
    m = _ARXIV_RE.search(s)
    return m.group(1) if m else None


_MIN_RUN = 8   # words — a quote is "grounded" if it shares a verbatim run this long


def excerpt_in_abstract(excerpt_text: str, abstract: str) -> bool:
    """Grounded iff the excerpt appears (normalized) in the abstract, OR contains a run
    of ≥_MIN_RUN consecutive words that does. The long-run rule tolerates the LLM adding
    a stray word (e.g. a trailing 'etc.') while still making wholesale fabrication fail —
    an invented quote won't contain 8 consecutive real words from the paper. Short quotes
    (< _MIN_RUN words) must match in full."""
    if not excerpt_text.strip():
        return False
    na, ne = _norm(abstract), _norm(excerpt_text)
    if ne and ne in na:
        return True
    words = ne.split()
    if len(words) < _MIN_RUN:
        return False
    return any(" ".join(words[i:i + _MIN_RUN]) in na
               for i in range(len(words) - _MIN_RUN + 1))


def excerpts_verdict(record: dict, abstract: str) -> list[tuple[str, bool]]:
    return [(e.get("text", ""), excerpt_in_abstract(e.get("text", ""), abstract))
            for e in record.get("verbatim_excerpts", [])]


def all_excerpts_real(record: dict, abstract: str) -> bool:
    v = excerpts_verdict(record, abstract)
    return bool(v) and all(ok for _, ok in v)


def citation_in_registry(record: dict, registry_ids: set[str]) -> bool:
    """The cited arXiv id must be a real paper we actually retrieved (no fabricated refs)."""
    aid = (record.get("source_citation") or {}).get("arxiv_id")
    return bool(aid) and aid in registry_ids


def schema_errors(record: dict) -> list[str]:
    from jsonschema import Draft202012Validator
    schema = json.loads(SCHEMA_PATH.read_text())
    return [e.message for e in Draft202012Validator(schema).iter_errors(record)]


def verify_record(record: dict, abstract: str, registry_ids: set[str]) -> dict:
    """Combine the deterministic checks. ok ⇒ safe to route to the human gate."""
    errs = schema_errors(record)
    cit = citation_in_registry(record, registry_ids)
    exc = all_excerpts_real(record, abstract)
    return {"ok": cit and exc and not errs,
            "citation_ok": cit, "excerpts_ok": exc, "schema_errors": errs}


# ── Extraction (LLM; injectable) ─────────────────────────────────────────────

_SYSTEM = (
    "You are a Research Analyst. Extract ONE HypothesisRecord (JSON) describing the "
    "technique in the paper ABSTRACT provided. Extract a record whenever the abstract "
    "describes a concrete method/technique. RULES: "
    "(1) Every string in verbatim_excerpts MUST be copied EXACTLY, word-for-word, from "
    "the text between <abstract> and </abstract> — never from these instructions or the "
    "problem statement. (2) Omit source_citation; it is set for you. (3) Output a single "
    "JSON object and NOTHING else (no prose, no markdown fences). Output {} ONLY if the "
    "abstract describes no technique at all."
)

_FIELD_HINT = (
    "JSON shape (all string fields non-empty; use \"not reported\" if the abstract omits a value):\n"
    "{\n"
    '  "title": "<short technique name>",\n'
    '  "claimed_effect": "<what it does, the paper\'s terms, 2-3 sentences>",\n'
    '  "verbatim_excerpts": [{"text": "<EXACT quote from inside <abstract>…</abstract>>"}],\n'
    '  "reported_results": "<claimed numbers + setup, or \\"not reported\\">",\n'
    '  "applicability_check": {"requirements": ["<req>"], "verdict": "applicable|not_applicable|uncertain", "notes": "<why>"},\n'
    '  "known_failure_modes": ["<limitation>"],\n'
    '  "implementation_difficulty": "config_change|minor_code_change|new_module|major_refactor",\n'
    '  "confidence_flags": {"claimed_effect": "low|medium|high"}\n'
    "}"
)


def _norm_excerpt(e) -> dict | None:
    """Accept either {'text': …} or a bare string; return {'text', 'location'} or None."""
    if isinstance(e, str):
        return {"text": e, "location": None}
    if isinstance(e, dict) and isinstance(e.get("text"), str):
        return {"text": e["text"], "location": e.get("location")}
    return None


def _ground(record: dict, abstract: str) -> dict | None:
    """Keep only excerpts that are genuinely in the abstract (drop hallucinated ones);
    require ≥1 to survive; coerce required string fields. None ⇒ no grounded technique."""
    real = []
    for e in record.get("verbatim_excerpts", []) or []:
        ne = _norm_excerpt(e)
        if ne and excerpt_in_abstract(ne["text"], abstract):
            real.append(ne)
    if not real:
        return None
    record["verbatim_excerpts"] = real
    if not isinstance(record.get("reported_results"), str) or not record["reported_results"].strip():
        record["reported_results"] = "Not reported in the abstract."
    return record


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _stamp_citation(record: dict, paper: dict) -> dict:
    """Set source_citation from the REAL paper (authoritative; never the LLM's)."""
    authors = paper.get("authors") or ["unknown"]
    record["source_citation"] = {
        "title": paper.get("title", "untitled"),
        "authors": authors if authors else ["unknown"],
        "year": int(paper.get("year") or datetime.now().year),
        "arxiv_id": clean_arxiv_id(paper.get("id")),
        "url": paper.get("url") or (f"https://arxiv.org/abs/{clean_arxiv_id(paper.get('id'))}" if paper.get("id") else None),
        "venue": None, "github_url": None,
    }
    return {k: v for k, v in record.items() if k in _ALLOWED_KEYS}


def extract_record(paper: dict, problem: str, chat_fn) -> dict | None:
    """chat_fn(system, user) -> str. Returns a grounded, citation-stamped record, or None."""
    abstract = paper.get("abstract", "")
    user = (f"PROBLEM (context only — never quote this): {problem}\n\n"
            f"PAPER TITLE: {paper.get('title','')}\n"
            f"<abstract>\n{abstract}\n</abstract>\n\n{_FIELD_HINT}")
    rec = _parse_json(chat_fn(_SYSTEM, user))
    if not rec:
        return None
    rec = _ground(rec, abstract)           # drop hallucinated excerpts; keep if ≥1 real
    if rec is None:
        return None
    return _stamp_citation(rec, paper)


# ── Orchestration ────────────────────────────────────────────────────────────

def _default_chat_fn(backend_name: str):
    """Build a real LLM chat_fn from the Search Strategist's backends (local by default)."""
    import os
    from agents.search_strategist import (_AnthropicBackend, _OpenAICompatibleBackend,
                                          _resolve_backend_name, LLAMACPP_BASE_URL)
    name = _resolve_backend_name(backend_name)
    if name == "anthropic":
        be = _AnthropicBackend(model="claude-sonnet-4-5",
                               api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    else:
        be = _OpenAICompatibleBackend(model="qwen2.5-7b-instruct",
                                      base_url=LLAMACPP_BASE_URL, api_key="local")
    return lambda system, user: be.chat_text([{"role": "user", "content": user}], system=system)


def route_records(records: list[dict]) -> None:
    """Survivors → hypothesis queue + the Mode A→B escalation gate (non-blocking)."""
    HYPOTHESIS_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    queue = json.loads(HYPOTHESIS_QUEUE.read_text()) if HYPOTHESIS_QUEUE.exists() else []
    for r in records:
        queue.append({"proposed_at": datetime.now(timezone.utc).isoformat(),
                      "title": r["title"], "record": r})
    HYPOTHESIS_QUEUE.write_text(json.dumps(queue, indent=2))
    if records:
        from services.gates import gate_mode_b_escalation
        gate_mode_b_escalation(block=False, n_records=len(records),
                               titles=", ".join(r["title"] for r in records))


def analyze(problem: str, query: str, max_papers: int = 5,
            chat_fn=None, backend: str = "auto") -> list[dict]:
    """Retrieve → extract → verify → route. Returns the verified records."""
    from tools.fetch_papers import arxiv_search, load_registry, save_registry, existing_ids

    reg = load_registry()
    have = {clean_arxiv_id(i) for i in existing_ids(reg)} - {None}
    papers = arxiv_search(query, max_results=max_papers)
    for p in papers:                       # add retrieved papers to the registry (ADR-0009)
        p["id"] = clean_arxiv_id(p.get("id"))   # normalize version suffix
        if p.get("id") and p["id"] not in have:
            reg.setdefault("papers", []).append(p)
            have.add(p["id"])
    save_registry(reg)

    chat_fn = chat_fn or _default_chat_fn(backend)
    verified: list[dict] = []
    for p in papers:
        rec = extract_record(p, problem, chat_fn)
        if not rec:
            print(f"  · {p.get('id')}: no record extracted")
            continue
        v = verify_record(rec, p.get("abstract", ""), have)
        status = "✅ verified" if v["ok"] else f"⚠️ rejected ({_why(v)})"
        print(f"  · {p.get('id')}: {rec.get('title','?')[:50]} — {status}")
        if v["ok"]:
            verified.append(rec)
    route_records(verified)
    print(f"→ {len(verified)}/{len(papers)} verified and routed to the gate")
    return verified


def _why(v: dict) -> str:
    bad = []
    if not v["citation_ok"]:
        bad.append("citation not in registry")
    if not v["excerpts_ok"]:
        bad.append("excerpt not in abstract")
    if v["schema_errors"]:
        bad.append(f"{len(v['schema_errors'])} schema error(s)")
    return "; ".join(bad)


def main():
    ap = argparse.ArgumentParser(description="Research Analyst — Mode B MVP (ADR-0015)")
    ap.add_argument("--problem", default=(
        "A ~0.5B constructed VLM is multi-skill variance-limited: it cannot robustly "
        "hold object-grounding and MCQ-reasoning at once. Seeking capacity-efficient "
        "small-VLM architectures or anti-forgetting multi-task distillation techniques."))
    ap.add_argument("--query", default="small vision-language model multi-task distillation forgetting")
    ap.add_argument("--max-papers", type=int, default=5)
    ap.add_argument("--backend", default="auto", help="auto|local|api (default local; ADR-0013)")
    args = ap.parse_args()
    print(f"▶ Research Analyst — query={args.query!r}")
    analyze(args.problem, args.query, args.max_papers, backend=args.backend)


if __name__ == "__main__":
    main()
