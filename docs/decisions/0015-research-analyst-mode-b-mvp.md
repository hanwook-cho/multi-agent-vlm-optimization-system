# ADR-0015 — Research Analyst (Mode B) MVP

**Date:** 2026-06-18
**Status:** Accepted
**Builds on:** ADR-0009 (literature ingestion), HLD §4.2 (Mode B), §6.2 (hypothesis record), §6.4 (citation-hallucination guard).

## Context

Mode A (the Search Strategist — propose configs/specs over *known* techniques) is
built and was exercised through P2-B1. Mode B — reading the *literature* to surface
*new* techniques — was designed in the HLD but never implemented. It is the project's
distinctive contribution versus plain AutoML ("Mode A is a re-implementation of a
known technique; Mode B is where the novelty lives"). After P2-B1 hit a capacity wall,
Mode B is also the cheap, on-thesis next step: it can surface the technique that
addresses the wall, at no training cost.

Most of the scaffolding already exists: `tools/fetch_papers.py` + the arXiv registry
(ADR-0009), `schemas/hypothesis_record.schema.json`, the Strategist's LLM backends,
and the escalation gate (`services/gates.gate_mode_b_escalation` + `decision_dossier`).
The only missing piece is the agent and — critically — the *trust boundary*.

## Decision

Build a **one-shot, human-gated Research Analyst MVP** (`agents/research_analyst.py`):
**retrieve → extract → verify → route.** The load-bearing decision is the **verifier**:

> **Citations and quotes are verified deterministically; the LLM is never trusted for
> them.** The `source_citation` is set by *us* from the real retrieved paper (the LLM
> doesn't get to name the paper). A record is routed to the human gate only if (a) its
> `arxiv_id` is a real paper in the registry, (b) **every `verbatim_excerpt` is an exact
> (whitespace/case-normalized) substring of the paper's abstract**, and (c) it
> schema-validates. This makes the AI-Scientist failure mode (HLD §6.4 — hallucinated
> citations/quotes) structurally impossible to pass the gate, by string-matching rather
> than asking the model to self-certify.

Extraction works from the **abstract only** in the MVP (keeps the excerpt check fully
deterministic and the cost to a single LLM call per paper). The LLM call is injectable,
so the verifier and extraction contract are unit-tested with no network/LLM.

## Scope

**In:** arXiv search (ADR-0009) → per-paper abstract → LLM `HypothesisRecord` →
deterministic verify (citation + verbatim + schema) → survivors written to
`artifacts/hypothesis_queue.json` and posted to the Mode A→B gate (surfaces in the
operator console's Approvals tab). CLI: `python agents/research_analyst.py --problem … --query …`.
CI-safe tests for the verifier + the citation-stamping/extraction contract.

**Out (deferred):** full-PDF/RAG extraction (excerpts beyond the abstract), autonomous
escalation triggering, multi-round agent loops, and auto-promotion of an approved
technique into a `StudentSpec` parameter (that stays a human/Strategist step).

## Consequences

- The system now has both halves: Mode A proposes over known techniques; Mode B reads
  the literature and proposes new ones — both human-gated, both on the local backend
  by default.
- The anti-hallucination guard is *structural*, not prompt-based — the strongest place
  to put it for a project whose credibility rests on honest, verifiable claims.
- Abstract-only extraction may miss techniques whose detail lives in the body; that is
  the explicit MVP/precision trade (better to surface fewer, fully-grounded records).
- Running it needs the local LLM server (`scripts/start_strategist_llm.sh`) or an
  opt-in API backend; retrieval needs network for the arXiv API.

## Open issues

- Promotion path: how an *approved* hypothesis record becomes a Mode A spec parameter
  (manual for now; a future `propose_from_record` on the Strategist could automate it).
- Body-text extraction with a verifier that stays deterministic (e.g. verify excerpts
  against fetched full text) — the natural next increment.
