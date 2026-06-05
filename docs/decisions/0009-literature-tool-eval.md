# ADR-0009 — Literature Ingestion Tooling

**Date:** 2026-06-05  
**Status:** Accepted  
**Phase:** Phase 0 — Reference Baselines

---

## Context

Phase 1 will run optimization experiments (quantization, pruning, distillation, KV-cache compression) on small-edge VLMs. Each experiment should be traceable to one or more papers that motivated it. The project needs tooling that can:

1. **Discover** relevant papers (arXiv preprints, conference proceedings)
2. **Store** paper metadata + abstracts in a structured, queryable way
3. **Link** papers to experiment configurations (`ExperimentConfig.hypothesis_source`)
4. **Retrieve** related work when designing new experiments

The key constraint is that this is a solo project. Complexity is expensive — a tool that requires hours of setup or maintenance is worse than a simple text file that gets used.

---

## Options Evaluated

### Option A — arXiv API + JSON registry (chosen)

**What:** A Python script queries the arXiv API for relevant papers by keyword, downloads title/abstract/authors/PDF URL. Results are stored in `docs/literature/registry.json`. A thin CLI (`tools/fetch_papers.py`) handles search and deduplication. Papers are linked to experiments by arXiv ID in `ExperimentConfig`.

**Pros:**
- Zero infrastructure — just a JSON file and a 50-line script
- arXiv has the vast majority of relevant ML papers (CVPR, ICLR, NeurIPS all post preprints)
- Queryable with `jq` or Python without standing up a vector DB
- Reproducible — the registry is version-controlled; anyone who clones the repo can reproduce the literature context
- arXiv API is free, no key required, rate limit is generous (3 req/s)

**Cons:**
- No semantic search — keyword search only (arXiv full-text search is not semantic)
- Won't surface papers that don't use the exact search terms
- Manual curation still required — arXiv search returns noisy results

**Verdict:** Sufficient for Phase 0–1. The project doesn't yet need semantic retrieval over 500 papers; it needs a handful of key papers per experiment tracked in a structured way.

---

### Option B — Semantic Scholar API

**What:** Semantic Scholar's open API returns papers with citation graphs, semantic similarity scores, and author disambiguation. More powerful search than arXiv.

**Pros:**
- Semantic search (`/paper/search` endpoint supports natural language queries)
- Citation graph: find papers that cite a given paper, or papers cited by it
- Covers non-arXiv papers (ACL Anthology, IEEE, ACM)

**Cons:**
- Requires an API key (free but needs registration)
- Rate limits are tighter (100 req/5min without key, 10k/day with key)
- Abstracts occasionally missing or truncated for older papers
- No PDF download — links out to publisher or Semantic Scholar reader

**Verdict:** Useful as a supplementary search tool for Phase 2 when the paper volume grows. Not needed for Phase 0–1. Noted as the upgrade path.

---

### Option C — Local PDF folder + LangChain vector retrieval

**What:** Download PDFs manually into `docs/literature/pdfs/`, extract text with `pypdf`, chunk and embed with a local model (e.g. `nomic-embed-text`), store in ChromaDB. Semantic search via LangChain retriever.

**Pros:**
- True semantic search over full paper text, not just abstracts
- Works offline
- Can answer questions like "which papers discuss MobileCLIP encoder compression?"

**Cons:**
- Significant setup: ChromaDB, embedding model, LangChain, PDF extraction pipeline
- Embedding all papers on each new clone is slow without GPU
- ChromaDB is not trivially version-controlled (binary blobs)
- Maintenance burden: dependency versions, model availability

**Verdict:** Over-engineered for a project with <50 papers in scope for Phase 1. Revisit for Phase 3+ if literature-aware agent design requires semantic retrieval over a large corpus.

---

## Decision

**Option A — arXiv API + JSON registry.**

Simple, reproducible, zero infrastructure. Upgraded to Option B (Semantic Scholar) when Phase 2 expands the experiment scope beyond ~50 papers.

---

## Implementation

### File layout

```
docs/literature/
  registry.json          — all tracked papers (arXiv ID, title, abstract, tags, links)
  README.md              — how to add papers, tagging conventions
tools/fetch_papers.py    — CLI: search arXiv, add to registry, dedup
```

### Registry schema

```json
{
  "papers": [
    {
      "id":        "2412.01234",
      "source":    "arxiv",
      "title":     "...",
      "authors":   ["..."],
      "year":      2024,
      "abstract":  "...",
      "url":       "https://arxiv.org/abs/2412.01234",
      "pdf_url":   "https://arxiv.org/pdf/2412.01234",
      "tags":      ["quantization", "vision-encoder", "mobile"],
      "added":     "2026-06-05",
      "notes":     "Motivates Task 4.1 KV-cache compression experiment"
    }
  ]
}
```

### Tags (Phase 1 relevant)

| Tag | Covers |
|---|---|
| `quantization` | GPTQ, AWQ, GGUF, INT4/INT8, mixed-precision |
| `pruning` | structured/unstructured, magnitude/gradient |
| `distillation` | knowledge distillation for VLMs |
| `vision-encoder` | ViT compression, MobileCLIP, efficient patch selection |
| `kv-cache` | KV cache compression, eviction, quantization |
| `mobile` | on-device inference, edge deployment |
| `vlm-arch` | VLM architecture papers (LLaVA, MiniCPM-V, LFM2, etc.) |
| `benchmark` | evaluation methodology, datasets |

### Linking to experiments

`ExperimentConfig` has a `hypothesis_source` field. For literature-motivated experiments, set it to the arXiv ID:

```json
{
  "hypothesis_source": "arxiv:2412.01234",
  "hypothesis": "Applying AWQ INT4 quantization to the LM backbone reduces TTFT by 30% with <2pp POPE accuracy drop"
}
```

---

## Seed papers to add

These are the papers directly relevant to Phase 1 experiments. To be added via `tools/fetch_papers.py` at Phase 1 kickoff:

| arXiv ID | Paper | Relevance |
|---|---|---|
| 2306.00978 | AWQ: Activation-aware Weight Quantization | LM backbone INT4 quantization |
| 2210.17323 | GPTQ: Accurate Post-Training Quantization | Alternative INT4 method |
| 2404.14619 | MiniCPM-V: A GPT-4V Level MLLM on Your Phone | MiniCPM-V architecture reference |
| 2501.05510 | LFM-2: Scalable and Efficient Multimodal Language Models | LFM2 architecture reference |
| 2211.05100 | FastViT: A Fast Hybrid Vision Transformer | Vision encoder efficiency baseline |
| 2402.09906 | SnapKV: LLM Knows What You are Looking for Before Generation | KV-cache compression |
| 2306.07929 | LLaVA-1.5: Improved Baselines with Visual Instruction Tuning | LLaVA architecture baseline |

---

## Consequences

- All Phase 1 experiment hypotheses cite a paper ID from the registry
- Literature discovery is manual + keyword arXiv search — acceptable for Phase 0–1 scope
- Upgrade path to Semantic Scholar API is documented and requires minimal code change (swap the search function in `fetch_papers.py`)
- The registry is version-controlled and part of the reproducibility record
