# Architecture Decision Records

One file per non-trivial design decision, named `NNNN-short-title.md`.

Each ADR covers: **Context** (what forced this decision), **Decision** (what was chosen),
**Rationale** (why), **Consequences** (what this commits us to or forecloses),
**Open issues** (if any remain).

ADRs are written as decisions are made, not retroactively. They are the authoritative
record of *why* the system is designed the way it is.

## Index (actual state)

The numbering diverged from the original Phase-0 plan. Authoritative current list:

| # | Title | Status |
|---|---|---|
| 0001 | Mac measurement methodology | ✅ written |
| 0002 | iOS measurement methodology | ✅ written |
| 0003 | iPhone baseline numbers | ✅ written |
| 0004 | Stage-A eval set | ✅ written *(repurposed — the plan had reserved 0004 for "Pi measurement methodology")* |
| 0005 | *(planned: Pi model-fit summary)* | ⏸️ deferred with Raspberry Pi 5 — never written |
| 0006 | *(planned: FastVLM-on-Pi not viable)* | ⏸️ deferred with Raspberry Pi 5 — never written |
| 0007 | *(planned: license posture)* | ❌ not written — decision recorded in [`../THIRD_PARTY.md`](../THIRD_PARTY.md) instead |
| 0008 | *(planned: public-repo timing)* | ❌ not written — decision executed (repo public at end of Phase 1); recorded in Goals §3 / Phase-0 plan §5.5 |
| 0009 | Literature-tool eval | ✅ written |
| 0010 | Search Strategist backend | ✅ written |
| 0011 | Phase-2 strategy correction | ✅ written |
| 0012 | System-driven student construction | ✅ written |
| 0013 | Human interface / operator console | ✅ written |

Per the policy above, ADRs are written *as decisions are made*, not retroactively — so the
unwritten 0005–0008 are left as-is. References to them elsewhere now point to where the
actual decision lives (or note it was deferred) rather than back-filling a record after the fact.
