# ADR-0013: Human Interface — the Operator Console

**Date:** 2026-06-15
**Status:** Proposed (for review)
**Context:** For the system to be *useful*, a human needs a real way to interact with it — frame the work at the start, watch progress and logs, approve the gated decisions, and intervene (pause/stop/kill). The HLD specified only a "Human Approval Queue" (§6.1 #8) and left the rest implicit, so the human-interaction story is currently invisible in the architecture and largely unbuilt (gap **B2**). This ADR proposes the interface. HLD §5.4 + Figure 1 are the summary; this is the design.

---

## Problem

§5.1–§5.3 of the HLD say *when* the human gates; they don't say *how* the human sees or drives the system. In practice today the entire interaction is **this chat (Claude Code) + the CLI**, with `dashboard.py` (Streamlit, read-only Phase-0/2 metrics) and `PushNotification` as the only purpose-built surfaces. That is enough for a single operator running it by hand, but it does not make the system legible or operable to anyone else, and it has **no stop/kill control, no approvals surface, and no start-of-run intake**.

The human is *most* involved at the **start** (goals, success criteria, target device, eval set, authorizing the search space) and at strategic gates — and least involved during the autonomous Mode A grind. The interface must match that shape: rich framing up front, glanceable status in the middle, crisp controls and approvals at the edges.

---

## Decision — one console, four surfaces, modality chosen per surface

Do **not** build a single monolithic GUI. Different interactions want different modalities; pick per surface and let them share one state backend (the metrics DB + queues already exist).

| # | Surface | Role | Modality (chosen) | Why |
|---|---|---|---|---|
| 1 | **Goal & scope intake** | Frame the problem; approve the search space the agent may explore; phase transitions | **Chat** for framing + a **checked-in config file** (`run.yaml`) as the durable record | Framing is open-ended and benefits from dialogue; the config file makes the approved scope reproducible and diffable |
| 2 | **Status dashboard** | Progress, live run/queue status, logs, Pareto frontier, ledger | **GUI** (extend `dashboard.py`) | Status is glanceable and visual; a dashboard already exists as the seed |
| 3 | **Approvals & gates** | The §5.1 decisions: deploy, big compute, eval-set change, Tier-2/spec approval, Mode-A→B escalation (Decision Dossier) | **GUI inbox** + **proactive notification**; **chat** for the reasoning | Approvals must be auditable (append-only log) and reach the human when away; the *context* (a dossier, a rationale) reads well in chat |
| 4 | **Run controls** | Pause / resume / stop a run / **kill** a runaway experiment | **GUI buttons + CLI** | Intervention must be immediate and unambiguous; both a button and a scriptable command |

Cross-cutting:
- **Notifications** (desktop + phone) are the async channel for "a gate needs you" / "a long run finished." Already used (`PushNotification`); formalize a notification policy.
- **Chat (the Search Strategist's voice)** is where rationales, "why this experiment," and "reconsider X" live — the agent already writes rationales (§6.1 #1); expose them.

---

## What exists vs. what to build

**Exists (reuse, don't rebuild):**
- `dashboard.py` (Streamlit) — read-only metrics/Pareto/Phase-2 views → the seed for surface #2.
- `metrics.db` + experiment ledger + experiment/construction queues → the shared state backend.
- `PushNotification` → the async alert channel.
- Chat via Claude Code → surfaces #1 and the reasoning half of #3, today.

**Build (phased, smallest-useful-first):**

- **H1 — Run config + controls (highest value, smallest).** A `run.yaml` intake (goal, criteria, device, eval set, allowed search space) read by the loop; a control file/flag the runners poll so the human can `pause`/`stop`/`kill` (CLI writes it; runners check it between steps — the loops already checkpoint per step). This closes the "no kill switch" gap directly.
- **H2 — Live status on the dashboard.** Extend `dashboard.py` with a "Runs" tab: queue depth, current experiment, last-N ledger rows, live log tail, and a kill button wired to H1's control flag.
- **H3 — Approvals inbox.** A persisted append-only approval queue (the HLD §6.1 #8 component) surfaced as a dashboard tab + notification; gated actions block on it. Decision Dossiers (§4.2) render here.
- **H4 — Strategist rationale view.** Surface the agent's proposal rationale + the hypothesis table state read-only, so the human understands *why* before approving.

H1 is the one to do first: it is small, removes a real safety gap (stop/kill), and makes unattended overnight runs (like the B1.3 chain) controllable.

---

## Consequences

- The "operator console" becomes a first-class part of the architecture (HLD §5.4, Figure 1 top band), not an implicit afterthought.
- Gap **B2** (no operator-facing surface) is now scoped into H1–H4 rather than left open.
- Stop/kill is enforced by a **control flag the runners poll between checkpoints**, not by killing processes blindly — consistent with the resumable job-queue model (HLD §7.4); a killed experiment costs at most one step.
- The backend is unchanged: all four surfaces are producers/consumers of the existing queues + ledger + (new) approval log. No new orchestration.

---

## Open questions

- **GUI framework:** stay on Streamlit (fast, already in use, fine for a single operator) or move to a small FastAPI + web UI when multi-user/remote becomes a need? Recommendation: stay on Streamlit through Phase 2; revisit with the §7.3 Mac-bridge upgrade.
- **Auth / multi-operator:** out of scope while it's a single-owner project; the append-only approval log should still record *who* approved for when that changes.
- **How much of intake is chat vs. form:** start chat-first (lowest friction), promote the settled fields into `run.yaml` as they stabilize.
