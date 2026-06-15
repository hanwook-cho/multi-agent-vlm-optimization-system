# ADR-0013: Human Interface — the Operator Console

**Date:** 2026-06-15
**Status:** Accepted (design finalized 2026-06-15; phased build H1–H4, H1 done)
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

## Console information architecture (finalized 2026-06-15)

The four surfaces above map onto this layout, validated against mockups:

- **Tabs:** `Setup` · `Monitor` · `Approvals`.
  - `Setup` — the new-optimization form; writes `run.yaml` (the `RunConfig` intake).
  - `Monitor` — live run status, scores vs benchmark, log tail, and the pause/stop/kill controls.
  - `Approvals` — the full inbox: pending gated decisions, history, and Decision Dossier detail.
- **Global chrome (present on every tab):**
  - **Chat dock** — a collapsible side drawer holding *one* continuous Search Strategist session, toggled from the top bar. Chat is **not** a tab — you steer/ask while looking at any view.
  - **Approvals bell** — a badge with the pending count, on every tab; opens the inbox.
  - **Backend indicator** — shows local vs api (the configurable, default-local backend).
- **Approvals = one queue, three surfaces.** A single approval log is surfaced as (1) the global bell, (2) an inline card on `Monitor` for the item blocking the active run, and (3) the `Approvals` tab. Urgent/blocking decisions additionally raise a slide-over + push notification. The three surfaces share one backend — they are not separate windows or separate state.
- **Single source of truth, no separate windows.** One chat session, one approval log, one run-control state; every surface is a view of those. Nothing opens a separate browser window — only drawers and tabs within the one app.

This keeps state unified and matches the HLD §7.1 model (queue + DB + approval queue): the console is just readers/writers of shared state.

## Chat: ownership, backend, and deployment (added 2026-06-15)

**Who owns the chat.** "Chat" is three jobs with three owners:
- **Status / telemetry** ("what's running?", "show the frontier") → the **UI** answers deterministically from `metrics.db` / the ledger. Not an LLM; not really chat.
- **Reasoning / steering** ("why P2-B1?", "reconsider", "stop after this") → the **Search Strategist agent** owns it. It already holds the state (hypothesis table, ledger, frontier) and writes rationales, so conversational steering is a small extension, not a new component. **Claude Code is the *development* harness, not the deployed chat.**
- **Open-ended framing / novel research** → optional escalation to a **frontier API** (same rationale as the Research Analyst, HLD §7.2), human-initiated and bounded.

**Backend is configurable; default is local.** The agent/chat backend is selectable between **local** (llama.cpp + Qwen2.5 — private, free) and **api** (frontier — opt-in, per-token cost), and **defaults to local**. API is strictly opt-in: it is selected only by an explicit `backend=` / `STRATEGIST_BACKEND` (`local`|`api`|…) / `run.yaml: chat_backend`, **never** implicitly by the presence of `ANTHROPIC_API_KEY`. Implemented in `agents/search_strategist._resolve_backend_name`; declared by `RunConfig.chat_backend` (default `local`).

**Chat respects the gates.** A chat request to do something gated (deploy, change the eval set, escalate to Mode B) does **not** bypass §5.1 — the chat *proposes and explains*; the human still approves the irreversible/expensive/epistemically-risky action. And per the instruction-source boundary, the chat owner acts on the operator's words, not on instructions embedded in content it reads (papers, web, tool output).

**Deployment / portability.** The UI front-end is a **web app** (Streamlit, localhost) and is **portable** (pure Python + SQLite + files) — it runs on Mac, Linux, or Windows unchanged. The **compute backend** (training/eval on MPS/Metal, CoreML for iPhone) is **Apple-Silicon-bound by design** and is *not* containerized: Docker on macOS has no MPS access, so the compute stays a native process. The UI and compute are already decoupled via the DB + files + control flag, so the UI may run on a second machine (HLD §7.2 Agent Mac) pointing at shared state. A **local chat backend is itself a backend service** (lives with the compute); an API chat backend keeps the UI host dependency-free. Docker is worthwhile only if/when the UI is hosted remotely (Linux, no GPU dependency) — not for the Mac compute.

## What exists vs. what to build

**Exists (reuse, don't rebuild):**
- `dashboard.py` (Streamlit) — read-only metrics/Pareto/Phase-2 views → the seed for surface #2.
- `metrics.db` + experiment ledger + experiment/construction queues → the shared state backend.
- `PushNotification` → the async alert channel.
- Chat via Claude Code → surfaces #1 and the reasoning half of #3, today.

**Build (phased, smallest-useful-first):**

- **H1 — Run config + controls (highest value, smallest). ✅ DONE (2026-06-15).** `services/run_control.py` — pause/stop/kill via a control file the long loops poll at each checkpoint (`rc.checkpoint()` in `build_student._train_loop` and the eval loop; a `TrainerCallback` in `finetune_vlm`). stop = graceful (caller saves), kill = abort, pause = block until resume; a halted run costs ≤1 step (consistent with §7.4). CLI: `python -m services.run_control {status,pause,resume,stop,kill,clear}`. Intake: `schemas/run_config.py` + `configs/run.example.yaml` (`run.yaml` → goal, criteria, device, eval set, allowed hypotheses), stamped into the build record. Enforcing `allowed_hypotheses` against the strategist's proposals is deferred to a later step. 11 tests.
- **H2 — Console shell + Monitor + controls. ✅ DONE (2026-06-15).** `operator_console.py` (Streamlit, `streamlit run operator_console.py`): global bar (state + backend indicator + approvals bell), Setup/Monitor/Approvals tabs, sidebar chat-dock placeholder. The `Monitor` tab shows current run (stage/step/loss parsed from the log), queue depth, recent constructed-student ledger rows with scores, live log tail, and **working pause/resume/stop/kill/clear buttons** wired to `run_control` (the loops react at their next checkpoint). Data layer factored into `services/console_data.py` (pure, 6 tests); the script verified end-to-end via Streamlit `AppTest` (no render exceptions).
- **H2b — Chat dock.** The global collapsible drawer holding one Search Strategist session (local backend by default), available on every tab.
- **H3 — Approvals (one queue, three surfaces). ✅ DONE (2026-06-15).** `services/approvals.py` — the persisted approval log (HLD §6.1 #8): `request_approval` / `list_pending` / `decide` / `wait_for_approval` (gated code can block on a request), with who/when/note recorded for audit. CLI: `python -m services.approvals {list,request,approve,reject}`. Surfaced as the console's global **bell** count, the inline **Monitor card** (top pending + approve/reject), and the **Approvals tab** (pending list + decision history) — all reading the one log. 7 tests. Pending: wiring specific gates to *create* requests (e.g. deploy/escalate), and the Decision Dossier render; urgent slide-over/push are follow-ups.
- **H4 — Setup form + rationale. ✅ DONE (2026-06-15).** The `Setup` tab is now a form (goal, success criteria, device, eval set, allowed hypotheses, chat-backend default, notes) that validates as a `RunConfig` and writes `run.yaml` (`schemas.run_config.save_run_config`) — no hand-editing. The `Approvals` tab gained a "strategy context" section: recent agent proposals + rationale (`console_data.latest_proposals`) and the hypothesis-table state (`console_data.hypothesis_rows`), so gated decisions come with the why.

H1–H4 are done — a live browser console (`streamlit run operator_console.py`): Setup form (writes `run.yaml`), Monitor (watch + stop/kill), chat dock (local/api), and Approvals (decide gated items, with strategy context). **The remaining gap is enforcement** — wiring specific gates (deploy, non-smoke build, escalation) to *create* approval requests and block on them via `approvals.wait_for_approval`, so the queue becomes load-bearing rather than an inbox.

---

## Consequences

- The "operator console" becomes a first-class part of the architecture (HLD §5.4, Figure 1 top band), not an implicit afterthought.
- Gap **B2** (no operator-facing surface) is now scoped into H1–H4 rather than left open.
- Stop/kill is enforced by a **control flag the runners poll between checkpoints**, not by killing processes blindly — consistent with the resumable job-queue model (HLD §7.4); a killed experiment costs at most one step.
- The backend is unchanged: all four surfaces are producers/consumers of the existing queues + ledger + (new) approval log. No new orchestration.

---

## Resolved at finalization (2026-06-15)

- **GUI framework → Streamlit** through Phase 2 (fast, already in use, single-operator). Revisit only if remote/multi-user is needed, alongside the §7.3 Mac-bridge upgrade.
- **Intake → both:** a `Setup` form (writes `run.yaml`) for the durable scope, plus the chat dock for conversational framing/steering.
- **Layout → tabs + global chrome** (see Console information architecture above); single source of truth per surface; no separate windows.

## Open questions

- **Auth / multi-operator:** out of scope while it's a single-owner project; the append-only approval log should still record *who* approved for when that changes.
