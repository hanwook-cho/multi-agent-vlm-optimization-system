# Multi-Agent VLM Optimization System

An autonomous agent system that compresses the time required to produce a competitive edge vision-language model — from the team-months of focused expert work that produced models like LFM2-VL-450M, SmolVLM-500M, and MiniCPM-V, to solo-developer-months using the system as the optimization tool.

**The system is the deliverable. The compressed time-to-result is the central claim. A competitive model is the proof-of-work.**

## Status

**Phase 2 in progress.** The Mode A loop is closed and the system now *constructs* models, not just configures them: the Search Strategist proposes a config experiment or a student-construction spec, deterministic services build/distill/evaluate it on a held-constant path, results land in the experiment ledger, and the agent re-routes (ADR-0011/0012). An operator console (below) gives a live UI to run, watch, steer, and gate it.

See [`STATUS.md`](STATUS.md) for the detailed state and [`docs/VLM_Optimization_HLD.md`](docs/VLM_Optimization_HLD.md) §6 (Figure 1) for the architecture.

## What this is

The system, given a vision-language task and a target edge device (iPhone 16 Pro, Raspberry Pi 5), autonomously produces a deployable inference pipeline measured against real on-device latency, memory, and accuracy. It starts from well-known optimization techniques (Mode A) and can escalate to research-driven exploration (Mode B) when known techniques are exhausted. Humans gate consequential decisions (architecture changes, eval-metric changes, mode escalation, device deploys) — see [`docs/VLM_Optimization_HLD.md`](docs/VLM_Optimization_HLD.md) §5.

## Operator console

A live browser UI (ADR-0013) to run, watch, steer, and gate the system: pause/stop/kill controls, current-run status and live logs, recent constructed-student scores, a docked chat with the Search Strategist (local by default), and an approvals queue for gated decisions.

![Operator console](docs/assets/operator_console.svg)

```bash
streamlit run operator_console.py     # → http://localhost:8501
```

## How to

Requires Python 3.11+ (CI runs 3.11; an Apple-Silicon Mac is needed for the MPS training/eval paths). Heavy ML deps (torch, transformers, peft) and `streamlit` are used by the runners and the console.

```bash
# 1. install
python -m pip install -r requirements-dev.txt          # test + dev deps
#    (torch / transformers / peft / streamlit as needed for runners + console)

# 2. run the test suite (deterministic logic; no model downloads)
python -m pytest -q

# 3. launch the operator console
streamlit run operator_console.py                       # → http://localhost:8501

# 4. read-only metrics dashboard (Phase 0/2 baselines, Pareto)
streamlit run dashboard.py
```

### Operator controls (also available from the CLI)

```bash
# pause / resume / stop (graceful) / kill (abort) a long run; loops react at the next checkpoint
python -m services.run_control status
python -m services.run_control pause "lunch"
python -m services.run_control resume
python -m services.run_control kill  "swap-thrashing"

# approvals queue (gated decisions): list / request / approve / reject
python -m services.approvals list
python -m services.approvals approve <id>
```

### Run a system-driven construction (Phase 2, compute-gated)

The Search Strategist proposes a `StudentSpec`; the construction loop assembles a student (LM + vision encoder + projector), distills it from the Qwen2.5-VL-3B teacher, and scores it same-path. See [`docs/decisions/0012-system-driven-student-construction.md`](docs/decisions/0012-system-driven-student-construction.md).

```bash
# the construction loop consumes the agent's queued spec, builds, evaluates, records to the ledger
python services/construction_loop.py --eval --align-steps 200 --distill-steps 1000

# the generic builder directly (with a spec), for one-off runs
python runners/build_student.py --spec tests/fixtures/student_spec_p2b1_qwen05b_siglip.json --smoke
```

Runs tee their output to `artifacts/logs/`, and the operator console auto-points at the newest one (with an optional auto-refresh) — so progress shows up on the Monitor tab without wiring a log path.

`run.yaml` (see [`configs/run.example.yaml`](configs/run.example.yaml)) declares the authorized goal, success criteria, eval set, allowed hypotheses, and the agent/chat backend (`local` by default, `api` opt-in).

## Documentation

- [`docs/VLM_Optimization_Goals.md`](docs/VLM_Optimization_Goals.md) — ultimate goal, success criteria, phase structure, conduct rules.
- [`docs/VLM_Optimization_HLD.md`](docs/VLM_Optimization_HLD.md) — architecture (agents + services, Mode A / Mode B); §6 Figure 1 is the system diagram; §6.5 Amendment A is system-driven construction.
- [`docs/VLM_Optimization_PriorArt.md`](docs/VLM_Optimization_PriorArt.md) — position relative to AutoML/NAS, LLM-driven AutoML, AI-Scientist, and production edge inference.
- [`docs/decisions/`](docs/decisions/) — ADRs (e.g. 0011 Phase-2 strategy correction, 0012 system-driven construction, 0013 operator console).
- [`docs/observations/`](docs/observations/) — dated experiment results, including negative results.

## What this is not

- Not a generic AutoML tool — VLM-specific, edge-specific.
- Not "fully autonomous research" — humans gate consequential decisions (see HLD §5).
- Not a competitor to Liquid AI or Apple on individual model quality — the contribution is the *method* that compresses optimization time, demonstrated by producing one competitive model in solo-months rather than team-months. See `docs/VLM_Optimization_PriorArt.md`.

## License

Apache 2.0 (see [`LICENSE`](LICENSE)). Third-party models and datasets used by this project are governed by their own licenses.
