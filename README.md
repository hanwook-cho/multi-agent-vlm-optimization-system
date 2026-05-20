# Multi-Agent VLM Optimization System

An autonomous agent system that compresses the time required to produce a competitive edge vision-language model — from the team-months of focused expert work that produced models like LFM2.5-VL-450M, SmolVLM-500M, and MiniCPM-V, to solo-developer-months using the system as the optimization tool.

## Status

**Phase 0 in progress** (foundations and reference baselines). The project is structured as five phases over ~6-7 months solo full-time-equivalent. See `docs/VLM_Optimization_Goals.md` §5 for the phase structure.

The repository is currently **private during Phase 0-1** and will go public at the end of Phase 1 alongside a working Mode A optimization loop. See `docs/decisions/0008-public-repo-timing.md` once written.

## What this is

The system, given a vision-language task and a target edge device (iPhone 16 Pro, Raspberry Pi 5), autonomously produces a deployable inference pipeline measured against real on-device latency, memory, and accuracy. It starts from well-known optimization techniques (Mode A) and can escalate to research-driven exploration (Mode B) when known techniques are exhausted.

**The system is the deliverable. The compressed time-to-result is the central claim. A competitive model is the proof-of-work.**

## What this is not

- Not a generic AutoML tool — VLM-specific, edge-specific.
- Not "fully autonomous research" — humans gate consequential decisions (architecture changes, evaluation metric changes, mode escalation, device deploys). See `docs/HLD.md` §5.
- Not a competitor to Liquid AI or Apple on individual model quality — the contribution is the *method* that compresses optimization time, demonstrated by producing one competitive model in solo-months rather than team-months. See `docs/PriorArt.md`.

## Documentation

- **`docs/VLM_Optimization_Goals.md`** — ultimate goal, success criteria, phase structure, conduct rules.
- **`docs/VLM_Optimization_HLD.md`** — architectural design (two agents + seven services, Mode A / Mode B, two-Mac topology).
- **`docs/VLM_Optimization_PriorArt.md`** — position relative to AutoML/NAS, LLM-driven AutoML, AI-Scientist, and production edge inference.
- **`docs/VLM_Optimization_DetailedPlan_Phase0.md`** — week-by-week Phase 0 execution plan.

## Quick start

Not yet runnable. Phase 0 establishes infrastructure and measured reference baselines; Phase 1 brings the Mode A loop online. Re-check end of Phase 1 for a working quick-start.

## How to follow along

Not yet — repo is private during Phase 0-1. After the Phase 1 public reveal, issues and discussions welcome.

## License

Apache 2.0 (see `LICENSE`). Third-party models and datasets used by this project are governed by their own licenses; see `docs/THIRD_PARTY.md` (written in Phase 0 Week 5).
