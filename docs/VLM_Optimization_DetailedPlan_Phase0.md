# Multi-Agent System for VLM Optimization — Detailed Plan, Phase 0

*Phase 0: Foundations and reference baselines. The first of five phase plans. Subsequent plans (Phase 1, 2, 3, 4) will be written after Phase 0 is reviewed and locked.*

**Status.** Draft v2 (updated to reflect Goals v3: expanded baseline set including SmolVLM-500M, MiniCPM-V 4.6, Qwen2.5-VL-3B; Phase 4 reusability proof in scope; time-compression framing).
**Last updated.** May 11, 2026.
**Companion documents.** Goals v3 (§5 Phase 0 exit criteria, §6 conduct rules), HLD (architecture), Prior Art (related work).
**Phase 0 goal.** Stand up project infrastructure and lock in measured reference baselines for four small-edge VLMs (LFM2.5-VL-450M, FastVLM-0.5B, SmolVLM-500M, MiniCPM-V 4.6) on iPhone 16 Pro and Raspberry Pi 5 (4 GB), plus measure Qwen2.5-VL-3B as the Phase 2 starting point on Mac mini (M4, 16 GB) initially, with measurements to be repeated on the M5 Pro 32 GB when it becomes available.
**Duration estimate.** 4-5 weeks solo, full-time-equivalent.

---

## 1. How to read this plan

Phase 0 is organized as five weeks. Each week has 2-6 concrete tasks. Each task has:

- **What:** the concrete deliverable, scoped tightly enough that "done" is unambiguous.
- **Why:** which Goals §5 exit criterion or HLD architectural commitment it serves.
- **How:** specific approach, key decisions already made, libraries/tools to use.
- **Risk and time honesty:** where you'll likely get stuck and what to do about it.
- **Done when:** the acceptance criterion. Boolean — either met or not.

Tasks within a week can usually be reordered or partially parallelized. Tasks across weeks have dependencies that are flagged.

The plan assumes you work focused hours on the project. If you split attention with other commitments, double the time estimates. **The honest solo failure mode is "spent three weeks on the repo skeleton and got stuck on iPhone provisioning"** — I've tried to call out the likely time sinks.

---

## 2. Phase 0 at a glance

**Week 1: Repo skeleton and DeviceDescriptor scaffolding.** Get the foundation right because everything else builds on it. Private repository (public at end of Phase 1 — see ADR-0008), license, docs ported in, all contract schemas defined, four DeviceDescriptors written (iPhone, Pi 5, Mac mini M4 16 GB, M5 Pro 32 GB).

**Week 2: Mac-only baselines + VLMEvalKit integration.** Measure Qwen2.5-VL-3B on Mac mini (the Phase 2 starting point; M5 Pro 32 GB measurements follow when available). Stand up VLMEvalKit and run quality evaluations of all five reference models on benchmark slices. *Why this comes before iPhone/Pi: the Mac mini is already in hand, no provisioning friction, and validating data-plumbing on Mac before pushing to iPhone/Pi means the slower-to-iterate devices have working measurement code by the time you touch them.*

**Week 3: iPhone reference baselines (4 models).** Stand up LFM2.5-VL-450M, FastVLM-0.5B, SmolVLM-500M, and MiniCPM-V 4.6 on iPhone 16 Pro. This is the riskiest week — Apple developer provisioning has bitten more solo projects than anyone wants to admit. The data-plumbing from Week 2 should now be debugged, so iPhone work focuses on iOS-specific issues.

**Week 4: Pi 5 reference baselines + eval set assembly.** Pi 5 4 GB with LFM2.5-VL-450M, SmolVLM-500M, MiniCPM-V 4.6 (where they fit) via llama.cpp. Frozen public-photo evaluation set assembled in parallel.

**Week 5: Dashboard, build-vs-adopt spike, license posture, blog draft.** Render the Pareto plot, decide on literature-ingestion tooling, document license posture, draft the reveal blog post.

**Exit gate at end of Week 5:** All ten Goals §5 Phase 0 exit criteria met. If any are unmet, Phase 0 extends; do not move to Phase 1 with gaps.

**Parallelization note:** While doing Week 2 Mac work, you can place the Pi 5 order (if not already on hand) and start Apple Developer provisioning (Task 3.1). Both can run in the background.

---

## 3. Week 1 — Repo skeleton and contracts

### Task 1.1: Initialize private repository (public at end of Phase 1)

**What:** Create a **private** GitHub repository with the project's name, Apache 2.0 license, and the three completed documents from `/mnt/user-data/outputs/` (Goals v2, HLD, Prior Art) committed to `docs/`.

**Why:** Goals D0.1, D0.2. The original "public Day 1" rule has been revised — developing privately through Phase 0-1 lets you iterate without audience pressure, and the public reveal at end of Phase 1 lands with a working Mode A loop as the first impression (much stronger than a Week 1 skeleton). This change is documented in ADR-0007 (see Task 4.6).

**How:** GitHub repository, **private** for now, default branch `main`. Choose **Apache 2.0** license — it's more compatible with potential commercial uses than MIT (patent grant) while still permissive enough for community adoption. `docs/` directory contains the three planning documents plus this Detailed Plan as it grows.

The repo *will* go public at end of Phase 1, so write everything as if it were already public — clean commit messages, no sensitive data in history, decisions documented, etc. The privacy is for shielding the unfinished state, not for hiding anything embarrassing.

**Risk and time honesty:** Low risk, half a day at most. The trap is over-engineering — don't waste time on elaborate GitHub Actions, branch policies, or PR templates yet. Just make it exist.

**Done when:** The repo exists (private), has the license file, and contains `docs/Goals.md`, `docs/HLD.md`, `docs/PriorArt.md` rendered correctly.

---

### Task 1.2: Write the README

**What:** A README at the repo root that explains the project to a first-time visitor in under 90 seconds of reading.

**Why:** The README is the project's front door. For Paths 1, 2, and 6 (per Goals §7), this is what people see first. A confusing or absent README kills the discoverability story.

**How:** Structure the README in this order:

1. One-sentence project description ("A multi-agent system that autonomously optimizes vision-language models for on-device deployment").
2. Status (current phase, what works today, what doesn't).
3. Quick-start (clone, install dependencies — even if currently aspirational, sketch what it will look like).
4. Architecture overview (one paragraph + a link to `docs/HLD.md`).
5. Goals overview (one paragraph + a link to `docs/Goals.md`).
6. How to contribute / how to follow along.
7. License.

Be honest about status. "Phase 0 in progress" is fine; "production-ready autonomous optimization system" is not.

**Risk and time honesty:** Half a day. The temptation is to oversell — resist it. Honest READMEs build more trust than ambitious ones, and the project will speak for itself through later phases.

**Done when:** A first-time visitor can read the README in 90 seconds and answer: what is this, what's its current status, where can I learn more.

---

### Task 1.3: Define JSON schemas for all contracts

**What:** Write JSON schemas for `DeviceDescriptor`, `ExperimentConfig`, `MetricsReport`, `AgentDecision`, `HypothesisRecord`, plus the agent-to-service contracts described in HLD §6 (Search Strategist Agent input/output, Research Analyst Agent input/output).

**Why:** Goals exit criterion 0.7. HLD §6 commits to schema-validated contracts as a core design principle. Defining these in Phase 0 — before any agent code exists — means Phase 1 builds against locked contracts, not moving targets.

**How:** Use [JSON Schema Draft 2020-12](https://json-schema.org/) format. Put schemas in `schemas/` directory, one file per schema, version-tagged via `$id`. Use [`pydantic`](https://docs.pydantic.dev/) v2 for Python-side validation — it can generate JSON Schema from Pydantic models, giving you typed Python classes for free.

Schema content per HLD §6 and §6.2:

- `DeviceDescriptor.json`: chip family, RAM, accelerators, supported runtimes, preferred runtime, quirks (free-form text), measurement harness name.
- `ExperimentConfig.json`: model identifier, compression spec (precision, group size, KV-cache precision), input resolution, vision-token budget, runtime backend, decode strategy, dataset hash, target device ID.
- `MetricsReport.json`: experiment ID, device ID, per-metric values (TTFT ms, decode tokens/sec, peak memory MB, on-disk size MB, energy if measured, quality scores per benchmark), timestamps, hardware fingerprint.
- `AgentDecision.json`: input hash, output hash, rationale (free-form text), confidence flags (per-field where applicable), timestamp.
- `HypothesisRecord.json`: per HLD §6.2 fields — title, source citation, claimed effect, verbatim excerpts (array of {text, page/line ref}), original hyperparameters, reported results, applicability check (structured: requirements + match-against-our-setup), known failure modes, implementation difficulty (enum: config-change | minor-code-change | new-module | major-refactor), proposed codebase insertion point, confidence flags (per field, enum: low | medium | high).

**Risk and time honesty:** 2 days. The trap is bikeshedding on field names. Spend time on the *structure* (what's required, what's optional, how nested fields relate) and accept that field names can be renamed in a non-breaking way later via schema versioning.

**Done when:** All eight schemas exist, validate as proper JSON Schema, have Pydantic equivalents, and have at least one example file each demonstrating realistic content.

---

### Task 1.4: Write three DeviceDescriptors

**What:** YAML files describing all devices used by the project: iPhone 16 Pro, Raspberry Pi 5 (4 GB), Mac mini (M4, 16 GB, available now), and the M5 Pro 32 GB (planned, not yet available).

**Why:** Goals exit criterion 0.7 (DeviceDescriptor schema), HLD §2.4 (devices as parameters). The act of writing real descriptors validates that the schema is workable. Both Mac descriptors are written now so the MetricsReport schema is exercised and ready; measurements on the M5 Pro simply won't be logged until the machine is in hand.

**How:** Create `configs/devices/iphone_16_pro.yaml`, `configs/devices/raspberry_pi_5_4gb.yaml`, `configs/devices/mac_mini_m4_16gb.yaml`, and `configs/devices/compute_mac_m5pro_32gb.yaml`.

For iPhone 16 Pro:
```yaml
device_id: iphone_16_pro
chip_family: apple_a18_pro
ram_gb: 8
accelerators: [ane, gpu]
supported_runtimes: [mlx, coreml]
preferred_runtime: mlx
quirks:
  - "MLX op coverage incomplete for some quantization patterns; CoreML fallback may be needed"
  - "ANE compilation can fail silently; verify with Instruments profiler"
  - "Thermal throttling kicks in after ~5 minutes of sustained inference; measure with cooling pauses"
measurement_harness: ios_swift_bridge_v1
```

For Pi 5 4 GB:
```yaml
device_id: raspberry_pi_5_4gb
chip_family: broadcom_bcm2712
ram_gb: 4
accelerators: [cpu]
supported_runtimes: [llamacpp_gguf, onnx_cpu]
preferred_runtime: llamacpp_gguf
quirks:
  - "Only ~3.5 GB usable RAM after OS; tight for VLM inference"
  - "No GPU/NPU acceleration; CPU-only via ARM NEON SIMD"
  - "Active cooling strongly recommended; thermal throttling on default cooling within 2 minutes of sustained load"
  - "Memory pressure can push to swap, which makes latency measurements meaningless; measure free memory before each run"
measurement_harness: pi_python_bridge_v1
```

For Mac mini (M4, 16 GB) — available now, used for all initial Mac measurements:
```yaml
device_id: mac_mini_m4_16gb
chip_family: apple_m4
ram_gb: 16
accelerators: [gpu]   # MPS — Mac doesn't expose ANE for general use
supported_runtimes: [mlx, pytorch_mps, onnx_cpu]
preferred_runtime: mlx
role: measurement_and_training   # Not a deployment target — distinct from iPhone/Pi
quirks:
  - "MPS bandwidth ~120 GB/s (M4 base chip); small-VLM inference is bandwidth-bound at this scale"
  - "Memory shared between CPU and GPU (unified); 16 GB sets the practical model-size ceiling at ~12 GB loaded"
  - "Activity Monitor's memory reporting differs from psutil's; document which you use"
  - "Mac mini is the measurement workstation for evaluating other devices; it is also the iOS dev host"
measurement_harness: mac_python_bridge_v1
```

For M5 Pro 32 GB — not yet available; descriptor written now, measurements logged when machine is in hand:
```yaml
device_id: compute_mac_m5pro_32gb
chip_family: apple_m5_pro
ram_gb: 32
accelerators: [gpu]   # MPS — Mac doesn't expose ANE for general use
supported_runtimes: [mlx, pytorch_mps, onnx_cpu]
preferred_runtime: mlx
role: measurement_and_training
quirks:
  - "MPS bandwidth ~275-300 GB/s; small-VLM training is bandwidth-bound at this scale"
  - "Memory shared between CPU and GPU (unified); training + LLM hosting compete if both on same Mac"
  - "Activity Monitor's memory reporting differs from psutil's; document which you use"
  - "Mac is the measurement workstation for evaluating other devices; it is also the iOS dev host"
  - "Not yet available; measurements pending — mac_mini_m4_16gb used in the interim"
measurement_harness: mac_python_bridge_v1
```

**Risk and time honesty:** Half a day. Easy task. The value is in writing realistic quirks — these are the operational reality of the device and need to be in the descriptor so the Deployment Dispatcher and measurement harness can respect them. The Mac descriptors introduce `role`, which distinguishes deployment targets from measurement/training workstations. Writing both Mac descriptors now means no schema work is needed when the M5 Pro arrives — only measurement runs.

**Done when:** All three descriptors exist, validate against the DeviceDescriptor schema, and the quirks fields contain honest device-specific notes (not generic boilerplate).

---

### Task 1.5: Repository structure scaffolding

**What:** Create the directory structure the Detailed Plan and HLD reference. Empty directories with placeholder README files are fine.

**Why:** Locks the layout so Phase 1 code goes in the right places.

**How:** Directory tree:

```
repo-root/
├── README.md
├── LICENSE
├── pyproject.toml          # Python project metadata, dependencies
├── .gitignore
├── docs/
│   ├── Goals.md
│   ├── HLD.md
│   ├── PriorArt.md
│   ├── DetailedPlan_Phase0.md   # this document
│   └── decisions/          # ADR-style decision logs, one per non-trivial choice
├── schemas/                # JSON Schema files
├── configs/
│   ├── devices/           # DeviceDescriptors
│   ├── experiments/       # ExperimentConfig instances
│   └── policies/          # Threshold Monitor policies, etc.
├── agents/                 # Phase 1+ agent code
├── services/              # Deterministic services (Experiment Runner, Pareto Tracker, etc.)
├── runners/               # Per-device experiment runners
├── datasets/              # Eval set, hash-pinned
├── artifacts/             # Result bundles, models, builds (gitignored, kept locally)
├── eval/                  # Evaluation harness wrappers
├── ios_harness/           # Swift code for iPhone measurement
├── pi_harness/            # Python code for Pi measurement
└── tools/                 # Utility scripts, one-off tools
```

Use `pyproject.toml` for Python (PEP 621 standard). Specify Python 3.11+. Lock initial dependencies: `pydantic>=2`, `pyyaml`, `numpy`, `torch`, `transformers`, `vlmevalkit` (pinned version), `pytest` for tests.

**Risk and time honesty:** Half a day. The trap is endless planning of the structure. The above is good enough; commit it and move on.

**Done when:** The directory tree exists, `pyproject.toml` is valid (try `pip install -e .` and confirm it works), and each empty directory has a `README.md` placeholder explaining what goes there.

---

**Week 1 exit check.** End of Week 1, you should have:

- Private repo (public at end of Phase 1), Apache 2.0 licensed, with three planning docs and this plan in `docs/`.
- README written honestly.
- 8 JSON schemas + 8 Pydantic equivalents in `schemas/`.
- 2 DeviceDescriptors in `configs/devices/`.
- Full repo directory structure scaffolded.

If you're not here by end of Week 1, do not paper over it. Honestly assess what slowed you down and adjust Week 2 expectations.

---

## 4. Week 2 — Mac-only baselines and VLMEvalKit

### Task 2.1: Qwen2.5-VL-3B on Mac mini (initial), M5 Pro 32 GB (when available)

**What:** Run Qwen2.5-VL-3B on the Mac mini (M4, 16 GB) now. When the M5 Pro 32 GB becomes available, repeat the same measurement run and log a second `MetricsReport`. Document non-fit on Pi 5 4 GB as a confirmed expectation.

**Why:** Goals exit criterion 0.4. Qwen2.5-VL-3B is the Phase 2 starting point — the "general-purpose, not-edge-optimized" model the system must compress down to a 450M-class edge model. Measuring it now establishes the unoptimized "before" picture against which Phase 2's "after" will be compared. The M5 Pro measurements matter because training in Phase 2 will run on the M5 Pro; having both baselines makes the hardware difference visible and keeps the comparison honest.

**How:**

1. Pull Qwen2.5-VL-3B-Instruct from Hugging Face (`Qwen/Qwen2.5-VL-3B-Instruct`).
2. Run via PyTorch + Transformers (MPS backend). Loading at FP16 should fit in ~7 GB; the 16 GB unified memory of the Mac mini gives comfortable headroom for activations and KV cache.
3. Choose 5 sample photos from a public source (Flickr30k, COCO, or Open Images — these will become part of the Stage A eval set in Week 4). Run inference on them; capture latency (per-token decode speed), peak memory (via `psutil.Process().memory_info()`), on-disk size.
4. Log to `MetricsReport` with `device_id: mac_mini_m4_16gb`. When the M5 Pro is available, repeat identically and log a second `MetricsReport` with `device_id: compute_mac_m5pro_32gb`.
5. Confirm non-fit on Pi 5 by *not even attempting*. Document the math: 3B FP16 ≈ 6 GB weights + activations + KV cache > Pi 5 4 GB. This is expected and confirms the size-reduction problem the system needs to solve.

**Risk and time honesty:** 1-2 days for the Mac mini run. The M4's ~120 GB/s memory bandwidth (vs. M5 Pro's ~275 GB/s) means decode throughput will be lower; this is expected and noted. We're measuring the unoptimized starting point on purpose. Quality scores on benchmark slices come in Task 2.2 via VLMEvalKit; Stage A eval set quality scores come in Week 4.

Document the measurement methodology in `docs/decisions/0001-mac-measurement-methodology.md` (first ADR — sets the pattern for the iPhone and Pi methodology ADRs that follow). The same methodology doc covers both Mac devices. **Critically, this methodology is the foundation for all downstream measurement work**; getting it right on the Mac mini in Week 2 means the Week 3 iPhone work and Week 4 Pi work inherit a working measurement pattern.

**Done when:** Qwen2.5-VL-3B has measured baseline numbers on Mac mini, logged as `MetricsReport`. Pi non-fit is documented. (M5 Pro `MetricsReport` is logged when that machine is available — it does not block Phase 0 completion.)

---

### Task 2.2: VLMEvalKit integration

**What:** A wrapper that runs VLMEvalKit on a model + benchmark slice, captures the results, and writes them in our `MetricsReport` format. Run it for all reference models plus Qwen2.5-VL-3B.

**Why:** Goals exit criterion 0.6. VLMEvalKit is the harness Liquid uses; running our reference models through it is what makes the Phase 2 success criterion measurable on apples-to-apples benchmarks.

**How:** Clone [`open-compass/VLMEvalKit`](https://github.com/open-compass/VLMEvalKit). It supports LFM2.5-VL, SmolVLM, MiniCPM-V, Qwen2.5-VL, and FastVLM as model adapters (verify each; write a thin adapter if not).

Steps:

1. Install VLMEvalKit on the Mac mini.
2. Configure evaluations on small slices of RealWorldQA (100 examples), MMBench dev-en (100 examples), POPE (100 examples). Use slices rather than full benchmarks to keep runtime reasonable on Mac.
3. Run sequentially: LFM2.5-VL-450M, SmolVLM-500M, MiniCPM-V 4.6, FastVLM-0.5B, Qwen2.5-VL-3B (each adds ~30min–2hr of Mac runtime).
4. Parse VLMEvalKit's output into our `MetricsReport` format. Quality metrics (accuracy, F1, etc.) join the same metrics database as latency/memory.

**Risk and time honesty:** 2-3 days. VLMEvalKit is well-maintained but has a steep learning curve — its config system is opinionated. Plan a day of "just figuring out how it wants to be configured" before the actual eval runs. Five models × three benchmark slices = 15 eval runs; some on the Mac mini will take hours each.

Note: Quality evaluation runs on the Mac, not on the iPhone or Pi. Quality is device-independent for a given model; latency/memory metrics on real devices are measured in Weeks 3 (iPhone) and 4 (Pi).

**Done when:** All five models have measured quality scores on RealWorldQA, MMBench dev-en, and POPE slices, stored in `MetricsReport` format.

---

**Week 2 exit check.** End of Week 2:

- Qwen2.5-VL-3B baseline on Mac mini (the "unoptimized starting point" reference).
- VLMEvalKit integrated and running.
- Quality metrics for all five reference models (LFM2.5-VL-450M, SmolVLM-500M, MiniCPM-V 4.6, FastVLM-0.5B, Qwen2.5-VL-3B) on three benchmark slices.
- One ADR written (Mac measurement methodology).

---

## 5. Week 3 — iPhone reference baselines

This is the riskiest week. Apple developer provisioning and iOS deployment of ML models has destroyed timelines for many projects. Be especially honest about progress this week.

### Task 3.1: Apple Developer account + provisioning setup

**What:** Get to a state where you can deploy a custom app to your iPhone 16 Pro from Xcode.

**Why:** Goals exit criteria 0.1 and 0.2 both require running models on a physical iPhone. Without working provisioning, those criteria cannot be met.

**How:** If you don't have an Apple Developer account ($99/year), create one — the free provisioning option works for personal devices but with weekly re-signing, which is painful for sustained development. Pay the $99 and save yourself ongoing friction.

Install Xcode 15+ (or whatever version supports iOS deployment to your iPhone's iOS version). Connect the iPhone via USB, register it as a development device, create a provisioning profile.

Smoke test: deploy a blank SwiftUI app to the iPhone from Xcode. If this works, you're unblocked for the rest of Week 3.

**Risk and time honesty:** This is the highest-risk task in Phase 0. If you've never done iOS development, plan for 1-2 days of friction (Xcode version mismatches, provisioning profile errors, iOS version requirements). If you're already set up, this is half a day.

The honest "stuck" scenario: you spend a week on this and still can't deploy. If that happens, the project either needs to move iPhone testing to TestFlight (more complex but more reliable) or partner with someone who has working iOS dev infrastructure. **Don't spend more than 3 days on provisioning before considering alternatives.** Note: if you started provisioning early during Week 2's Mac work (as recommended in the at-a-glance parallelization note), this risk is significantly reduced.

**Done when:** A blank app deploys from Xcode to your iPhone 16 Pro and runs.

---

### Task 3.2: LFM2.5-VL-450M on iPhone via LEAP

**What:** Get LFM2.5-VL-450M running on iPhone 16 Pro, producing captions and VQA answers for sample images, with measured TTFT, decode tokens/sec, peak memory, and on-disk size.

**Why:** Goals exit criterion 0.1.

**How:** Liquid AI ships the [LEAP SDK](https://www.liquid.ai/leap) for iOS with Swift bindings designed for "few lines of code" integration. They also have a sample iOS app called Apollo that does roughly this — clone it as a starting point if available.

Approach:

1. Pull the LFM2.5-VL-450M GGUF weights from Hugging Face (Liquid publishes these under their LFM Open License v1.0).
2. Use the LEAP SDK example as a starting point. Modify it to load LFM2.5-VL-450M and run inference on a hardcoded test image with a hardcoded prompt.
3. Instrument the Swift code to measure:
   - TTFT: time from prompt submission to first token emitted.
   - Decode tokens/sec: tokens emitted in the steady state after the first.
   - Peak memory: use `os_proc_available_memory()` or `mach_task_basic_info` to sample memory before/during/after inference.
   - On-disk size: just the size of the GGUF file plus any associated files.

4. Log results to a JSON file that conforms to the `MetricsReport` schema from Task 1.3.

5. Repeat on 5-10 sample images to verify the numbers are stable (within ~10% across runs after a warmup).

**Risk and time honesty:** 2-3 days. Likely time sinks:
- The LEAP SDK might be at a different version than the docs assume; expect to dig through Liquid's GitHub.
- Memory measurement on iOS is genuinely tricky (jetsam, ARC, etc.) — the numbers you get will be approximations.
- The first run after install is slow (model load time); the steady-state numbers are what you want, not the cold-start ones.

Document your methodology in `docs/decisions/0002-ios-measurement-methodology.md`.

**Done when:** LFM2.5-VL-450M runs on the iPhone, produces output for at least 5 sample photos, and a `MetricsReport` JSON exists with stable TTFT, decode speed, peak memory, and on-disk size.

---

### Task 3.3: FastVLM-0.5B on iPhone via apple/ml-fastvlm

**What:** Same as Task 3.2 but for FastVLM-0.5B using Apple's published demo app.

**Why:** Goals exit criterion 0.2.

**How:** Clone [`apple/ml-fastvlm`](https://github.com/apple/ml-fastvlm) from GitHub. The repo includes an `app/` directory with a SwiftUI iOS demo. Apple publishes pre-exported MLX builds for FastVLM-0.5B at FP16 / INT8 / INT4.

Steps:

1. Clone the repo, follow their iOS app build instructions.
2. Deploy to your iPhone 16 Pro.
3. Run inference on the same 5-10 sample photos used in Task 3.2.
4. Instrument the same way as Task 3.2: TTFT, decode tokens/sec, peak memory, on-disk size.
5. Log to a `MetricsReport` JSON.

**Risk and time honesty:** 1-2 days, assuming Task 3.1 worked. The FastVLM demo app is well-documented and Apple supports it. The main risk is the demo app's UI being different from what you need for measurement — you may need to modify it to log timings cleanly.

**Done when:** FastVLM-0.5B runs on the iPhone, produces output for the same 5 sample photos, and a `MetricsReport` JSON exists.

---

### Task 3.4: SmolVLM-500M and MiniCPM-V 4.6 on iPhone

**What:** Get SmolVLM-500M and MiniCPM-V 4.6 running on iPhone 16 Pro, with measured metrics for each.

**Why:** Goals exit criterion 0.3. Expanding the reference baseline to four models gives a stronger comparison for Phase 2's success claim ("competitive with at least two of the four references"). Both are widely-used open-source edge VLMs in the target size class.

**How:**

For **SmolVLM-500M**:
1. Pull weights from Hugging Face (`HuggingFaceTB/SmolVLM-Instruct` or similar — verify the latest small variant).
2. SmolVLM is published with mobile demo code; check `HuggingFaceTB/smollm` and SmolVLM's GitHub for iOS examples. If none exist, MLX-LM or llama.cpp via a Swift wrapper are workable paths.
3. Apache 2.0 license — cleanest of all the candidates, no constraints.
4. Run inference on the same sample photos used in Tasks 2.2/2.3.
5. Same measurement instrumentation: TTFT, decode tokens/sec, peak memory, on-disk size.

For **MiniCPM-V 4.6**:
1. Pull the GGUF weights from `openbmb/MiniCPM-V-4.6-gguf` (or the latest official OpenBMB release).
2. OpenBMB publishes iOS adaptation code under the MiniCPM-V repo's `demo/` directory — use that as the starting point.
3. Same measurement methodology.

**Important note for both:** MiniCPM-V 4.6 at 1.3B params will be tight on iPhone 16 Pro's 8 GB unified memory but should run. SmolVLM-500M will run comfortably. If either fails to fit, document the failure mode (out-of-memory, slow loading, etc.) — this is informative for Phase 2's design.

**Risk and time honesty:** 2-3 days for both. SmolVLM is easier (smaller, more open) — likely 1 day. MiniCPM-V 4.6 might take 1-2 days because the on-iOS deployment path is less polished than LFM's LEAP SDK or Apple's FastVLM demo. The cumulative result is the broadest possible iPhone baseline comparison.

**Done when:** Both models run on the iPhone, produce output for the same 5 sample photos, and `MetricsReport` JSONs exist for each.

---

### Task 3.5: Sanity-check the iPhone numbers against published claims

**What:** Compare measured numbers against published claims for all four iPhone reference models.

**Why:** If measured numbers are wildly different from published claims, the measurement methodology is broken and Phase 1's optimization claims will be too. Catch this now, not in Phase 2.

**How:** Make a table in `docs/decisions/0003-iphone-baseline-numbers.md`:

| Model | TTFT measured | TTFT published claim | Peak memory measured | On-disk size measured | Notes |
|---|---|---|---|---|---|
| LFM2.5-VL-450M | ... | n/a (Jetson published) | ... | ... | |
| FastVLM-0.5B | ... | <120 ms (iPhone 16 Pro) | ... | ~1 GB | |
| SmolVLM-500M | ... | n/a | ... | ... | |
| MiniCPM-V 4.6 | ... | n/a (OpenBMB ships demo videos) | ... | ... | |

If FastVLM-0.5B TTFT is 200ms when Apple claims <120ms, something is off — wrong precision, wrong runtime, cold-start contamination, thermal throttling, or measurement methodology error. Investigate before proceeding.

**Risk and time honesty:** Half a day. The trap is hand-waving discrepancies. If your TTFT is 2× the published claim, dig in.

**Done when:** Either your numbers match published claims within ~20%, or you have a documented explanation for why they don't.

---

**Week 3 exit check.** End of Week 3:

- Provisioning works, you can deploy to iPhone freely.
- LFM2.5-VL-450M baseline numbers measured and logged.
- FastVLM-0.5B baseline numbers measured and logged.
- SmolVLM-500M baseline numbers measured and logged.
- MiniCPM-V 4.6 baseline numbers measured and logged (or non-fit documented).
- Numbers sanity-checked against published claims for all four.
- Two ADRs written (iOS measurement methodology, iPhone baseline numbers).

If iPhone provisioning blocked you for >3 days, escalate the decision: continue trying, or pivot to a different testing strategy. Do not silently slip. If MiniCPM-V 4.6 doesn't fit on iPhone, document that and move on — it's informative, not a failure.

---

## 6. Week 4 — Pi 5 baseline and evaluation set

### Task 4.1: Pi 5 hardware setup and OS

**What:** A Raspberry Pi 5 (4 GB) with Raspberry Pi OS 64-bit (Bookworm-based or newer), active cooling, networked, and SSH-accessible from your Macs.

**Why:** Prerequisite for Goals exit criteria 0.1 (LFM on Pi) and 0.3 (SmolVLM/MiniCPM-V on Pi).

**How:** Standard Pi setup. The active cooling part is non-optional given the 4 GB Pi's tendency to thermal-throttle under sustained inference (per the DeviceDescriptor quirks in Task 1.4). The official Active Cooler for Pi 5 (~$5) is sufficient.

OS: Raspberry Pi OS 64-bit. Some llama.cpp optimizations require 64-bit; do not use the 32-bit version.

Networking: connect to the same Wi-Fi network as the Macs. Set up SSH key-based access so you can run commands from the Mac mini without typing passwords.

Install dependencies on the Pi:
- Build tools: `apt install build-essential cmake git python3-pip`
- llama.cpp: clone from `ggml-org/llama.cpp`, build with `cmake -B build` then `cmake --build build --config Release -j`. Compile with `-DLLAMA_NATIVE=ON` for ARM NEON SIMD optimization.

**Risk and time honesty:** 1 day. If you haven't used a Pi before, slightly more. The main time sink is dependency installation (apt updates can be slow).

**Done when:** SSH from Mac to Pi works without password, llama.cpp builds and runs on the Pi (try `./build/bin/llama-cli --help` to verify), and `vcgencmd measure_temp` shows the Pi is not overheating under idle load.

---

### Task 4.2: LFM2.5-VL-450M on Pi 5 via llama.cpp

**What:** Get LFM2.5-VL-450M running on Pi 5 via llama.cpp, producing captions and VQA answers, with measured TTFT, decode tokens/sec, peak memory, on-disk size.

**Why:** Goals exit criterion 0.1 (LFM2.5-VL-450M on both devices).

**How:** llama.cpp supports vision-language models via the `mtmd` (multimodal) interface. LFM2.5-VL has community GGUF builds; check Liquid's Hugging Face org and `LiquidAI/lfm2-vl` repos for the latest.

Steps:

1. Download the LFM2.5-VL-450M GGUF (Q4_0 quantization to match the Liquid published reference) plus the vision encoder weights (usually a separate `mmproj` file).
2. Run inference from the Pi shell:
   ```bash
   ./build/bin/llama-mtmd-cli \
     -m models/lfm2-vl-450m-q4_0.gguf \
     --mmproj models/lfm2-vl-450m-mmproj.gguf \
     --image test_photo.jpg \
     -p "Describe this image." \
     -n 128
   ```
3. Instrument timing: llama.cpp prints timing summaries at the end of each run, including prompt-eval time (related to TTFT) and decode tokens/sec. Capture these via stdout parsing or use the `--log-file` option.
4. Memory: monitor with `free -h` before/during/after, or use `/proc/<pid>/status` for the process specifically. Watch for swap usage — if any swap is hit, the latency numbers are invalid and the run should be flagged.
5. On-disk size: just file size of model + mmproj.

Important: **on 4 GB Pi, you must verify free memory before each run.** If <2 GB free, the run will likely page to swap and the numbers will be garbage. Add a pre-flight check to the measurement script.

**Risk and time honesty:** 2-3 days. Likely time sinks:
- Finding the right GGUF: Liquid's official GGUFs may or may not have the vision projector packaged correctly; community GGUFs may have inconsistent naming. Allow time to verify you have a working set.
- The Pi is slow. A single inference run takes seconds to tens of seconds. Iterating on measurement methodology is much slower than on Mac.
- Swap-related contamination of latency numbers is a real risk on 4 GB. Be paranoid about pre-flight memory checks.

Document methodology in `docs/decisions/0004-pi-measurement-methodology.md`.

**Done when:** LFM2.5-VL-450M runs on the Pi, produces output for 5 sample photos, `MetricsReport` JSON exists, and the measurement was confirmed not to hit swap.

---

### Task 4.3: SmolVLM-500M and MiniCPM-V 4.6 on Pi 5

**What:** Get SmolVLM-500M running on Pi 5, and attempt MiniCPM-V 4.6 (which will likely not fit at 1.3B params on 4 GB Pi). Measure both, or document non-fit.

**Why:** Goals exit criterion 0.3 (Pi reference baselines). SmolVLM-500M is the smallest and most-likely-to-fit competitor; MiniCPM-V 4.6's likely non-fit on Pi 5 4 GB is itself informative data about what Pi 5 4 GB can actually run.

**How:**

For **SmolVLM-500M**:
1. Hugging Face has SmolVLM GGUF builds (or convert with llama.cpp's `convert-hf-to-gguf.py` if needed). Use Q4_0 quantization to match LFM's reference quantization.
2. Run via `llama-mtmd-cli` on the Pi (same harness as Task 4.2).
3. Same pre-flight memory check, same instrumentation, same `MetricsReport` output.

For **MiniCPM-V 4.6**:
1. Attempt the same pipeline. At 1.3B params + vision encoder + activations, this almost certainly exceeds 4 GB. Expected outcome: out-of-memory or unusable swap thrashing.
2. If it doesn't fit, document the failure mode (OOM at load? OOM mid-inference? Swap-thrashing slow?) in `docs/decisions/0005-pi-model-fit-summary.md`. This documentation is the deliverable.
3. If it does fit (surprise outcome — would mean very aggressive quantization works on Pi), measure and log.

**Risk and time honesty:** 1-2 days for both. SmolVLM should be straightforward (it's small and well-supported). MiniCPM-V is mostly a confirm-and-document exercise.

**Done when:** SmolVLM-500M `MetricsReport` exists on Pi. MiniCPM-V 4.6 either has a `MetricsReport` (if it fits) or a documented non-fit reason in the ADR.

---

### Task 4.4: Verify FastVLM on Pi 5 is not viable

**What:** Confirm and document that FastVLM-0.5B does not run usefully on Pi 5 4 GB.

**Why:** Goals exit criterion 0.2 and the HLD framing rests on "FastVLM is iPhone-bound, our system targets both devices" — but you should verify this is actually true rather than asserted.

**How:** Try to run FastVLM-0.5B on the Pi via the most plausible path (likely ONNX export of the model + onnxruntime CPU EP on the Pi). One of:

1. FastVLM weights exist as ONNX at Hugging Face (verify) — if so, attempt `onnxruntime` inference on the Pi.
2. If no ONNX export exists, try converting from the HF Transformers checkpoint locally to ONNX using `optimum`.

What "not viable" likely looks like:
- Model is too large for 4 GB Pi (FastVLM-0.5B at FP16 is ~1 GB; with activations and KV cache during inference, it pushes over 4 GB).
- Inference takes minutes per query instead of seconds (unacceptably slow).
- Quality is somehow degraded by the export path.

Whatever you find, document it. A documented "FastVLM does not run usefully on Pi 5 4 GB because [specific reason]" is more useful than a vague "FastVLM is Apple-Silicon-only" claim.

**Risk and time honesty:** 1 day. The trap is sinking time into making this work. The point is to confirm it doesn't, not to optimize it.

**Done when:** A documented attempt and outcome exists in `docs/decisions/0006-fastvlm-on-pi-not-viable.md`. The result either confirms unviability or surprises you (in which case, document the surprise and update the project's framing).

---

### Task 4.5: Assemble Stage A evaluation set

**What:** A frozen, hash-pinned evaluation set of 200 photos, 100 captions, and 100 VQA pairs drawn from public sources.

**Why:** Goals exit criterion 0.5. Phase 1+ optimization is judged against this set. If the set is wrong, everything downstream is wrong.

**How:** Source photos from:
- Flickr30k (free, ~31k photos with 5 captions each)
- COCO Captions (free, hundreds of thousands of photos with multiple captions each)
- Open Images V7 (free, photos with various labels)

Select 200 photos that are:
- Personal-photo-like (not stock photography, not memes, not artistic renders)
- Diverse: indoor/outdoor, single subject/group, daytime/night, various activities, various ages of people
- Free of identifiable real people when possible (privacy)
- Captioned in the source dataset

For each photo:
1. Copy to `datasets/stage_a_proxy/photos/<id>.jpg`.
2. Hash-pin via SHA-256, record in `datasets/stage_a_proxy/manifest.json`.

For 100 captions:
1. Take 100 of the 200 photos.
2. Use the source dataset's caption(s) if they're descriptive enough.
3. Write or supplement captions for those that aren't (this is manual work — budget 2-3 hours).
4. Record in `datasets/stage_a_proxy/captions.json` with photo_id → caption mapping.

For 100 VQA pairs:
1. Take a different 100 of the 200 photos.
2. Write one or two questions per photo covering scene, activity, count, time, mood.
3. Write the expected answer.
4. Record in `datasets/stage_a_proxy/vqa.json`.

Hash-pin the entire eval set as a single SHA-256 of the manifest. This hash goes into every `ExperimentConfig` that uses the eval set, per HLD §6.4 (golden regression).

**Risk and time honesty:** 2-3 days. The honest time sink is writing 100 VQA pairs — it's monotonous work, harder than it looks. Don't crowdsource it; the quality is too important.

A note on dataset licensing: Flickr30k, COCO, Open Images have research-use licenses. You can use them for this project. Whether you can *redistribute* the curated subset depends on the licenses. For the public release (D0.5), redistribute only the *list of photo IDs* and your own captions/VQA pairs; users can fetch the actual photos themselves. Document this in the eval set README.

**Done when:** Eval set exists with 200 photos, 100 captions, 100 VQA pairs, all hash-pinned. The eval set's manifest hash is recorded somewhere central.

---

**Week 4 exit check.** End of Week 4:

- Pi 5 4 GB set up, llama.cpp built, SSH-accessible.
- LFM2.5-VL-450M baseline numbers measured on Pi.
- SmolVLM-500M baseline numbers measured on Pi.
- MiniCPM-V 4.6 either measured on Pi (if it fits) or non-fit documented.
- FastVLM-on-Pi unviability documented.
- Stage A evaluation set assembled, hash-pinned, ready to use.
- Four ADRs written (Pi methodology, Pi model-fit summary, FastVLM-Pi, eval set composition).

---

## 7. Week 5 — Dashboard, spike, license posture, Phase 0 close

---

### Task 5.1: Metrics database and dashboard

**What:** A SQLite database that stores all `MetricsReport` records, plus a minimal Streamlit dashboard that renders per-device Pareto frontier plots with all reference markers visible.

**Why:** Goals exit criterion 0.8. HLD §6 commits to SQLite-backed metrics. The dashboard is the human-facing surface — anyone who clones the repo (after Phase 1 reveal) and runs `streamlit run dashboard.py` can see the project's current state.

**How:**

Database schema (SQLite):
- `experiments(id, config_hash, device_id, timestamp, status, notes)`
- `metrics(experiment_id, metric_name, metric_value, unit)`
- `agent_decisions(id, agent_name, input_hash, output_hash, rationale, timestamp)` — empty in Phase 0, fills in Phase 1+

Use `sqlite3` directly from Python; no ORM needed for this scale.

Dashboard (Streamlit) with four tabs:
1. **iPhone Pareto:** scatter plot of (TTFT, on-disk size) with LFM2.5-VL-450M, FastVLM-0.5B, SmolVLM-500M, and MiniCPM-V 4.6 as labeled markers.
2. **Pi 5 Pareto:** scatter plot of (decode tokens/sec, peak memory) with LFM2.5-VL-450M, SmolVLM-500M, and (if it fit) MiniCPM-V 4.6 as labeled markers. Note that FastVLM is documented as non-viable on Pi.
3. **Mac starting point:** Qwen2.5-VL-3B's measured baseline, clearly labeled "not edge-viable — this is what Phase 2 must optimize *from*."
4. **Quality summary:** table of RealWorldQA / MMBench / POPE scores per reference model.

Keep the dashboard ugly but functional. Streamlit's defaults are fine. The point is "the data is visible," not "the dashboard is beautiful."

**Risk and time honesty:** 2 days. The trap is over-designing the dashboard. Make it work, move on.

**Done when:** Running `streamlit run dashboard.py` produces a working dashboard with four tabs and the Phase 0 baseline data populated.

---

### Task 5.2: Literature-ingestion tooling spike

**What:** A 1-day evaluation of PaperQA2, OpenScholar, and AI-Scientist v2 as potential components for the future Research Analyst Agent (Phase 3). Decide build-vs-adopt for each. Document the decision.

**Why:** Goals exit criterion 0.9. Prior art survey suggested this could save weeks of Phase 3 work, but only if the tools actually fit.

**How:** Time-box this to one day. Approach:

1. **PaperQA2** (FutureHouse): clone repo, run their example on a sample arXiv paper, evaluate output quality and whether the API matches what we'd need from the Research Analyst Agent.
2. **OpenScholar** (Ai2): same — their pipeline is more research-summary-focused; check if their structured extraction matches our `HypothesisRecord` format.
3. **AI-Scientist v2** (Sakana): clone, read the agentic tree search code, assess whether it could be adapted to our use case or is too coupled to paper-writing.

For each, write a short evaluation in `docs/decisions/0009-literature-tool-eval.md` with:
- What it does well
- What it doesn't do (relative to our `HypothesisRecord` needs)
- Adopt as-is / fork / take inspiration / ignore
- Estimated effort to integrate vs. build from scratch

**Risk and time honesty:** 1 day if you time-box ruthlessly. The trap is going down a rabbit hole on any one tool. The output is a decision, not a deep evaluation.

**Done when:** ADR exists with build-vs-adopt decisions for all three tools.

---

### Task 5.3: Phase 0 blog post draft (private — published with Phase 1 reveal)

**What:** A draft blog post explaining the project, kept in the repo at `docs/blog/draft_phase_0.md`. Will be merged into the Phase 1 reveal post when the repo goes public, or published as a separate post at that time.

**Why:** Goals D0.3. The post itself moves to end of Phase 1 (per the revised public-reveal timing), but writing it during Phase 0 captures the framing while it's fresh. End-of-Phase-1-you will be busy with the Mode A loop; Phase-0-you has time to write carefully.

**How:** Suggested structure (same as the original plan, ~1500 words):

1. **Hook** (~150 words): the problem. "Edge VLM deployment requires manual orchestration across many decisions. Teams at Apple and Liquid AI take 6-18 months to produce one good edge VLM. What if a system could do the orchestration?"

2. **What I'm building** (~300 words): brief description of the autonomous agent system, the Mode A / Mode B framing, the two target devices, the validation bar (LFM2.5-VL-450M).

3. **Why this is hard** (~400 words): cite Beel et al.'s evaluation of AI-Scientist. Explain why "fully autonomous research" doesn't work yet, and why the design choices in this project (LLM-as-analyst, hypothesis records as implementation kits, Decision-Dossier-gated escalation) specifically address those failures.

4. **Where this fits in the landscape** (~300 words): brief positioning relative to AutoML / NAS, LLM-driven AutoML (MONAQ, Trirat et al.), AI-Scientist, and production edge inference (LFM2.5-VL, FastVLM). Link to the Prior Art document for depth.

5. **What I've measured so far** (~200 words): include one screenshot of the Phase 0 Pareto dashboard. State the reference numbers honestly. "Here's where the bar is."

6. **License posture** (~100 words): brief, honest statement about which licenses govern which artifacts, why distillation will use LFM2-VL-3B not FastVLM-7B, what the project produces and under what terms. (See Task 5.4 for the underlying `THIRD_PARTY.md`.)

7. **What's next** (~150 words): Phase 1 will build the Mode A loop. Link to the repo (placeholder; will be live at public reveal), invite issues.

Be honest. Don't oversell. The post is more valuable if it accurately conveys "interesting project at an early stage" than if it claims more than has been done.

**Risk and time honesty:** 1-2 days of focused writing. The trap is endless rewriting. Three drafts is enough. Since it won't publish for another month or more, save final polish for the public-reveal moment.

**Done when:** Draft exists in the repo, is approximately the right length, and represents the project honestly.

---

### Task 5.4: Write `THIRD_PARTY.md` for license posture

**What:** A clear document listing every third-party model, dataset, library, and tool the project uses, with its license and the project's compliance posture.

**Why:** License hygiene matters now (project owner knows what they're committing to) and will matter more later (anyone reviewing the public repo wants to see this exists). For a project that benchmarks against commercial-license-adjacent models like FastVLM, being explicit about the license posture is non-optional.

**How:** Create `docs/THIRD_PARTY.md` covering at minimum:

- **LFM2.5-VL-450M, LFM2-VL-3B (Liquid AI):** LFM Open License v1.0. Free for commercial use under $10M annual revenue. Used as benchmark baseline and as fine-tuning starting point / distillation teacher. Derivative models will be released under LFM Open License v1.0.
- **FastVLM-0.5B, FastVLM-1.5B, FastVLM-7B (Apple):** Apple Sample Code License. Used as benchmark reference only. No derivative work. Not used as distillation teacher. Measurement code on iPhone is written from scratch, not derived from Apple's demo app code.
- **VLMEvalKit:** Apache 2.0 (verify). Used as evaluation harness.
- **llama.cpp:** MIT (verify). Used as Pi 5 runtime.
- **PyTorch, Transformers, MLX, Optimum, etc.:** Various permissive licenses. Standard ML tooling.
- **Flickr30k, COCO Captions, Open Images:** Research-use licenses. Photo IDs and our own captions/VQA pairs are redistributed; original photos are not — users fetch them from the source datasets.

For each, note: license name, permitted uses, our use, and whether we redistribute or only reference.

In the repo, also keep a `licenses/` subdirectory containing the exact LICENSE files received from each upstream source. This is your audit trail.

**Risk and time honesty:** Half a day. The trap is over-lawyering. You're not writing a legal opinion — you're documenting the project's posture in language a developer-reader can understand. If something is genuinely ambiguous, write "interpretation: conservative reading is X; we will follow that unless we get explicit clarification from the upstream."

**Done when:** `THIRD_PARTY.md` exists in `docs/`, covers all third-party material, and the `licenses/` directory contains archived copies of each upstream LICENSE file.

---

### Task 5.5: ADR-0007 (license posture) and ADR-0008 (public-repo timing)

**What:** Two short Architecture Decision Records.

**ADR-0007 — License posture decision.** Documents *why* the project takes the license posture it does: (a) use LFM2-VL-3B as the distillation teacher rather than FastVLM-7B, (b) treat FastVLM as benchmark-only, (c) release derivative models under inherited upstream licenses (LFM Open License for LFM-derived, Qwen license for Qwen-derived), (d) write iPhone measurement code from scratch rather than fork Apple's demo.

**ADR-0008 — Public repo timing.** Documents the decision to keep the repository private through Phase 0-1 and reveal at end of Phase 1, rather than the originally-considered Day-1-public approach.

**Why:** ADRs capture the *reasoning* behind decisions, not just the decisions themselves. Six months from now, "why did I avoid FastVLM distillation" or "why didn't I make the repo public on Day 1" should not require re-deriving the rationale.

**How:** Create both files with standard ADR structure (Context / Decision / Rationale / Consequences / Open issues):

- `docs/decisions/0007-license-posture.md`
- `docs/decisions/0008-public-repo-timing.md`

**Risk and time honesty:** Half a day for both. Don't over-write — ADRs are deliberately short (1-2 pages).

**Done when:** Both ADRs exist, are concise, accurately capture the reasoning.

(Note: ADR-0009 — literature-tool eval — was created by Task 5.2.)

---

### Task 5.6: Phase 0 retrospective

**What:** A short retrospective document covering what went well in Phase 0, what was harder than expected, and what to adjust for Phase 1.

**Why:** Solo execution benefits enormously from explicit retrospection. The retrospective also becomes a public artifact at end of Phase 1 (when the repo goes public) and builds trust through honesty.

**How:** Half-page document at `docs/retrospectives/phase_0.md`. Five sections:

1. What got done (vs. the plan)
2. What took longer than expected (and why)
3. What didn't get done (and why)
4. What I'd do differently next time
5. Adjustments to the Phase 1 plan based on what I learned

This is *honest*, not promotional. If you spent 5 days on iPhone provisioning, write that. Future-you (and any reader of the public repo) will benefit more from honesty than from a sanitized account.

**Risk and time honesty:** Half a day.

**Done when:** The retrospective exists and is honest.

---

## 8. Phase 0 exit gate

End of Week 5, before declaring Phase 0 complete, verify all ten Goals §5 exit criteria are met:

| Criterion | Status | Verification |
|---|---|---|
| 0.1 LFM2.5-VL-450M on iPhone + Pi 5 | ☐ | Metrics in DB on both devices, reproducible |
| 0.2 FastVLM-0.5B on iPhone; non-viability on Pi 5 documented | ☐ | iPhone metrics in DB; Pi non-fit ADR-0005 |
| 0.3 SmolVLM-500M and MiniCPM-V 4.6 on iPhone and Pi 5 (where they fit) | ☐ | Metrics in DB, non-fits documented |
| 0.4 Qwen2.5-VL-3B on Mac mini; non-fit on Pi 5 documented | ☐ | Mac metrics in DB; Pi non-fit confirmed by math, not attempt |
| 0.5 Frozen Stage A evaluation set | ☐ | Hash-pinned, manifest exists |
| 0.6 Reference models evaluated via VLMEvalKit | ☐ | Quality metrics for all 5 models on 3 benchmark slices |
| 0.7 JSON schemas for all contracts | ☐ | All 8 schemas + Pydantic + `HypothesisRecord` |
| 0.8 Dashboard with reference markers | ☐ | `streamlit run` works, 4 tabs populated |
| 0.9 Build-vs-adopt spike | ☐ | ADR-0009 with decisions |
| 0.10 `THIRD_PARTY.md` documenting license posture | ☐ | Document and `licenses/` directory exist |

Plus, all Phase 0 deliverables (D0.1–D0.6 per Goals §5) and supporting ADRs:

| Deliverable | Status |
|---|---|
| D0.1 Private repo (public at end of Phase 1) | ☐ |
| D0.2 Documents in repo | ☐ |
| D0.3 Reproducible measurement scripts | ☐ |
| D0.4 Frozen eval set with redistribution-safe manifest | ☐ |
| D0.5 JSON schemas tagged and versioned | ☐ |
| D0.6 Blog post draft (publishes with Phase 1 reveal) | ☐ |
| ADR-0002 (iOS measurement methodology) | ☐ |
| ADR-0003 (iPhone baseline numbers) | ☐ |
| ADR-0004 (Pi measurement methodology) | ☐ |
| ADR-0005 (Pi model-fit summary) | ☐ |
| ADR-0006 (FastVLM-on-Pi not viable) | ☐ |
| ADR-0001 (Mac measurement methodology) | ☐ |
| ADR-0007 (license posture) | ☐ |
| ADR-0008 (public-repo timing) | ☐ |
| ADR-0009 (literature-tool eval) | ☐ |

If any exit criterion or deliverable is unmet, Phase 0 extends. **Do not move to Phase 1 with gaps.** Phase 1 builds on Phase 0; gaps in Phase 0 will multiply through subsequent phases.

If all are met: Phase 0 is complete. Commit a tag (`v0.0-phase-0-complete`) to the (still-private) repo. Begin Phase 1.

---

## 9. Phase 0 risks and contingencies

**Risk: iPhone provisioning eats the week.** If Task 2.1 takes more than 3 days, pivot. Options: (a) borrow access to someone else's iOS dev setup, (b) pay for an iOS expert on Upwork/Toptal for a few hours of setup help (~$200-500), (c) defer iPhone measurements to use the iOS Simulator (which gives qualitative-but-not-latency results) and revisit later.

**Risk: Pi 5 4 GB OOMs on LFM2.5-VL-450M.** If Task 3.2 cannot get LFM2.5-VL-450M to run within 4 GB, the project's "Pi 5 is a target" claim becomes "Pi 5 8 GB is a target." Either acquire an 8 GB Pi 5 (~$80) or revise the project scope. Don't fake it.

**Risk: Stage A eval set takes a week.** 100 VQA pairs is genuinely tedious. If you're slipping, reduce to 50 VQA pairs for Phase 0 with a commitment to expand to 100 before Phase 2. Don't let perfect be the enemy of good.

**Risk: VLMEvalKit doesn't support our models.** If the adapter doesn't exist, writing one is 2-3 days. If you're under time pressure, you can defer VLMEvalKit integration to Week 5 (extending Phase 0 by a week) rather than skip it. The quality measurement is essential for Phase 2's success criterion.

**Risk: blog post writing blocks progress.** If you're spending more than 2 days on the blog post, ship a shorter version (800 words) and move on. The post can be expanded later. Working software beats unwritten blog posts.

---

## 10. What this plan does not specify

For clarity, these are deliberately deferred:

- **Specific Search Strategist Agent code.** That's Phase 1.
- **Training pipeline details.** That's Phase 2.
- **Research Analyst Agent implementation.** That's Phase 3.
- **Cloud GPU rental decisions.** Phase 2 may need it; Phase 0 does not.
- **Optional arXiv preprint structure.** Phase 2.
- **Optional Stage B target data integration.** Future, only if Stage B materializes.

Each of these gets its own detailed plan in subsequent phase plans.

---

## 11. Open questions for the project owner before Phase 0 starts

These should be resolved before you start Week 1, since they affect concrete tasks:

**Q1. What's your current Apple Developer setup?** If you have an active $99/year account and your iPhone 16 Pro is already registered as a development device, Week 2's Task 2.1 is half a day. If not, budget the full 1-2 days and possibly defer iPhone work by a few days while provisioning sorts itself out.

**Q2. Do you have a Pi 5 4 GB on hand, or do you need to order one?** Order time is ~1 week. Don't start Phase 0 without it; you'll just rush at the end of Week 3. If you're ordering, do it before Week 1 starts.

**Q3. Where will you host the eventual end-of-Phase-1 reveal blog post?** Personal blog, dev.to, Hashnode, Substack, Medium, or repo-only (just on GitHub Pages from the repo itself). Decision is small but worth making before drafting the post in Phase 0, since house style varies. Repo-only via GitHub Pages is fine and zero-setup.

**Q4. Apache 2.0 vs. MIT for the license?** Recommended Apache 2.0 in Task 1.1. If you have a strong preference for MIT, override it. Both are fine.

---

**Decisions already made (locked):**

- License for project code: Apache 2.0 (overridable to MIT if preferred — see Q4).
- Public repo timing: private through Phase 0-1, public at end of Phase 1. Reason: stronger first impression with a working system; less audience pressure during early development. Documented in ADR-0008.
- License posture for third-party models: LFM2.5-VL-450M used as Phase 1 baseline; Qwen2.5-VL-3B used as Phase 2 starting point under Qwen license; FastVLM used as benchmark reference only; distillation teacher will be LFM2-VL-3B (not FastVLM-7B). Documented in ADR-0007 and `THIRD_PARTY.md`.
- Phase 1-2 starting point: Qwen2.5-VL-3B (general, not-edge-optimized). LFM2.5-VL-450M is Phase 1 baseline only; Phase 2 demonstrates the harder general → edge journey from Qwen2.5-VL-3B.
- Phase 4 (reusability proof on second task) is now in project scope, not deferred to future work.
- Reference baseline set: LFM2.5-VL-450M, FastVLM-0.5B, SmolVLM-500M, MiniCPM-V 4.6 (the four small-edge VLMs), plus Qwen2.5-VL-3B (the unoptimized starting point).

---

## 12. Phase 0 verification

Verification is the discipline that catches the gap between "looks correct" and "is correct." For Phase 0, the verification effort is small (most work is configuration and measurement, not complex code), but the *habit* established here matters for Phases 1-3 where verification carries real weight.

**Two principles:**

1. **Verify at the boundary of each unit of work**, not just at phase exit. A schema's round-trip test runs before moving to the next schema. A device's measurement is sanity-checked against published claims before moving to the next device. Bugs caught early cost minutes; bugs caught at phase exit cost days.

2. **Verification produces an artifact.** Every verification check writes its result somewhere persistent — a test passing in CI, a row in `docs/verifications/phase_0.md`, a screenshot in the dashboard's history. "I checked it" without an artifact doesn't count, because three weeks from now you won't remember which version you checked.

### 12.1 Automated verification

These run on every commit (eventually via CI; manually via `pytest` for now):

| Check | What it verifies | When |
|---|---|---|
| `pytest tests/test_schemas.py` | Each JSON Schema validates its example fixtures; each Pydantic model round-trips JSON cleanly | After every schema change |
| `python -c "from schemas import *"` | All schema modules import without errors | After every schema change |
| `python -c "import yaml; [yaml.safe_load(open(f)) for f in glob('configs/devices/*.yaml')]"` | All DeviceDescriptor YAMLs parse | After Task 1.4 and any later device-config change |
| `jsonschema -i tests/fixtures/<name>.json schemas/<schema>.schema.json` | Each fixture validates against its schema (CLI sanity check beyond Pydantic) | After every schema or fixture change |

Add a `Makefile` target `make verify-phase-0-auto` that runs all four. Should take seconds.

### 12.2 Manual verification (signed off in `docs/verifications/phase_0.md`)

These can't be automated and require human judgment. Record each in `docs/verifications/phase_0.md` with date, check name, result (pass/fail), and brief notes:

**Week 2 (Mac):**
- [ ] Qwen2.5-VL-3B loads on Mac mini, produces non-empty output for 5 sample photos
- [ ] Qwen2.5-VL-3B memory measurement stable across 3 runs (variance < 20%)
- [ ] VLMEvalKit produces quality scores for all 5 reference models on 3 benchmark slices, scores logged in metrics DB
- [ ] VLMEvalKit scores for FastVLM-0.5B and LFM2.5-VL-450M within ~20% of published claims on overlapping benchmarks

**Week 3 (iPhone):**
- [ ] Apple Developer provisioning works; blank app deploys to iPhone 16 Pro
- [ ] LFM2.5-VL-450M produces sensible captions for 5 sample photos on iPhone (caption quality is human-judged, not numerically scored)
- [ ] LFM2.5-VL-450M TTFT within ~20% of published claim on iPhone (or documented explanation if not)
- [ ] FastVLM-0.5B TTFT under ~120ms on iPhone (Apple's published claim)
- [ ] SmolVLM-500M produces sensible captions on iPhone
- [ ] MiniCPM-V 4.6 produces sensible captions on iPhone, OR non-fit explicitly documented
- [ ] All iPhone measurements logged in metrics DB with consistent device_id and harness version

**Week 4 (Pi):**
- [ ] LFM2.5-VL-450M produces sensible captions on Pi 5
- [ ] Pi 5 measurements confirmed to not hit swap (`free -h` checks pre and post)
- [ ] SmolVLM-500M produces sensible captions on Pi 5 (or non-fit documented)
- [ ] FastVLM-on-Pi non-viability empirically confirmed (or surprising-fit documented in ADR)
- [ ] Stage A eval set: 200 photos exist, 100 captions exist, 100 VQA pairs exist, manifest hash recorded

**Week 5 (close):**
- [ ] Dashboard renders without errors, all 4 tabs populated, all 5 reference markers visible
- [ ] `THIRD_PARTY.md` covers every third-party model, dataset, and runtime used
- [ ] All 9 ADRs exist at expected paths
- [ ] Phase 0 blog post draft exists, ~1500 words, honestly framed (not over-promising)
- [ ] Retrospective written, honestly addressing what slipped and why

### 12.3 Verification artifact

Create `docs/verifications/phase_0.md` with the manual checklist above as the template. As each check passes, fill in date and brief notes. The completed document is the verification artifact for Phase 0. Commit it.

Example entry format:
```markdown
### Qwen2.5-VL-3B loads on Mac mini
**Date:** 2026-05-19
**Result:** Pass
**Notes:** Loaded at FP16, ~7.1 GB resident. First inference 12.3s
(cold start). Steady-state decode ~8.4 tok/s. Output for sample photo
"beach_sunset.jpg" was a coherent 2-sentence description.
```

### 12.4 What verification does NOT include in Phase 0

For clarity, these are deliberately *not* part of Phase 0 verification:

- **Performance benchmarking against the success criteria.** Phase 0 establishes reference baselines; it doesn't optimize against them. That's Phase 1+.
- **End-to-end agent loop tests.** No agents exist yet. The Search Strategist Agent is built in Phase 1.
- **Cross-device deployment validation.** Each device is verified independently in Phase 0; the system that orchestrates across them comes in Phase 1.
- **CI/CD setup.** Manual `pytest` is sufficient for Phase 0's scale. CI can be added in Phase 1 if commit volume justifies it.

The Phase 1 plan will define its own §12 verification with the appropriate depth for what gets built there.

---

*Next document, after Phase 0 completes: "VLM_Optimization_DetailedPlan_Phase1.md."*