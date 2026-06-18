# Observation: Constructed Student Validated on Apple Silicon via MLX (the edge half)

**Date:** 2026-06-18
**Context:** The constructed students had only ever been scored *same-path on the Mac* — never run in a deployable runtime, leaving the "edge" half of the proof-of-work unproven. Per ADR-0014, we port the `d3423bc0` student (SigLIP + fresh MLP projector + Qwen2.5-0.5B + LoRA) to MLX in Python, **verify it reproduces the PyTorch student**, then measure Apple-Silicon performance — before committing to any Swift/iPhone work.
**Verdict:** **Success — parity gate PASSED and the student is edge-viable.** The MLX re-implementation faithfully reproduces the evaluated PyTorch student, and runs fast in a small footprint on Apple Silicon. The iPhone Swift port is now *optional confirmation*, not a blocker.

---

## Results (`d3423bc0`, isolated `.venv-mlx`: mlx 0.31.2 / mlx-lm 0.31.3 / mlx-vlm 0.6.3)

| Step | Result |
|---|---|
| **1. LM half** (merge LoRA → Qwen2.5-0.5B → MLX) | ✅ converts + generates coherent text |
| **2. Vision + projector** (faithful MLX SigLIP + `projector.pt`) | ✅ `last_hidden_state` vs transformers **max\|Δ\| = 2.0e-04** |
| **3. Parity gate** (assembled MLX vs PyTorch `StudentVLM`, same image+prompt) | ✅ **identical greedy output** (`'No'`) — gate PASS |
| **3.5. Mac (Apple-Silicon) perf** | **TTFT 118 ms** (incl. vision), **80.9 tok/s** decode, **peak ≈1.6 GB** |

## Reading

- **The MLX student == the evaluated student.** A faithful vision tower (2e-04) plus
  an identical decode means the on-device numbers correspond to the *same* model
  whose quality we measured (POPE bal-acc 68.3) — the numbers are coherent, not a
  different model that happens to run.
- **Edge-viable.** 1.6 GB peak fits the iPhone 16 Pro comfortably — the exact
  constraint that made Qwen2.5-VL-3B (6.5 GB) non-viable. 80.9 tok/s on the M4 and
  118 ms TTFT are well inside usable bounds. The student is forecast to run on the
  iPhone; the Mac-MLX number is itself a legitimate Apple-Silicon edge result.
- **Swift port now optional.** Per ADR-0014's gate logic, step 3.5 already gives a
  credible edge result; the iPhone `MLXVLMRunner` run becomes confirmation, to do
  only if an iPhone-specific number is needed.

## What this closes

The proof-of-work had two halves: *quality* (Mac same-path eval — done) and *edge
deployability* (unproven until now). This closes the second half for `d3423bc0`:
the system-constructed student is faithfully runnable and fast in a real
Apple-Silicon inference runtime, in an iPhone-sized footprint.

## Caveats / follow-ups

- Parity verified on one image+prompt + the 2e-04 vision match; a cheap follow-up is
  a multi-image greedy-parity sweep and an MLX-vs-PyTorch floor-adjusted eval-slice
  comparison (re-score POPE in MLX, confirm ≈68.3).
- Perf is **Mac M4** (Apple Silicon, same MLX runtime), not the iPhone itself; the
  iPhone has less RAM + ANE specifics. The 1.6 GB footprint strongly forecasts
  viability but the device number requires the (optional) Swift step.
- Internal-only quality numbers as always (memory `benchmark-eval-internal-only`).

## Artifacts

- Exporter: `runners/export_student_mlx.py` (`--step lm|vision|parity|perf|all`).
- MLX student: `artifacts/students/build_d3423bc0155b/mlx_export/` (gitignored).
- Decision: [`ADR-0014`](../decisions/0014-on-device-deployment-via-mlx-swift.md).
