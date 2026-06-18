# ADR-0014 — On-device deployment of constructed students via MLX-Swift

**Date:** 2026-06-18
**Status:** Accepted
**Context revises:** the Deployment Dispatcher / iOS-harness path for *constructed* students (not the GGUF reference models).

## Context

The iOS harness (`ios_harness/`, `LlamaVLMRunner`) runs **GGUF** reference models
(LFM2, SmolVLM, MiniCPM-V) via **llama.cpp + libmtmd**. The Phase-2 *constructed
students* (`d3423bc0` POPE, `b2feb6b1` MMBench) are a different shape: a
custom-assembled PyTorch VLM — SigLIP-base vision encoder → a **freshly-trained
2-layer MLP projector** → image-embed **prepend** (LLaVA-style, no placeholder
token) → Qwen2.5-0.5B-Instruct **+ a LoRA adapter**. They have only ever been
scored **same-path on the Mac (n=100)**, never run on the iPhone — so the *edge*
half of the proof-of-work (real TTFT / memory on device) is unvalidated.

The harness cannot run them as-is: llama.cpp/mtmd expects a GGUF model + a GGUF
`mmproj` for a *known* vision architecture and a chat template with an `<image>`
marker. A custom projector + no-placeholder prepend does not map cleanly.

## Decision

**Deploy constructed students on-device via MLX (MLX-Swift on the device; mlx-vlm
in Python for conversion/validation), not via GGUF/llama.cpp.** The harness gains a
second runner (`MLXVLMRunner`) alongside `LlamaVLMRunner`, selected when the chosen
model is a constructed student.

**De-risk in Python on the Mac first:** before any Swift work, port the assembled
forward to MLX in Python and verify it **reproduces the PyTorch student's outputs /
floor-adjusted scores**. Only proceed to the Swift integration once parity holds.

## Rationale

- **We own the model code.** The student is our assembly, so we re-express its
  (simple) forward in MLX rather than fighting a converter that assumes a fixed
  architecture — the opposite of the GGUF/mtmd situation.
- **MLX is Apple's de-facto on-device VLM runtime.** Apple's own small-VLM demos
  (FastVLM) use MLX; the environment already has the `mlx-swift-examples` checkout
  with the `MLXVLM` library.
- **Conversion is flexible.** PyTorch→MLX weight conversion (`mlx_lm.convert`,
  mlx-vlm) is well-trodden for Qwen2.5 + SigLIP; the projector is a trivial MLP.
- **GGUF was rejected** as fragile/high-effort for a custom-assembled VLM (custom
  mmproj + faked chat template). **Core ML / ExecuTorch** deferred: generative-VLM
  export is finicky and Apple themselves use MLX here.

## Plan (incremental, each step verifiable)

1. **LM half (foundational):** merge the LoRA into Qwen2.5-0.5B, convert to MLX,
   verify the MLX LM loads and generates coherent text. **✅ DONE (2026-06-18)** —
   verified in an isolated `.venv-mlx` (mlx 0.31.2 + mlx-lm 0.31.3 + mlx-vlm 0.6.3 +
   torch); the MLX LM generates coherent text. See `runners/export_student_mlx.py`.
2. **Vision + projector:** convert SigLIP-base to MLX; load `projector.pt` into an
   MLX MLP; implement SigLIP-encode → project → prepend. **✅ DONE (2026-06-18)** —
   faithful functional MLX SigLIP; `last_hidden_state` matches transformers to
   **max|Δ| = 2.0e-04**.
3. **Assembled-forward parity (the gate):** greedy-decode a descriptive prompt
   through the PyTorch `StudentVLM` and the MLX student on several images; require
   identical output on each. **✅ DONE / GATE PASSED (2026-06-18)** — **5/5 images
   produce identical multi-token captions** (strengthened from an initial single
   yes/no example; a coincidental multi-image multi-token match is effectively
   impossible). Combined with the 2e-04 vision parity, the MLX student **faithfully
   reproduces the evaluated PyTorch student.** (The parity relies on the Python
   `mlx_lm` model's `input_embeddings` path — see step 4.)
3.5. **Mac (Apple-Silicon MLX) perf:** measure the assembled student on the M4 — a
   real on-device-class number with no Swift. **✅ DONE (2026-06-18)** —
   **TTFT 118 ms (incl. vision), 80.9 tok/s decode, peak ≈1.6 GB.** The 1.6 GB peak
   fits the iPhone 16 Pro budget comfortably (vs. Qwen2.5-VL-3B's 6.5 GB non-viable),
   so the student is forecast to run on-device.
3.6. **On-device bundle.** **✅ DONE (2026-06-18)** — `--step bundle` writes a
   self-contained MLX bundle (`lm_mlx/` + `vision.safetensors` + `projector.safetensors`,
   both fp16, + `student_config.json`), ~1.14 GB, that a Swift runner would load.
4. **Swift / iPhone — ⏸️ DEFERRED (2026-06-18).** A real blocker surfaced while
   wiring `MLXVLMRunner`: the student is **LLaVA-style** (it *prepends* image embeds),
   but `mlx-swift-examples`' `Qwen2Model` only accepts **token ids** — its inner
   `layers` are `fileprivate` and there is **no public `inputsEmbeds` path** (unlike
   the Python `mlx_lm` model, which has one; the parity in step 3 depends on it). So
   the runner can't be a thin port of `LlamaVLMRunner` — it needs a **vendored custom
   Qwen2 forward** (or a patched MLXLLM fork). Since step 3.5 already gives a
   **verified Apple-Silicon edge number**, the literal iPhone run is a *nice-to-have*,
   not load-bearing — **deferred** rather than worth the custom-decoder effort now. A
   WIP runner (SigLIP/projector/image faithfully ported; LM glue blocked) is preserved
   at `ios_harness/VLMHarness/Inference/MLXVLMRunner.swift.draft` (kept out of the
   Xcode build). Reopen if an iPhone-specific number is required.
5. **Record** the result to the ledger / observations — done via the
   2026-06-18 on-device-validation observation.

## Consequences

- A new optional dependency extra (`mlx`: mlx, mlx-lm, mlx-vlm) for the conversion
  path. The harness gains a second, MLX-based runner.
- Step 3 is the **go/no-go gate**: if MLX parity can't be achieved, reassess (Core ML
  / a GGUF attempt / report LM-only proxy numbers).
- This validates the *constructed* students specifically; the GGUF reference-model
  path (`LlamaVLMRunner`) is unchanged.

## Open issues

- Whether mlx-vlm has a ready SigLIP+Qwen LLaVA-style class to reuse, or we add a
  small custom model definition (the prepend logic is simple either way).
- Device memory headroom for SigLIP encode + 0.5B decode + image tokens on the
  iPhone 16 Pro (the very constraint the measurement will quantify).
