# Eval

Evaluation harness wrappers. Wraps VLMEvalKit for standard benchmarks and implements
the custom Stage A photo-memory eval. All eval code produces `BenchmarkScore` objects
that are collected into a `MetricsReport` by the Evaluation Harness service.

Planned modules (Phase 1):
- `vlmevalkit_wrapper.py` — thin adapter around VLMEvalKit; maps its output format
  to our `BenchmarkScore` schema; handles the RealWorldQA / MMBench / POPE slices
- `stage_a_eval.py` — captioning (CIDEr/BLEU against reference captions) and VQA
  (exact-match + LLM-as-judge for open-ended answers) against `datasets/stage_a_proxy/`

Quality evaluation runs on the Compute Mac regardless of the target deployment device.
Quality is device-independent for a given set of weights.
