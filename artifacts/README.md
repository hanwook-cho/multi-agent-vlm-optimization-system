# Artifacts

Result bundles, exported models, and device builds. **Gitignored** — contents are
large, locally produced, and reconstructible from the experiment ledger + configs.

This directory exists in the repo so the path is stable. Its contents are not committed.

## Structure (produced by the system at runtime)

```
artifacts/
  experiments/
    <experiment_id>/        # SHA-256 of the ExperimentConfig
      config.json           # The ExperimentConfig that produced this
      metrics.json          # The MetricsReport
      model/                # Exported model files (GGUF, CoreML, MLX weights)
  baselines/
    <model_id>/             # Reference model measurements from Phase 0
      <device_id>/
        metrics.json
```

To archive important artifacts for sharing, use content-addressed bundles
and publish them separately (e.g. HuggingFace Hub, GitHub Releases).
