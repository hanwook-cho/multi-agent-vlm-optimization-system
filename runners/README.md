# Runners

Per-device, per-runtime experiment runners. Each module knows how to:
1. Accept an `ExperimentConfig`
2. Execute the inference or training job on the target device
3. Capture and return a `MetricsReport`

All runners are invoked by the Experiment Runner service (`services/experiment_runner.py`).
The Deployment Dispatcher (`services/deployment_dispatcher.py`) selects the right runner
based on `DeviceDescriptor.preferred_runtime`.

Planned modules:
- `mlx_runner.py` — MLX inference on iPhone and Mac (Phase 1)
- `llamacpp_runner.py` — llama.cpp/GGUF inference on Pi 5 (Phase 1)
- `coreml_runner.py` — CoreML inference on iPhone, CoreML fallback (Phase 1)
- `pytorch_mps_runner.py` — PyTorch MPS for training on Compute Mac (Phase 2)
