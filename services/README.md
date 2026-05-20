# Services

Deterministic components. No LLM required; standard logging only.
See `docs/HLD.md §6.1` for the full list and each service's role.

| Service | Module | Phase | Description |
|---|---|---|---|
| Experiment Runner | `experiment_runner.py` | 1 | Takes an ExperimentConfig, dispatches to the appropriate runner, captures a MetricsReport |
| Evaluation Harness | `evaluation_harness.py` | 1 | Wraps VLMEvalKit and the Stage A eval; produces BenchmarkScore objects |
| Pareto Tracker | `pareto_tracker.py` | 1 | Maintains per-device Pareto frontiers; computes dominance |
| Deployment Dispatcher | `deployment_dispatcher.py` | 1 | Reads DeviceDescriptor, selects export pipeline, packages artifact |
| Threshold Monitor | `threshold_monitor.py` | 1 | Watches signals from HLD §4.2; posts Decision Dossier when thresholds cross |
| Human Approval Queue | `approval_queue.py` | 1 | Append-only log; web/CLI surface for all gated decisions |
| Technique Registry | `technique_registry.py` | 1 | Codified Mode A search space; scans `techniques/` at startup |
