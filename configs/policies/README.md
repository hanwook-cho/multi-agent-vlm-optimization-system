# Threshold Monitor Policies

YAML policy files consumed by the Threshold Monitor service (Phase 1+).
Each file defines the signal thresholds that trigger a Decision Dossier.

Policy fields (to be formally schematised in Phase 1):
- `pareto_velocity_window`: number of recent experiments to assess for Pareto movement
- `stagnation_threshold`: minimum new Pareto points per window before flagging
- `search_coverage_target`: fraction of Mode A search space to explore before considering escalation
- `per_experiment_compute_cap`: wall-clock and cost limits that auto-kill a run
- `canary_regression_threshold`: loss divergence threshold for auto-kill

Added in Phase 1 alongside the Threshold Monitor implementation.
