"""
schemas/run_config.py
─────────────────────
RunConfig — the operator's "goal & scope intake" (ADR-0013 H1, the start surface).

A durable, checked-in record of what the human authorized for a run: the goal, the
success criteria, the target device, the eval set, and the search space the agent
may explore. The HLD §5.1 gates "what the agent may search" at the start; this file
is that authorization in reproducible, diffable form (vs. only living in chat).

H1 scope: the schema + loader + a stamped record on the build. Enforcing the
allowed-search-space against the Search Strategist's proposals is a later step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    goal: Annotated[str, Field(min_length=1, description="One-line objective for this run.")]
    chat_backend: Annotated[Literal["local", "api"], Field(
        default="local",
        description="Which backend powers the agent/operator chat. 'local' = llama.cpp "
                    "+ Qwen2.5 (private, free, default); 'api' = frontier API (opt-in, "
                    "per-token cost). Maps to STRATEGIST_BACKEND (ADR-0013).")]
    success_criteria: Annotated[dict[str, float], Field(
        default_factory=dict,
        description="Named numeric bars, e.g. {'POPE': 86.0}. Empty = qualitative only.")]
    target_device: Annotated[str, Field(default="mac_mini_m4_16gb", description="DeviceDescriptor.device_id.")]
    eval_set: Annotated[list[str], Field(
        default_factory=lambda: ["POPE", "RealWorldQA", "MMBench_DEV_EN"],
        description="Benchmarks defining 'good' for this run.")]
    allowed_hypotheses: Annotated[list[str], Field(
        default_factory=list,
        description="Hypothesis ids the agent may pursue (empty = all open). The authorized search space.")]
    notes: Annotated[str | None, Field(default=None, description="Free-form operator note.")]


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run.yaml. Raises if the file is missing or invalid."""
    import yaml
    data = yaml.safe_load(Path(path).read_text()) or {}
    return RunConfig.model_validate(data)
