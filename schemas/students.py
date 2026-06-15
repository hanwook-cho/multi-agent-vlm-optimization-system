"""
schemas/students.py
───────────────────
StudentSpec — a declarative, content-addressable description of a VLM the system
*constructs* (Phase 2, ADR-0012). It is to `runners/build_student.py` what
ExperimentConfig is to the Experiment Runner: the Search Strategist proposes a
StudentSpec, the builder assembles → aligns → distills → evaluates it, and the
result lands in the ledger keyed by the spec's content hash.

This converts model construction from a Tier-2 human one-off into a Tier-1-style
search point the agent drives. The human writes the generic builder once; every
*instance* is a spec.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class ProjectorSpec(BaseModel):
    """The vision→LM bridge. An MLP mapping vision-encoder hidden states into the
    LM embedding space (the only newly-initialised module when init='scratch')."""
    type: Annotated[Literal["mlp"], Field(default="mlp", description="Projector architecture. Only 'mlp' is supported today.")]
    depth: Annotated[int, Field(default=2, ge=1, le=4, description="Number of Linear layers in the projector MLP.")]
    hidden: Annotated[int, Field(default=2048, ge=64, le=8192, description="Hidden width of the projector MLP.")]

    model_config = {"use_enum_values": True}


class AlignSpec(BaseModel):
    """Stage-1: train ONLY the projector to connect the (frozen) modalities."""
    data: Annotated[str, Field(description="Named alignment dataset (e.g. a caption cache key like 'coco_caption_5k').")]
    steps: Annotated[int, Field(default=2000, ge=1, description="Optimizer steps for projector alignment.")]


class DistillSpec(BaseModel):
    """Stage-2: LoRA-distill the assembled student from the teacher on a task-aligned set."""
    teacher: Annotated[str, Field(default="Qwen2.5-VL-3B", description="Teacher model the targets came from.")]
    data: Annotated[str, Field(description="Named distillation cache key (e.g. 'qa_balanced_5k' — task-aligned + hard negatives, the P2-D2 fix).")]
    lora_r: Annotated[int, Field(default=16, ge=1, le=256, description="LoRA rank.")]
    epochs: Annotated[int, Field(default=3, ge=1, description="Distillation epochs.")]
    rehearse_frac: Annotated[float, Field(default=0.2, ge=0.0, le=1.0, description="Fraction of caption/rehearsal data mixed in to fight forgetting (P2-D1/D2 lesson).")]


class EvalSpec(BaseModel):
    """Same-path MCQ eval suite the constructed student is scored on (P2-1.3 methodology)."""
    benchmarks: Annotated[list[str], Field(default_factory=lambda: ["POPE", "RealWorldQA", "MMBench_DEV_EN"], description="Benchmarks to score against.")]
    n: Annotated[int, Field(default=100, ge=1, description="Samples per benchmark slice.")]


class StudentSpec(BaseModel):
    """A fully-specified, constructible student VLM. Content-addressable: its
    SHA-256 is the construction experiment_id used in the ledger / MetricsReport."""

    lm: Annotated[str, Field(min_length=1, description="HF id of the language-model backbone (e.g. 'Qwen/Qwen2.5-0.5B-Instruct'). Same-family as the teacher eases distillation.")]
    vision: Annotated[str, Field(min_length=1, description="HF id of the vision encoder (e.g. 'google/siglip-base-patch16-224').")]
    projector: Annotated[ProjectorSpec, Field(default_factory=ProjectorSpec, description="Vision→LM bridge spec.")]
    init: Annotated[str, Field(default="scratch", description="'scratch' (random projector, pretrained LM+vision) or 'adapt:<hf_id>' to warm-start from an existing small VLM.")]
    align: Annotated[AlignSpec, Field(description="Stage-1 projector alignment recipe.")]
    distill: Annotated[DistillSpec, Field(description="Stage-2 distillation recipe.")]
    eval: Annotated[EvalSpec, Field(default_factory=EvalSpec, description="Eval suite.")]
    target_device_id: Annotated[str, Field(default="mac_mini_m4_16gb", description="Foreign key to DeviceDescriptor.device_id — where the build runs.")]
    notes: Annotated[str | None, Field(default=None, description="Free-form human/agent annotation. Excluded from the content hash.")]

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def init_value_is_known(self) -> "StudentSpec":
        if self.init != "scratch" and not self.init.startswith("adapt:"):
            raise ValueError("init must be 'scratch' or 'adapt:<hf_id>'")
        if self.init.startswith("adapt:") and len(self.init) <= len("adapt:"):
            raise ValueError("init='adapt:' requires a model id after the colon")
        return self

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON (excluding notes). Used as experiment_id."""
        canonical = self.model_dump_json(exclude={"notes"}, exclude_none=False)
        return hashlib.sha256(canonical.encode()).hexdigest()
