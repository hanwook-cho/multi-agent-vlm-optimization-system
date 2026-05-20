from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from schemas.devices import Runtime


class WeightDtype(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class ActivationDtype(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    DYNAMIC = "dynamic"


class KVCacheDtype(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class DecodeStrategy(str, Enum):
    GREEDY = "greedy"
    TOP_K = "top_k"
    SPECULATIVE = "speculative"


class CompressionSpec(BaseModel):
    weight_dtype: Annotated[WeightDtype, Field(description="Numeric format for stored weights.")]
    activation_dtype: Annotated[ActivationDtype | None, Field(default=None, description="Activation format. dynamic = per-tensor dynamic quantization. Null for weight-only schemes.")]
    group_size: Annotated[int | None, Field(default=None, ge=1, description="Group size for group-wise weight quantization (e.g. 32, 64, 128). Null when not group-wise.")]
    kv_cache_dtype: Annotated[KVCacheDtype | None, Field(default=None, description="KV cache format. Null uses runtime default (typically fp16).")]
    sparsity: Annotated[float | None, Field(default=None, ge=0.0, le=1.0, description="Fraction of weights zeroed by structured pruning. Null when no pruning.")]

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def group_size_only_for_quantized_weights(self) -> CompressionSpec:
        if self.group_size is not None and self.weight_dtype not in ("int4", "int8"):
            raise ValueError(
                f"group_size is only applicable for int4/int8 weight_dtype, got '{self.weight_dtype}'"
            )
        return self


class ExperimentConfig(BaseModel):
    model_id: Annotated[str, Field(min_length=1, description="HuggingFace model ID or absolute local path.")]
    compression: Annotated[CompressionSpec, Field(description="Compression scheme for weights and activations.")]
    input_resolution: Annotated[int | None, Field(default=None, ge=64, le=4096, description="Image side length in pixels. Null uses model default.")]
    vision_token_budget: Annotated[int | None, Field(default=None, ge=1, description="Max vision tokens. Null uses model default.")]
    runtime_backend: Annotated[Runtime, Field(description="Inference runtime. Must be in target device's supported_runtimes.")]
    decode_strategy: Annotated[DecodeStrategy, Field(description="Token sampling strategy.")]
    decode_top_k: Annotated[int | None, Field(default=None, ge=1, description="k for top-k sampling. Required when decode_strategy is top_k.")]
    speculative_draft_model_id: Annotated[str | None, Field(default=None, description="Draft model for speculative decoding. Required when decode_strategy is speculative.")]
    dataset_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the eval set manifest. Pins the eval set version.")]
    target_device_id: Annotated[str, Field(min_length=1, description="Foreign key to DeviceDescriptor.device_id.")]
    notes: Annotated[str | None, Field(default=None, description="Free-form human annotation. Not used by any service.")]

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def decode_strategy_fields_consistent(self) -> ExperimentConfig:
        if self.decode_strategy == "top_k" and self.decode_top_k is None:
            raise ValueError("decode_top_k is required when decode_strategy is top_k")
        if self.decode_strategy == "speculative" and self.speculative_draft_model_id is None:
            raise ValueError("speculative_draft_model_id is required when decode_strategy is speculative")
        if self.decode_strategy != "top_k" and self.decode_top_k is not None:
            raise ValueError(f"decode_top_k must be null when decode_strategy is '{self.decode_strategy}'")
        if self.decode_strategy != "speculative" and self.speculative_draft_model_id is not None:
            raise ValueError(f"speculative_draft_model_id must be null when decode_strategy is '{self.decode_strategy}'")
        return self

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON serialisation. Used as experiment_id in MetricsReport."""
        canonical = self.model_dump_json(exclude={"notes"}, exclude_none=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


class ExperimentStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    OOM = "oom"


class HardwareFingerprint(BaseModel):
    """Observed runtime state of the device during measurement.

    Contains ONLY dynamic state — fields that can differ between runs on the
    same device. Static device properties (chip family, total RAM, accelerators)
    are not duplicated here; they live in DeviceDescriptor, reachable via
    MetricsReport.device_id.

    The swap_contaminated property is the primary signal for Pareto Tracker
    reliability filtering: non-zero swap on Pi 5 makes latency figures suspect.
    """

    os_version: Annotated[str, Field(min_length=1, description="OS version string at measurement time (e.g. 'iOS 18.3.1', 'macOS 15.4').")]
    available_ram_mb: Annotated[float, Field(ge=0.0, description="Free RAM in MB immediately before inference.")]
    thermal_state: Annotated[str | None, Field(default=None, description="Device thermal state. Free string; device-specific (e.g. Apple: 'nominal'/'fair'/'serious'/'critical').")]
    swap_used_mb: Annotated[float | None, Field(default=None, ge=0.0, description="Swap in use in MB at measurement time. None means swap was not measured. Non-zero means latency figures in the parent MetricsReport are unreliable — see swap_contaminated.")]

    @property
    def swap_measured(self) -> bool:
        """True if swap usage was recorded for this measurement run."""
        return self.swap_used_mb is not None

    @property
    def swap_contaminated(self) -> bool:
        """True if swap was actively in use during measurement.

        When True, latency fields in the parent MetricsReport (ttft_ms,
        decode_tokens_per_sec) are unreliable and should be excluded or
        down-weighted by the Pareto Tracker.

        Returns False when swap_used_mb is None (not measured). Callers that
        need to distinguish "no swap" from "swap not measured" should also
        check swap_measured.
        """
        return self.swap_used_mb is not None and self.swap_used_mb > 0.0


class BenchmarkScore(BaseModel):
    """A single metric from a single benchmark evaluation."""

    benchmark: Annotated[str, Field(min_length=1, description="Benchmark name as reported by VLMEvalKit or custom harness (e.g. 'RealWorldQA', 'POPE').")]
    metric: Annotated[str, Field(min_length=1, description="Metric name within the benchmark (e.g. 'accuracy', 'f1', 'cider').")]
    value: Annotated[float, Field(description="Numeric value. Scale is benchmark-specific.")]


class MetricsReport(BaseModel):
    """Measured results from a completed (or failed) experiment run.

    Links back to the originating ExperimentConfig via experiment_id
    (ExperimentConfig.content_hash()). All performance fields are None
    when status is not COMPLETED.
    """

    experiment_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the ExperimentConfig (content_hash()). Foreign key to the experiment ledger.")]
    device_id: Annotated[str, Field(min_length=1, description="Foreign key to DeviceDescriptor.device_id.")]
    status: Annotated[ExperimentStatus, Field(description="Outcome of the experiment run.")]
    error_message: Annotated[str | None, Field(default=None, description="Error detail when status is not COMPLETED.")]
    started_at: Annotated[datetime, Field(description="Timezone-aware timestamp when the run began.")]
    completed_at: Annotated[datetime, Field(description="Timezone-aware timestamp when the run ended.")]
    hardware_fingerprint: Annotated[HardwareFingerprint, Field(description="Device runtime state at measurement time.")]
    ttft_ms: Annotated[float | None, Field(default=None, ge=0.0, description="Time to first token in milliseconds.")]
    decode_tokens_per_sec: Annotated[float | None, Field(default=None, ge=0.0, description="Steady-state decode throughput in tokens/sec.")]
    peak_memory_mb: Annotated[float | None, Field(default=None, ge=0.0, description="Peak RAM usage in MB during inference.")]
    on_disk_size_mb: Annotated[float | None, Field(default=None, ge=0.0, description="Total model file size on disk in MB.")]
    energy_j: Annotated[float | None, Field(default=None, ge=0.0, description="Energy consumed during the benchmark run in joules. None when not measured.")]
    quality_scores: Annotated[list[BenchmarkScore], Field(default_factory=list, description="Per-benchmark quality measurements. Empty when status is not COMPLETED.")]

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def perf_fields_null_on_failure(self) -> MetricsReport:
        if self.status != "completed":
            non_null = [
                f for f in ("ttft_ms", "decode_tokens_per_sec", "peak_memory_mb", "on_disk_size_mb", "energy_j")
                if getattr(self, f) is not None
            ]
            if non_null:
                raise ValueError(
                    f"Performance fields must be null when status is '{self.status}': {non_null}"
                )
            if self.quality_scores:
                raise ValueError(
                    f"quality_scores must be empty when status is '{self.status}'"
                )
        return self

    @model_validator(mode="after")
    def completed_at_after_started_at(self) -> MetricsReport:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not be earlier than started_at")
        return self
