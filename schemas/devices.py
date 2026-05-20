from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class Accelerator(str, Enum):
    ANE = "ane"
    GPU = "gpu"
    CPU = "cpu"


class Runtime(str, Enum):
    MLX = "mlx"
    COREML = "coreml"
    LLAMACPP_GGUF = "llamacpp_gguf"
    ONNX_CPU = "onnx_cpu"
    PYTORCH_MPS = "pytorch_mps"


class DeviceRole(str, Enum):
    DEPLOYMENT_TARGET = "deployment_target"
    MEASUREMENT_AND_TRAINING = "measurement_and_training"


class DeviceDescriptor(BaseModel):
    device_id: Annotated[str, Field(min_length=1, description="Unique identifier; foreign key in ExperimentConfig and MetricsReport.")]
    chip_family: Annotated[str, Field(min_length=1, description="Hardware chip family (e.g. apple_a18_pro, broadcom_bcm2712).")]
    ram_gb: Annotated[int, Field(ge=1, description="Total physical RAM in gigabytes.")]
    accelerators: Annotated[list[Accelerator], Field(min_length=1, description="Hardware acceleration units available.")]
    supported_runtimes: Annotated[list[Runtime], Field(min_length=1, description="Runtimes the Deployment Dispatcher may target on this device.")]
    preferred_runtime: Annotated[Runtime, Field(description="Default runtime; must appear in supported_runtimes.")]
    quirks: Annotated[list[str], Field(description="Operational notes: failure modes, measurement gotchas, thermal characteristics.")]
    measurement_harness: Annotated[str, Field(min_length=1, description="Harness implementation name (e.g. ios_swift_bridge_v1).")]
    role: Annotated[DeviceRole, Field(default=DeviceRole.DEPLOYMENT_TARGET, description="deployment_target (iPhone, Pi) or measurement_and_training (Compute Mac).")]

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def preferred_runtime_must_be_supported(self) -> DeviceDescriptor:
        if self.preferred_runtime not in self.supported_runtimes:
            raise ValueError(
                f"preferred_runtime '{self.preferred_runtime}' is not in supported_runtimes {self.supported_runtimes}"
            )
        return self
