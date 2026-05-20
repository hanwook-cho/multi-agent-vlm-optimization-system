#!/usr/bin/env python3
"""
runners/measure_mac.py — Task 2.1 measurement script (MPS backend).

Loads Qwen2.5-VL-3B-Instruct at FP16 on MPS, runs a warmup pass, then
measures TTFT / decode tokens/sec / peak memory / on-disk size over a
list of images. Outputs a MetricsReport JSON.

Usage (from repo root):
    python runners/measure_mac.py \\
        --images path/img1.jpg path/img2.jpg path/img3.jpg path/img4.jpg path/img5.jpg \\
        --output results/qwen25_vl_3b_fp16_mac_mini.json

Requirements (install once):
    pip install torch torchvision transformers accelerate qwen-vl-utils pillow psutil

The MetricsReport written here uses:
    device_id:     mac_mini_m4_16gb
    experiment_id: SHA-256 of the ExperimentConfig (content_hash())
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

import psutil
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LogitsProcessor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from schemas import ExperimentConfig, MetricsReport
from schemas.experiments import (
    CompressionSpec,
    DecodeStrategy,
    ExperimentStatus,
    HardwareFingerprint,
    WeightDtype,
)
from schemas.devices import Runtime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE_ID = "mac_mini_m4_16gb"
PROMPT = "Describe this image in detail."
MAX_NEW_TOKENS = 150
WARMUP_MAX_NEW_TOKENS = 50  # shorter warmup to save time


# ---------------------------------------------------------------------------
# TTFT timer (injected as a LogitsProcessor)
# ---------------------------------------------------------------------------

class TTFTTimer(LogitsProcessor):
    """Records wall-clock time of the first token generation call."""

    def __init__(self) -> None:
        self.first_token_time: float | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        return scores


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _macos_version() -> str:
    ver = platform.mac_ver()[0]
    return f"macOS {ver}" if ver else platform.platform()


def _thermal_state() -> str | None:
    """Return Apple thermal state string via pmset, or None on failure."""
    try:
        out = subprocess.check_output(
            ["pmset", "-g", "therm"], text=True, timeout=3
        )
        # pmset output: "CPU_Speed_Limit         = 100\n..."
        # Thermal level is implicit; just return the raw relevant line.
        for line in out.splitlines():
            if "CPU_Speed_Limit" in line:
                limit = line.split("=")[-1].strip()
                # 100 = nominal, <100 = throttling
                return "nominal" if limit == "100" else f"throttled (cpu_limit={limit}%)"
    except Exception:
        pass
    return None


def _available_ram_mb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024)


def _swap_used_mb() -> float:
    return psutil.swap_memory().used / (1024 * 1024)


def _peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _mps_allocated_mb() -> float:
    """MPS driver-allocated memory in MB (includes model weights + activations)."""
    try:
        return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def _on_disk_size_mb(model_id: str) -> float:
    """Sum of all files in the HuggingFace cache for this model."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    # Cached directory name: models--{org}--{repo} with slashes → double dash
    dir_name = "models--" + model_id.replace("/", "--")
    model_dir = cache_root / dir_name
    if not model_dir.exists():
        return 0.0
    total = sum(
        f.stat().st_size
        for f in model_dir.rglob("*")
        if f.is_file() and not f.is_symlink()
    )
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Dataset hash
# ---------------------------------------------------------------------------

def _dataset_hash(image_paths: list[Path]) -> str:
    """SHA-256 of sorted absolute paths + file sizes (lightweight fingerprint)."""
    h = hashlib.sha256()
    for p in sorted(image_paths):
        h.update(str(p.resolve()).encode())
        h.update(b":")
        h.update(str(p.stat().st_size).encode())
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Single-image measurement
# ---------------------------------------------------------------------------

def measure_image(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    device: torch.device,
    max_new_tokens: int,
    warmup: bool = False,
) -> dict:
    """
    Run inference on one image. Returns dict with ttft_ms, decode_tps,
    peak_memory_mb, output_text.
    """
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    n_input_tokens = inputs["input_ids"].shape[1]

    ttft_timer = TTFTTimer()
    memory_before = _mps_allocated_mb()
    t_start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[ttft_timer],
        )

    t_end = time.perf_counter()
    memory_after = _mps_allocated_mb()

    n_generated = output_ids.shape[1] - n_input_tokens
    ttft_ms = (
        (ttft_timer.first_token_time - t_start) * 1000
        if ttft_timer.first_token_time is not None
        else None
    )
    decode_time_s = (
        t_end - ttft_timer.first_token_time
        if ttft_timer.first_token_time is not None
        else None
    )
    # n_generated - 1: first token is accounted for in TTFT; remainder is decode
    decode_tps = (
        (n_generated - 1) / decode_time_s
        if decode_time_s and n_generated > 1
        else None
    )

    output_text = processor.decode(
        output_ids[0][n_input_tokens:], skip_special_tokens=True
    )

    return {
        "ttft_ms": ttft_ms,
        "decode_tps": decode_tps,
        "peak_memory_mb": max(memory_before, memory_after),
        "n_generated": n_generated,
        "output_text": output_text,
        "warmup": warmup,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Qwen2.5-VL-3B on Mac mini (MPS)")
    parser.add_argument(
        "--images", nargs="+", required=True, metavar="PATH",
        help="Paths to 5+ image files (JPEG/PNG).",
    )
    parser.add_argument(
        "--output", default="results/qwen25_vl_3b_fp16_mac_mini.json",
        help="Path to write the MetricsReport JSON.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help=f"Max tokens to generate per image (default: {MAX_NEW_TOKENS}).",
    )
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images]
    for p in image_paths:
        if not p.exists():
            sys.exit(f"Image not found: {p}")

    if len(image_paths) < 1:
        sys.exit("Provide at least one image.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device check
    # ------------------------------------------------------------------
    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available. Requires macOS 12.3+ and an Apple Silicon Mac.")
    device = torch.device("mps")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Build ExperimentConfig (needed for experiment_id / content_hash)
    # ------------------------------------------------------------------
    dataset_hash = _dataset_hash(image_paths)
    experiment_config = ExperimentConfig(
        model_id=MODEL_ID,
        compression=CompressionSpec(weight_dtype=WeightDtype.FP16),
        input_resolution=None,
        vision_token_budget=None,
        runtime_backend=Runtime.PYTORCH_MPS,
        decode_strategy=DecodeStrategy.GREEDY,
        dataset_hash=dataset_hash,
        target_device_id=DEVICE_ID,
        notes="Task 2.1 baseline — unoptimized FP16, MPS backend, Mac mini M4 16 GB",
    )
    experiment_id = experiment_config.content_hash()
    print(f"experiment_id: {experiment_id}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\nLoading {MODEL_ID} at FP16 on MPS …")
    load_start = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    load_elapsed = time.perf_counter() - load_start
    print(f"Model loaded in {load_elapsed:.1f}s")
    print(f"On-disk size: {_on_disk_size_mb(MODEL_ID):.0f} MB")

    # ------------------------------------------------------------------
    # Warmup (one pass, result discarded)
    # ------------------------------------------------------------------
    print(f"\nWarmup pass on {image_paths[0].name} …")
    warmup_result = measure_image(
        model, processor, image_paths[0], device,
        max_new_tokens=WARMUP_MAX_NEW_TOKENS,
        warmup=True,
    )
    print(f"  warmup TTFT: {warmup_result['ttft_ms']:.0f} ms  "
          f"  decode: {warmup_result['decode_tps']:.1f} tok/s  (discarded)")

    # ------------------------------------------------------------------
    # Measurement loop
    # ------------------------------------------------------------------
    started_at = datetime.now(timezone.utc)
    results = []
    peak_memory_mb_overall = 0.0

    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {img_path.name}")
        snap_ram = _available_ram_mb()
        r = measure_image(
            model, processor, img_path, device,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(r)
        peak_memory_mb_overall = max(peak_memory_mb_overall, r["peak_memory_mb"])

        ttft_str = f"{r['ttft_ms']:.0f} ms" if r["ttft_ms"] is not None else "N/A"
        tps_str  = f"{r['decode_tps']:.1f} tok/s" if r["decode_tps"] is not None else "N/A"
        print(f"  TTFT: {ttft_str}   decode: {tps_str}   tokens: {r['n_generated']}")
        print(f"  output: {r['output_text'][:120].strip()!r}")

    completed_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Aggregate (median over all images)
    # ------------------------------------------------------------------
    def _median(values: list[float | None]) -> float | None:
        vals = sorted(v for v in values if v is not None)
        if not vals:
            return None
        mid = len(vals) // 2
        return (vals[mid - 1] + vals[mid]) / 2 if len(vals) % 2 == 0 else vals[mid]

    ttft_median    = _median([r["ttft_ms"] for r in results])
    tps_median     = _median([r["decode_tps"] for r in results])
    on_disk_mb     = _on_disk_size_mb(MODEL_ID)

    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  TTFT (median):         {ttft_median:.0f} ms" if ttft_median else "  TTFT: N/A")
    print(f"  Decode (median):       {tps_median:.1f} tok/s" if tps_median else "  Decode: N/A")
    print(f"  Peak memory (RSS):     {peak_memory_mb_overall:.0f} MB")
    print(f"  On-disk size:          {on_disk_mb:.0f} MB")

    # ------------------------------------------------------------------
    # Build and validate MetricsReport
    # ------------------------------------------------------------------
    fingerprint = HardwareFingerprint(
        os_version=_macos_version(),
        available_ram_mb=_available_ram_mb(),
        thermal_state=_thermal_state(),
        swap_used_mb=_swap_used_mb(),
    )
    if fingerprint.swap_contaminated:
        print(f"\nWARNING: swap in use ({fingerprint.swap_used_mb:.0f} MB) — "
              "latency figures are unreliable (swap_contaminated=True).")

    report = MetricsReport(
        experiment_id=experiment_id,
        device_id=DEVICE_ID,
        status=ExperimentStatus.COMPLETED,
        started_at=started_at,
        completed_at=completed_at,
        hardware_fingerprint=fingerprint,
        ttft_ms=ttft_median,
        decode_tokens_per_sec=tps_median,
        peak_memory_mb=peak_memory_mb_overall,
        on_disk_size_mb=on_disk_mb if on_disk_mb > 0 else None,
    )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_path.write_text(report.model_dump_json(indent=2))
    print(f"\nMetricsReport written to {output_path}")

    # Also save the ExperimentConfig alongside it so experiment_id is reproducible
    config_path = output_path.with_name(output_path.stem + "_config.json")
    config_path.write_text(experiment_config.model_dump_json(indent=2))
    print(f"ExperimentConfig written to {config_path}")


if __name__ == "__main__":
    main()
