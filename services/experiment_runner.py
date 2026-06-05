"""
services/experiment_runner.py
──────────────────────────────
Experiment Runner service for Phase 1.

Accepts an ExperimentConfig, runs Mac quality evaluation (CLIP-score on
Stage A captions), writes an iPhone-ready flag for manual device deployment,
and returns a validated MetricsReport.

Phase 1 scope:
  - Mac quality eval: always runs on PyTorch MPS (fp16/fp32 or pre-quantized AWQ)
  - iPhone performance: manual step; runner writes a ready-flag that the human
    acts on. Use import_iphone_results() to merge device measurements later.

Supported backends (Phase 1):
  - pytorch_mps + fp16 / fp32 / bfloat16  → standard HF transformers load
  - pytorch_mps + int4                     → pre-quantized AWQ checkpoint
                                             (model_id must be a local path)
  - llamacpp_gguf + any dtype              → quality eval runs on PyTorch fp16
                                             proxy; GGUF goes to iPhone for perf

Usage:
    from services.experiment_runner import ExperimentRunner
    from schemas.experiments import ExperimentConfig, CompressionSpec

    config = ExperimentConfig(
        model_id="LiquidAI/LFM2-VL-450M",
        compression=CompressionSpec(weight_dtype="fp16"),
        runtime_backend="pytorch_mps",
        decode_strategy="greedy",
        dataset_hash="e2128ae022b3720375d7c866a037b6d8ec4b399ff92cb59e6065ec9fb7f3e29f",
        target_device_id="iphone16pro-001",
    )
    runner = ExperimentRunner()
    report = runner.run(config)
    print(report.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

# ── Project root resolution ───────────────────────────────────────────────────

_HERE        = Path(__file__).parent
PROJECT_ROOT = _HERE.parent

# Make project root importable (for runners/, schemas/)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.experiments import (
    BenchmarkScore,
    ExperimentConfig,
    ExperimentStatus,
    HardwareFingerprint,
    MetricsReport,
)

# ── Constants ─────────────────────────────────────────────────────────────────

STAGE_A_HASH    = "e2128ae022b3720375d7c866a037b6d8ec4b399ff92cb59e6065ec9fb7f3e29f"
STAGE_A_DIR     = PROJECT_ROOT / "datasets" / "stage_a"
CAPTIONS_FILE   = STAGE_A_DIR / "captions.json"
PHOTOS_DIR      = STAGE_A_DIR / "photos"
PROXY_PHOTOS    = PROJECT_ROOT / "datasets" / "stage_a_proxy" / "photos"

LEDGER_DIR      = PROJECT_ROOT / "artifacts" / "experiment_ledger"
DEVICE_READY    = PROJECT_ROOT / "artifacts" / "device_ready"

DEVICE_ID_MAC   = "mac-mini-m4-16gb"
EVAL_PROMPT     = "Describe what you see in this image."
MAX_NEW_TOKENS  = 128

# Phase 0 model keys
MODEL_HF_IDS = {
    "LFM2-VL-450M":  "LiquidAI/LFM2-VL-450M",
    "SmolVLM-500M":  "HuggingFaceTB/SmolVLM-500M-Instruct",
    "MiniCPM-V-4.6": "openbmb/MiniCPM-V-4.6",
    "FastVLM-0.5B":  "apple/FastVLM-0.5B",
}


# ── Hardware fingerprint ──────────────────────────────────────────────────────

def _mac_hardware_fingerprint() -> HardwareFingerprint:
    """Snapshot Mac runtime state: OS version + available RAM."""
    import psutil

    os_ver = f"macOS {platform.mac_ver()[0]}"
    mem    = psutil.virtual_memory()
    avail  = mem.available / (1024.0 * 1024.0)

    # Thermal state via powermetrics (best-effort; requires no sudo on M-series)
    thermal = None
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True, text=True, timeout=3,
        )
        if "CPU_Speed_Limit" in result.stdout:
            limit = int([l for l in result.stdout.splitlines()
                         if "CPU_Speed_Limit" in l][0].split()[-1])
            thermal = "nominal" if limit == 100 else f"throttled_{limit}pct"
    except Exception:
        pass

    return HardwareFingerprint(
        os_version=os_ver,
        available_ram_mb=round(avail, 1),
        thermal_state=thermal,
    )


# ── Stage A eval-set loader ───────────────────────────────────────────────────

def _load_stage_a_items(max_images: int | None = None) -> list[dict]:
    """
    Load available Stage A items: {filename, caption, image_path}.
    Looks in datasets/stage_a/photos/ first, then stage_a_proxy/photos/.
    Skips entries whose image file cannot be found (graceful degradation
    when COCO photos haven't been downloaded).
    """
    if not CAPTIONS_FILE.exists():
        raise FileNotFoundError(f"Stage A captions not found: {CAPTIONS_FILE}")

    captions = json.loads(CAPTIONS_FILE.read_text())
    items = []
    for filename, meta in captions.items():
        for photos_dir in (PHOTOS_DIR, PROXY_PHOTOS):
            p = photos_dir / filename
            if p.exists():
                items.append({
                    "filename": filename,
                    "caption":  meta["caption"],
                    "path":     p,
                })
                break

    if not items:
        raise RuntimeError(
            "No Stage A images found. Download COCO val2017 images to "
            f"{PHOTOS_DIR} or add proxy images to {PROXY_PHOTOS}."
        )

    if max_images is not None:
        items = items[:max_images]

    return items


# ── Model loaders ─────────────────────────────────────────────────────────────
# Each loader returns (model, processor) and the corresponding infer_fn.
# Infer signature: infer_fn(model, processor, image: PIL.Image, prompt: str,
#                           device: str) -> str

def _load_lfm2(model_id: str, device: str, weight_dtype: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    dtype = _torch_dtype(weight_dtype)
    proc  = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    return model, proc

def _infer_lfm2(model, proc, image: Image.Image, prompt: str, device: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    text   = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _load_smolvlm(model_id: str, device: str, weight_dtype: str):
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    dtype = _torch_dtype(weight_dtype)
    proc  = AutoProcessor.from_pretrained(model_id)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device).eval()
    return model, proc

def _infer_smolvlm(model, proc, image: Image.Image, prompt: str, device: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text   = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=text, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _load_minicpm(model_id: str, device: str, weight_dtype: str):
    import glob
    from transformers import MiniCPMV4_6ForConditionalGeneration, AutoProcessor
    from huggingface_hub import snapshot_download
    # Find the snapshot that has preprocessor_config.json
    cache_base = os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4.6/snapshots"
    )
    snapshots = glob.glob(os.path.join(cache_base, "*/preprocessor_config.json"))
    local_path = os.path.dirname(snapshots[0]) if snapshots else \
                 snapshot_download(model_id, local_files_only=True)
    proc  = AutoProcessor.from_pretrained(local_path)
    dtype = torch.float16  # MiniCPM-V requires float16 on MPS
    model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        local_path, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    return model, proc

def _infer_minicpm(model, proc, image: Image.Image, prompt: str, device: str) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text   = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(images=[image], text=text, return_tensors="pt").to(device)
    n_in   = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return proc.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def _load_fastvlm(model_id: str, device: str, weight_dtype: str):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    proc  = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    return model, proc

def _infer_fastvlm(model, proc, image: Image.Image, prompt: str, device: str) -> str:
    from transformers import CLIPImageProcessor
    img_proc = CLIPImageProcessor.from_pretrained(
        "apple/FastVLM-0.5B", subfolder="image_processor", trust_remote_code=True
    )
    pixel_values = img_proc(images=image, return_tensors="pt").pixel_values.to(device)
    full_prompt  = f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids    = proc.tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(inputs=input_ids, pixel_values=pixel_values,
                             max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return proc.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# ── Model registry ────────────────────────────────────────────────────────────

# Maps model short-key → (load_fn, infer_fn)
# load_fn(model_id, device, weight_dtype) → (model, processor)
# infer_fn(model, processor, image, prompt, device) → str

_MODEL_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "LiquidAI/LFM2-VL-450M":               (_load_lfm2,     _infer_lfm2),
    "HuggingFaceTB/SmolVLM-500M-Instruct":  (_load_smolvlm,  _infer_smolvlm),
    "openbmb/MiniCPM-V-4.6":               (_load_minicpm,  _infer_minicpm),
    "apple/FastVLM-0.5B":                   (_load_fastvlm,  _infer_fastvlm),
}


def _torch_dtype(weight_dtype: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "int8": torch.float16,   # INT8 = weight-only; activations stay fp16
        "int4": torch.float16,   # INT4 = weight-only; activations stay fp16
    }.get(weight_dtype, torch.bfloat16)


def _resolve_loader(model_id: str) -> tuple[Callable, Callable]:
    """Look up the loader/infer pair for a model_id, trying short keys too."""
    if model_id in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_id]
    # Try matching by short name in MODEL_HF_IDS
    for short_key, hf_id in MODEL_HF_IDS.items():
        if model_id == short_key:
            return _MODEL_REGISTRY[hf_id]
    raise NotImplementedError(
        f"No loader registered for model_id='{model_id}'. "
        f"Known models: {list(_MODEL_REGISTRY)}"
    )


# ── CLIP scorer (imported from runners/) ──────────────────────────────────────

def _run_clip_eval(
    model,
    processor,
    infer_fn: Callable,
    items: list[dict],
    device: str,
    experiment_id: str,
) -> tuple[list[BenchmarkScore], dict]:
    """
    Generate descriptions for Stage A items and compute CLIP-score.
    Returns (list[BenchmarkScore], raw_predictions_dict).
    """
    from runners.compute_clip_score import CLIPScorer

    predictions = []
    images_pil  = []

    print(f"  Generating descriptions for {len(items)} images…")
    for item in items:
        img = Image.open(item["path"]).convert("RGB")
        images_pil.append(img)
        text = infer_fn(model, processor, img, EVAL_PROMPT, device)
        predictions.append({"image": item["filename"], "text": text})
        print(f"    {item['filename']}: \"{text[:70]}\"")

    print("  Scoring with CLIP…")
    scorer = CLIPScorer(device=device)
    texts  = [p["text"] for p in predictions]
    scores = scorer.score(images_pil, texts)

    mean_score = sum(scores) / len(scores)
    variance   = sum((s - mean_score) ** 2 for s in scores) / max(len(scores) - 1, 1)
    std_score  = variance ** 0.5

    print(f"  CLIP-score: {mean_score:.2f} ± {std_score:.2f}  (n={len(scores)})")

    quality_scores = [
        BenchmarkScore(benchmark="stage_a_caption", metric="clip_score_mean",  value=round(mean_score, 4)),
        BenchmarkScore(benchmark="stage_a_caption", metric="clip_score_std",   value=round(std_score,  4)),
        BenchmarkScore(benchmark="stage_a_caption", metric="clip_score_n",     value=len(scores)),
    ]

    raw = {
        "experiment_id":    experiment_id,
        "model_key":        "experiment",
        "prompt":           EVAL_PROMPT,
        "clip_model":       "openai/clip-vit-large-patch14",
        "mean_clip_score":  round(mean_score, 4),
        "std_clip_score":   round(std_score, 4),
        "n":                len(scores),
        "predictions":      [
            {**p, "clip_score": round(s, 4)}
            for p, s in zip(predictions, scores)
        ],
    }

    # Free CLIP scorer
    del scorer

    return quality_scores, raw


# ── Device-ready flag ─────────────────────────────────────────────────────────

def _write_device_ready_flag(config: ExperimentConfig, experiment_id: str) -> Path:
    """
    Write a JSON flag file indicating this experiment config is ready for
    on-device performance measurement (iPhone 16 Pro).

    The flag includes the experiment_id and enough context for the human
    to identify the correct GGUF / model checkpoint to deploy.

    Returns the path of the flag file.
    """
    DEVICE_READY.mkdir(parents=True, exist_ok=True)
    flag_path = DEVICE_READY / f"{experiment_id[:12]}_ready.json"

    payload = {
        "experiment_id":   experiment_id,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "model_id":        config.model_id,
        "compression":     config.compression.model_dump(),
        "runtime_backend": config.runtime_backend,
        "target_device_id": config.target_device_id,
        "notes":           config.notes,
        "instructions": (
            "Deploy the model to the target device and run the VLMHarness "
            "measurement protocol (5 warm-up + 5 measured runs). "
            "Export the MetricsReport JSON and call "
            "ExperimentRunner.import_iphone_results(experiment_id, path)."
        ),
    }

    flag_path.write_text(json.dumps(payload, indent=2))
    return flag_path


# ── Experiment ledger ─────────────────────────────────────────────────────────

def _save_to_ledger(report: MetricsReport, config: ExperimentConfig) -> Path:
    """Persist MetricsReport JSON to the experiment ledger."""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    path = LEDGER_DIR / f"{report.experiment_id}.json"
    payload = {
        "report": json.loads(report.model_dump_json()),
        "config": json.loads(config.model_dump_json()),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# ── Main runner ───────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Runs Mac quality evaluation for a given ExperimentConfig and produces
    a MetricsReport. iPhone performance fields are populated later via
    import_iphone_results().

    Args:
        project_root:  Path to repo root. Auto-detected when None.
        mac_device:    PyTorch device for Mac eval ("mps", "cuda", "cpu").
        max_images:    Cap on Stage A images used for eval. None = all available.
                       Set to a small value (e.g. 5) for fast iteration during
                       development; use None for real experiment runs.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        mac_device: str | None = None,
        max_images: int | None = None,
    ):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.max_images   = max_images

        if mac_device is None:
            if torch.backends.mps.is_available():
                mac_device = "mps"
            elif torch.cuda.is_available():
                mac_device = "cuda"
            else:
                mac_device = "cpu"
        self.mac_device = mac_device

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, config: ExperimentConfig) -> MetricsReport:
        """
        Full pipeline:
          1. Validate config against known constraints
          2. Load Stage A eval items
          3. Load model with specified compression
          4. Run CLIP-score eval on Stage A caption images
          5. Persist predictions + MetricsReport to ledger
          6. Write iPhone-ready flag (for manual device deployment)
          7. Return MetricsReport

        Performance fields (ttft_ms, decode_tokens_per_sec, peak_memory_mb)
        are None in the returned report — populate via import_iphone_results().
        """
        started_at    = datetime.now(timezone.utc)
        experiment_id = config.content_hash()
        fingerprint   = _mac_hardware_fingerprint()

        print(f"\n{'═' * 60}")
        print(f"  ExperimentRunner.run()")
        print(f"  experiment_id : {experiment_id[:16]}…")
        print(f"  model_id      : {config.model_id}")
        print(f"  compression   : {config.compression.weight_dtype}"
              f"  backend={config.runtime_backend}")
        print(f"  device        : {self.mac_device}  ({fingerprint.os_version})")
        print(f"  available_ram : {fingerprint.available_ram_mb:.0f} MB")
        print(f"{'═' * 60}\n")

        model = processor = None
        try:
            # ── 1. Validate dataset hash ──────────────────────────────────
            if config.dataset_hash != STAGE_A_HASH:
                raise ValueError(
                    f"dataset_hash mismatch: config has {config.dataset_hash[:16]}…, "
                    f"runner expects {STAGE_A_HASH[:16]}…"
                )

            # ── 2. Load Stage A items ─────────────────────────────────────
            items = _load_stage_a_items(max_images=self.max_images)
            print(f"  Stage A: {len(items)} images found\n")

            # ── 3. Load model ─────────────────────────────────────────────
            effective_model_id = self._resolve_eval_model_id(config)
            load_fn, infer_fn  = _resolve_loader(effective_model_id)

            print(f"  Loading model: {effective_model_id}  "
                  f"(dtype={config.compression.weight_dtype})")
            t_load = time.perf_counter()
            model, processor = load_fn(
                effective_model_id,
                self.mac_device,
                config.compression.weight_dtype,
            )
            print(f"  Model loaded in {time.perf_counter() - t_load:.1f}s\n")

            # ── 4. CLIP eval ──────────────────────────────────────────────
            quality_scores, raw_preds = _run_clip_eval(
                model, processor, infer_fn, items,
                device=self.mac_device,
                experiment_id=experiment_id,
            )

            # ── 5. Free model memory ──────────────────────────────────────
            model = processor = None   # null-out before del so finally block is safe
            if self.mac_device == "mps":
                torch.mps.empty_cache()

            # ── 6. Build report ───────────────────────────────────────────
            report = MetricsReport(
                experiment_id=experiment_id,
                device_id=DEVICE_ID_MAC,
                status=ExperimentStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                hardware_fingerprint=fingerprint,
                quality_scores=quality_scores,
                # Performance fields are None — populated via import_iphone_results()
            )

            # ── 7. Persist ────────────────────────────────────────────────
            ledger_path = _save_to_ledger(report, config)
            print(f"\n  Ledger → {ledger_path}")

            # Save raw predictions alongside the ledger entry
            preds_path = LEDGER_DIR / f"{experiment_id}_preds.json"
            preds_path.write_text(json.dumps(raw_preds, indent=2))
            print(f"  Preds  → {preds_path}")

            # ── 8. iPhone-ready flag ──────────────────────────────────────
            flag_path = _write_device_ready_flag(config, experiment_id)
            print(f"  Ready  → {flag_path}")

            return report

        except MemoryError:
            return self._failure_report(
                experiment_id, ExperimentStatus.OOM,
                "Out of memory during model load or inference",
                started_at, fingerprint,
            )
        except Exception:
            return self._failure_report(
                experiment_id, ExperimentStatus.FAILED,
                traceback.format_exc().strip(),
                started_at, fingerprint,
            )
        finally:
            if model is not None:
                try:
                    del model, processor
                    if self.mac_device == "mps":
                        torch.mps.empty_cache()
                except Exception:
                    pass

    def import_iphone_results(
        self,
        experiment_id: str,
        iphone_json_path: Path,
    ) -> MetricsReport:
        """
        Merge iPhone performance measurements into an existing ledger entry.

        iphone_json_path: path to the MetricsReport JSON exported by VLMHarness
        on the device (the JSON contains ttft_ms, decode_tokens_per_sec,
        peak_memory_mb, etc.).

        Returns an updated MetricsReport with both quality_scores (from Mac
        eval) and performance fields (from iPhone) populated.

        Overwrites the ledger entry in place.
        """
        ledger_path = LEDGER_DIR / f"{experiment_id}.json"
        if not ledger_path.exists():
            raise FileNotFoundError(
                f"No ledger entry for experiment_id={experiment_id[:16]}…\n"
                f"Run ExperimentRunner.run(config) first."
            )

        ledger = json.loads(ledger_path.read_text())
        existing_report = MetricsReport(**ledger["report"])

        iphone_data = json.loads(Path(iphone_json_path).read_text())

        # Extract performance fields from the VLMHarness export format
        perf = iphone_data.get("performance", iphone_data)   # tolerate both flat and nested
        ttft_ms          = perf.get("ttft_ms") or perf.get("ttft_ms_mean")
        tps              = perf.get("decode_tokens_per_sec_mean") or perf.get("tps")
        peak_mem_mb      = perf.get("peak_memory_mb_mean") or perf.get("peak_memory_mb")
        on_disk_mb       = perf.get("on_disk_size_mb")

        # iPhone hardware fingerprint (best-effort from the JSON)
        iphone_fp = HardwareFingerprint(
            os_version=iphone_data.get("os_version", "iOS (unknown)"),
            available_ram_mb=iphone_data.get("available_ram_mb", 0.0),
            thermal_state=iphone_data.get("thermal_state"),
        )

        updated_report = MetricsReport(
            experiment_id=existing_report.experiment_id,
            device_id=iphone_data.get("device_id", "iphone16pro-001"),
            status=ExperimentStatus.COMPLETED,
            started_at=existing_report.started_at,
            completed_at=datetime.now(timezone.utc),
            hardware_fingerprint=iphone_fp,
            ttft_ms=ttft_ms,
            decode_tokens_per_sec=tps,
            peak_memory_mb=peak_mem_mb,
            on_disk_size_mb=on_disk_mb,
            quality_scores=existing_report.quality_scores,
        )

        # Re-save to ledger
        config_data = ledger["config"]
        ledger["report"] = json.loads(updated_report.model_dump_json())
        ledger["iphone_merged_at"] = datetime.now(timezone.utc).isoformat()
        ledger_path.write_text(json.dumps(ledger, indent=2))

        print(f"  Merged iPhone results into {ledger_path}")
        print(f"  TTFT={ttft_ms}ms  TPS={tps}  Mem={peak_mem_mb}MB")

        return updated_report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_eval_model_id(self, config: ExperimentConfig) -> str:
        """
        Determine which model_id to use for Mac quality eval.

        - pytorch_mps + fp16/fp32 → use config.model_id directly (HF or local)
        - pytorch_mps + int4      → model_id must be a local path to AWQ checkpoint
        - llamacpp_gguf + any     → use fp16 HF equivalent for quality eval
                                    (GGUF performance measured on iPhone)
        """
        backend      = config.runtime_backend
        weight_dtype = config.compression.weight_dtype

        if backend == "llamacpp_gguf":
            # Quality eval uses the PyTorch fp16 model as proxy.
            # The GGUF runtime affects performance (TTFT, TPS) not description quality
            # for the compressed sizes we test in Phase 1.
            print(
                f"  [llamacpp_gguf backend] Quality eval will use PyTorch fp16 proxy "
                f"for {config.model_id}. iPhone perf measured separately with GGUF."
            )
            return config.model_id

        if backend == "pytorch_mps":
            if weight_dtype == "int4":
                # Expect model_id to point to a pre-quantized AWQ checkpoint on disk
                local_path = Path(config.model_id)
                if not local_path.exists():
                    raise FileNotFoundError(
                        f"INT4 eval requires a pre-quantized AWQ checkpoint at model_id path. "
                        f"Path not found: {config.model_id}\n"
                        f"Run AWQ quantization first:\n"
                        f"  from autoawq import AutoAWQForCausalLM; ..."
                    )
                return config.model_id
            # fp16 / fp32 / bfloat16 → use model_id directly
            return config.model_id

        raise NotImplementedError(
            f"Backend '{backend}' is not supported by ExperimentRunner in Phase 1. "
            f"Supported: pytorch_mps, llamacpp_gguf."
        )

    def _failure_report(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        error_message: str,
        started_at: datetime,
        fingerprint: HardwareFingerprint,
    ) -> MetricsReport:
        report = MetricsReport(
            experiment_id=experiment_id,
            device_id=DEVICE_ID_MAC,
            status=status,
            error_message=error_message[:2000],   # cap length
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            hardware_fingerprint=fingerprint,
        )
        # Save failure to ledger too (useful for debugging)
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        fail_path = LEDGER_DIR / f"{experiment_id}_FAILED.json"
        fail_path.write_text(report.model_dump_json(indent=2))
        print(f"\n  ❌ {status}: {error_message.splitlines()[-1][:80]}")
        print(f"  Saved to {fail_path}")
        return report
