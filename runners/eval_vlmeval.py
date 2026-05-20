#!/usr/bin/env python3
"""
runners/eval_vlmeval.py — Task 2.2 quality evaluation runner.

Runs each reference model over 100-example slices of three image benchmarks
(POPE, RealWorldQA, MMBench_DEV_EN) using VLMEvalKit for data loading and
scoring. Writes one MetricsReport JSON per model × benchmark combination.

Usage (from repo root):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python runners/eval_vlmeval.py
    PYTORCH_ENABLE_MPS_FALLBACK=1 python runners/eval_vlmeval.py \\
        --models Qwen2.5-VL-3B SmolVLM-500M \\
        --benchmarks POPE RealWorldQA \\
        --n-samples 100

Requirements:
    pip install torch transformers accelerate pillow psutil qwen_vl_utils
    VLMEvalKit vendored at vendor/VLMEvalKit (already done)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd
import psutil
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "VLMEvalKit"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from schemas import ExperimentConfig, MetricsReport
from schemas.experiments import (
    BenchmarkScore,
    CompressionSpec,
    DecodeStrategy,
    ExperimentStatus,
    HardwareFingerprint,
    WeightDtype,
)
from schemas.devices import Runtime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE_ID = "mac_mini_m4_16gb"

# Models: (vlmeval_key, hf_model_id, weight_dtype, notes)
MODEL_REGISTRY: dict[str, dict] = {
    "LFM2-VL-450M": {
        "hf_id": "LiquidAI/LFM2-VL-450M",
        "family": "lfm2vl",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
    },
    "SmolVLM-500M": {
        "hf_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "family": "smolvlm",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
    },
    "MiniCPM-V-4_5": {
        "hf_id": "openbmb/MiniCPM-V-4_5",
        "family": "minicpm",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
    },
    "Qwen2.5-VL-3B": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2_5_vl",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
    },
    "FastVLM-0.5B": {
        "hf_id": "apple/FastVLM-0.5B",
        "family": "fastvlm",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
    },
}

BENCHMARKS = ["POPE", "RealWorldQA", "MMBench_DEV_EN"]

# Prompt suffixes that help models produce parseable answers.
POPE_PROMPT_SUFFIX = " Please answer Yes or No only."
MCQ_PROMPT_SUFFIX = " Answer with only the letter A, B, C, or D."


# ---------------------------------------------------------------------------
# MPS model base
# ---------------------------------------------------------------------------

class MPSModel(Protocol):
    model_key: str

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        ...

    def unload(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Model implementations
# ---------------------------------------------------------------------------

def _mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available — requires macOS 12.3+ with Apple Silicon")
    return torch.device("mps")


class Qwen25VLModel:
    """Qwen2.5-VL-3B-Instruct on MPS (float16)."""

    model_key = "Qwen2.5-VL-3B"

    def __init__(self, hf_id: str) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        device = _mps_device()
        print(f"  Loading {hf_id} …")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            local_files_only=False,
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.device = device

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = question + (MCQ_PROMPT_SUFFIX if is_mcq else POPE_PROMPT_SUFFIX)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
            )
        n_in = inputs["input_ids"].shape[1]
        return self.processor.decode(out[0][n_in:], skip_special_tokens=True).strip()

    def unload(self) -> None:
        del self.model
        torch.mps.empty_cache()


class SmolVLMModel:
    """SmolVLM-500M-Instruct on MPS."""

    model_key = "SmolVLM-500M"

    def __init__(self, hf_id: str) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        device = _mps_device()
        print(f"  Loading {hf_id} …")
        self.model = AutoModelForVision2Seq.from_pretrained(
            hf_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.device = device

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = question + (MCQ_PROMPT_SUFFIX if is_mcq else POPE_PROMPT_SUFFIX)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=text, images=[image], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        return self.processor.decode(out[0][n_in:], skip_special_tokens=True).strip()

    def unload(self) -> None:
        del self.model
        torch.mps.empty_cache()


class MiniCPMVModel:
    """MiniCPM-V-4_5 on MPS (trust_remote_code)."""

    model_key = "MiniCPM-V-4_5"

    def __init__(self, hf_id: str) -> None:
        from transformers import AutoModel, AutoTokenizer
        device = _mps_device()
        print(f"  Loading {hf_id} …")
        self.model = AutoModel.from_pretrained(
            hf_id, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.device = device

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = question + (MCQ_PROMPT_SUFFIX if is_mcq else POPE_PROMPT_SUFFIX)
        msgs = [{"role": "user", "content": [image, prompt]}]
        try:
            res = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        except Exception:
            # Some MPS ops fall back; retry with sampling disabled explicitly
            res = self.model.chat(
                image=None, msgs=msgs, tokenizer=self.tokenizer,
                sampling=False,
            )
        return str(res).strip()

    def unload(self) -> None:
        del self.model
        torch.mps.empty_cache()


class LFM2VLModel:
    """LFM2-VL-450M on MPS (float16, AutoModelForImageTextToText)."""

    model_key = "LFM2-VL-450M"

    def __init__(self, hf_id: str) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        device = _mps_device()
        print(f"  Loading {hf_id} …")
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, attn_implementation="sdpa",
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.device = device

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        image = Image.open(image_path).convert("RGB")
        suffix = (
            "\nAnswer with only the letter A, B, C, or D."
            if is_mcq else
            "\nPlease answer Yes or No only."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": question + suffix},
        ]}]
        chat_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            images=[image], text=[chat_text], return_tensors="pt",
        ).to(dtype=torch.float16, device=self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=32, use_cache=True)
        decoded = self.processor.decode(out[0], skip_special_tokens=False)
        response = decoded.split("<|im_start|>assistant\n")[-1].strip()
        return response.replace("<|im_end|>", "").strip()

    def unload(self) -> None:
        del self.model
        torch.mps.empty_cache()


class FastVLMModel:
    """Apple FastVLM-0.5B — loaded generically; skipped if weights unavailable."""

    model_key = "FastVLM-0.5B"

    def __init__(self, hf_id: str) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor
        device = _mps_device()
        print(f"  Loading {hf_id} …")
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.device = device

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = question + (MCQ_PROMPT_SUFFIX if is_mcq else POPE_PROMPT_SUFFIX)
        inputs = self.processor(images=[image], text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        return self.processor.decode(out[0][n_in:], skip_special_tokens=True).strip()

    def unload(self) -> None:
        del self.model
        torch.mps.empty_cache()


_FAMILY_TO_CLASS = {
    "qwen2_5_vl": Qwen25VLModel,
    "smolvlm": SmolVLMModel,
    "minicpm": MiniCPMVModel,
    "lfm2vl": LFM2VLModel,
    "fastvlm": FastVLMModel,
}


def load_model(model_key: str) -> MPSModel:
    cfg = MODEL_REGISTRY[model_key]
    cls = _FAMILY_TO_CLASS[cfg["family"]]
    return cls(cfg["hf_id"])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_dataset_slice(dataset_name: str, n: int):
    """Return (dataset_obj, sliced_dataframe)."""
    from vlmeval.dataset import build_dataset
    ds = build_dataset(dataset_name)
    df = ds.data.head(n).copy()
    return ds, df


def get_image_path(ds, row: pd.Series, dataset_name: str) -> str:
    """Return local path to the image for this row, extracting from TSV if needed."""
    if "image_path" in row and pd.notna(row["image_path"]):
        return str(row["image_path"])
    # base64 image stored in TSV — dump to LMUData images dir
    path = ds.dump_image(row, dataset_name)
    return str(path)


def build_mcq_question(row: pd.Series) -> tuple[str, bool]:
    """Return (full_question, is_mcq)."""
    q = str(row["question"])
    options = []
    for letter in ("A", "B", "C", "D"):
        if letter in row and pd.notna(row[letter]) and str(row[letter]).strip().lower() != "nan":
            options.append(f"{letter}. {row[letter]}")
    if "hint" in row and pd.notna(row.get("hint")):
        hint = str(row["hint"]).strip()
        if hint and hint.lower() != "nan":
            q = hint + "\n" + q
    if options:
        q = q + "\n" + "\n".join(options)
        return q, True
    return q, False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_benchmark(ds, result_df: pd.DataFrame, eval_dir: Path, tag: str) -> dict[str, float]:
    """
    Write result_df to a temp file and call VLMEvalKit's evaluate().
    Returns dict of metric_name → value.
    """
    from vlmeval.smp import dump, load
    eval_file = eval_dir / f"{tag}_predictions.xlsx"
    result_df.to_excel(eval_file, index=False)

    try:
        scores = ds.evaluate(str(eval_file), model="exact_matching", nproc=1)
        if scores is None:
            return {}
        if isinstance(scores, dict):
            return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        if hasattr(scores, "to_dict"):
            return {k: float(v) for k, v in scores.iloc[-1].items()
                    if isinstance(v, (int, float))}
    except Exception as exc:
        print(f"  WARNING: VLMEvalKit evaluate() failed ({exc}); computing accuracy manually.")
        # Fallback: exact match accuracy
        if "answer" in result_df.columns and "prediction" in result_df.columns:
            correct = (
                result_df["prediction"].str.strip().str.upper() ==
                result_df["answer"].str.strip().str.upper()
            ).sum()
            return {"accuracy": correct / len(result_df)}
    return {}


# ---------------------------------------------------------------------------
# Hardware snapshot
# ---------------------------------------------------------------------------

def _hardware_fingerprint() -> HardwareFingerprint:
    import platform
    ver = platform.mac_ver()[0]
    return HardwareFingerprint(
        os_version=f"macOS {ver}" if ver else platform.platform(),
        available_ram_mb=psutil.virtual_memory().available / (1024 * 1024),
        swap_used_mb=psutil.swap_memory().used / (1024 * 1024),
    )


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def _dataset_hash_from_indices(indices) -> str:
    h = hashlib.sha256()
    for idx in sorted(indices):
        h.update(str(idx).encode())
        h.update(b"\n")
    return h.hexdigest()


def run_eval(
    model_keys: list[str],
    benchmark_names: list[str],
    n_samples: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_scratch = output_dir / "_eval_scratch"
    eval_scratch.mkdir(exist_ok=True)

    all_reports: list[dict] = []

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        # Load model (once per model, shared across benchmarks)
        try:
            model = load_model(model_key)
        except Exception as exc:
            print(f"  SKIP: could not load model — {exc}")
            continue

        cfg = MODEL_REGISTRY[model_key]

        for bench_name in benchmark_names:
            print(f"\n  Benchmark: {bench_name} ({n_samples} samples)")

            # Load dataset slice
            try:
                ds, df = get_dataset_slice(bench_name, n_samples)
            except Exception as exc:
                print(f"  SKIP: dataset load failed — {exc}")
                continue

            # Build ExperimentConfig for this (model, benchmark) pair
            dataset_hash = _dataset_hash_from_indices(df["index"].tolist())
            exp_cfg = ExperimentConfig(
                model_id=cfg["hf_id"],
                compression=CompressionSpec(weight_dtype=cfg["dtype"]),
                runtime_backend=cfg["runtime"],
                decode_strategy=DecodeStrategy.GREEDY,
                dataset_hash=dataset_hash,
                target_device_id=DEVICE_ID,
                notes=f"Task 2.2 quality eval — {bench_name} {n_samples}-example slice",
            )
            experiment_id = exp_cfg.content_hash()

            # Inference loop
            started_at = datetime.now(timezone.utc)
            predictions: list[str] = []
            failed = 0

            for i, (_, row) in enumerate(df.iterrows()):
                if i % 10 == 0:
                    print(f"    [{i}/{n_samples}] …", end="\r", flush=True)
                try:
                    img_path = get_image_path(ds, row, bench_name)
                    question, is_mcq = build_mcq_question(row)
                    pred = model.infer(img_path, question, is_mcq)
                except Exception as exc:
                    print(f"\n    WARNING row {row.get('index', i)}: {exc}")
                    pred = ""
                    failed += 1
                predictions.append(pred)

            print(f"    [{n_samples}/{n_samples}] done  ({failed} errors)")
            completed_at = datetime.now(timezone.utc)

            # Build result DataFrame for VLMEvalKit scoring
            result_df = df.copy()
            result_df["prediction"] = predictions

            # Score
            tag = f"{model_key.replace('/', '_')}_{bench_name}"
            scores = score_benchmark(ds, result_df, eval_scratch, tag)
            print(f"    Scores: {scores}")

            # Build MetricsReport
            quality_scores = [
                BenchmarkScore(benchmark=bench_name, metric=k, value=v)
                for k, v in scores.items()
            ]
            status = ExperimentStatus.COMPLETED if failed < n_samples else ExperimentStatus.FAILED
            report = MetricsReport(
                experiment_id=experiment_id,
                device_id=DEVICE_ID,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                hardware_fingerprint=_hardware_fingerprint(),
                quality_scores=quality_scores,
            )

            # Save report
            out_file = output_dir / f"{tag}.json"
            out_file.write_text(report.model_dump_json(indent=2))
            cfg_file = output_dir / f"{tag}_config.json"
            cfg_file.write_text(exp_cfg.model_dump_json(indent=2))
            print(f"    Saved: {out_file.name}")
            all_reports.append({"model": model_key, "benchmark": bench_name, "scores": scores})

        model.unload()

    # Summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Model':<22} {'Benchmark':<20} {'Scores'}")
    print("-" * 60)
    for r in all_reports:
        scores_str = "  ".join(f"{k}={v:.3f}" for k, v in r["scores"].items())
        print(f"{r['model']:<22} {r['benchmark']:<20} {scores_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2.2 VLMEvalKit quality eval (MPS)")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to evaluate (default: all 5)",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=BENCHMARKS,
        choices=BENCHMARKS,
        help="Benchmarks to run (default: all 3)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of examples per benchmark slice (default: 100)",
    )
    parser.add_argument(
        "--output", default="results/eval_task_2_2",
        help="Output directory for MetricsReport JSON files",
    )
    args = parser.parse_args()

    print(f"Models:     {args.models}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"N samples:  {args.n_samples}")
    print(f"Output:     {args.output}")

    run_eval(
        model_keys=args.models,
        benchmark_names=args.benchmarks,
        n_samples=args.n_samples,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
