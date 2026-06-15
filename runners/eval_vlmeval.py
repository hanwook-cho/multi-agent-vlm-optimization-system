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
    "MiniCPM-V-4.6": {
        "hf_id": "openbmb/MiniCPM-V-4.6",
        "family": "minicpm46",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
    },
    "Qwen2.5-VL-3B": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2_5_vl",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
    },
    "Qwen2.5-VL-3B-Q4_K_M": {
        # P2-1.3: the actual on-device GGUF bundle (text Q4_K_M + mmproj F16),
        # evaluated via llama.cpp/mtmd rather than HF fp16. Produced by
        # scripts/convert_qwen25vl_gguf.sh.
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2_5_vl_gguf",
        "dtype": WeightDtype.INT4,
        "runtime": Runtime.LLAMACPP_GGUF,
        "model_path": "models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Q4_K_M.gguf",
        "mmproj_path": "models/qwen2.5-vl-3b-gguf/mmproj-Qwen2.5-VL-3B-f16.gguf",
    },
    "Qwen2.5-VL-3B-F16-GGUF": {
        # P2-1.3 control: F16 GGUF via the SAME llama.cpp/mtmd path as Q4_K_M.
        # Lets us decompose the GGUF-vs-fp16 delta into runtime effect (transformers
        # fp16 vs F16 GGUF) and pure quantization effect (F16 GGUF vs Q4_K_M GGUF).
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2_5_vl_gguf",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.LLAMACPP_GGUF,
        "model_path": "models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-f16.gguf",
        "mmproj_path": "models/qwen2.5-vl-3b-gguf/mmproj-Qwen2.5-VL-3B-f16.gguf",
    },
    "FastVLM-0.5B": {
        "hf_id": "apple/FastVLM-0.5B",
        "family": "fastvlm",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
    },
    "LFM2-VL-450M-distill": {
        # Phase 2 P2-D2 student: base LFM2-VL-450M + LoRA adapter distilled from the
        # Qwen2.5-VL-3B teacher using TASK-ALIGNED Q&A targets (11.2K pairs) + 20%
        # caption rehearsal. Supersedes the P2-D1 caption-only pilot
        # (artifacts/students/lfm2vl_distill_pilot_s0), which regressed POPE 86.2->38.5.
        # Evaluated on the SAME fp16 transformers path as the LFM2-VL-450M baseline.
        "hf_id": "LiquidAI/LFM2-VL-450M",
        "family": "lfm2vl_distill",
        "dtype": WeightDtype.FP16,
        "runtime": Runtime.PYTORCH_MPS,
        "adapter_path": "artifacts/students/lfm2vl_qa_distill_s0/adapter",
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
    """SmolVLM-500M-Instruct on MPS.

    transformers 5.x dropped AutoModelForVision2Seq; use SmolVLMForConditionalGeneration
    directly.  AutoProcessor resolves to Idefics3Processor (same interface).
    """

    model_key = "SmolVLM-500M"

    def __init__(self, hf_id: str) -> None:
        from transformers import SmolVLMForConditionalGeneration, AutoProcessor
        device = _mps_device()
        print(f"  Loading {hf_id} …")
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
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
    """MiniCPM-V (4.6 / 4.5) on MPS (trust_remote_code).

    transformers 5.x _finalize_model_loading calls self.all_tied_weights_keys which is
    set as an instance attribute in PreTrainedModel.__init__.  MiniCPMV's custom
    __init__ doesn't call super().__init__ in all paths, so the attribute is missing.
    Patch PreTrainedModel._move_missing_keys_from_meta_to_device to handle this case.
    """

    model_key = "MiniCPM-V-4.6"

    def __init__(self, hf_id: str) -> None:
        from transformers import AutoModel, AutoTokenizer, PreTrainedModel
        device = _mps_device()
        print(f"  Loading {hf_id} …")

        # Patch for transformers 5.x compatibility: ensure all_tied_weights_keys exists
        _orig_move = PreTrainedModel._move_missing_keys_from_meta_to_device

        def _patched_move(self_model, missing_and_mismatched, *args, **kwargs):
            if not hasattr(self_model, "all_tied_weights_keys"):
                self_model.all_tied_weights_keys = {}
            return _orig_move(self_model, missing_and_mismatched, *args, **kwargs)

        PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move
        try:
            self.model = AutoModel.from_pretrained(
                hf_id, trust_remote_code=True,
                torch_dtype=torch.float16, low_cpu_mem_usage=True,
            ).to(device).eval()
        finally:
            PreTrainedModel._move_missing_keys_from_meta_to_device = _orig_move

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.device = device

        # model.chat() creates self.processor internally via AutoProcessor if none is set.
        # That processor's tokenizer is a DIFFERENT (unpatched) instance.  Pre-build the
        # processor here, patch its tokenizer, and assign to self.model.processor so
        # chat() picks it up directly.
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

        # Patch for transformers 5.x: TokenizersBackend no longer exposes these attributes
        # via __getattr__; set as instance attributes so __dict__ lookup finds them first.
        #
        # MiniCPM-V image boundary tokens (from preprocessor_config.json):
        #   im_start_token  = "<image>"   → id 151669
        #   im_end_token    = "</image>"  → id 151670
        #   slice_start     = "<slice>"   → id 151679
        #   slice_end       = "</slice>"  → id 151680
        #   bos_id          = bos_token_id (= <|im_start|> = 151644 in Qwen2 tokenizer)
        _MINICPM_TOKEN_PATCHES = [
            ("im_start_id",    "<image>"),
            ("im_end_id",      "</image>"),
            ("slice_start_id", "<slice>"),
            ("slice_end_id",   "</slice>"),
            ("bos_id",         None),   # special: set to bos_token_id
        ]

        def _patch_tokenizer(tok: object) -> None:
            for name, token_str in _MINICPM_TOKEN_PATCHES:
                if not hasattr(tok, name):
                    if token_str is None:
                        tid = tok.bos_token_id
                    else:
                        tid = tok.convert_tokens_to_ids(token_str)
                        if tid == tok.unk_token_id:
                            tid = tok.bos_token_id  # safe fallback
                    setattr(tok, name, tid)

        _patch_tokenizer(processor.tokenizer)  # for input encoding (processor path)
        _patch_tokenizer(self.tokenizer)        # for output decoding (_decode_text path)
        self.model.processor = processor

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


class LFM2VLDistilledModel(LFM2VLModel):
    """LFM2-VL-450M + a Strategy B distillation LoRA adapter (Phase 2 student).

    Loads the base model exactly as LFM2VLModel (same fp16 transformers/MPS path)
    and applies the LoRA adapter, so a comparison against the LFM2-VL-450M baseline
    is on the SAME inference path (P2-1.3 methodology). infer()/unload() inherited.
    """

    model_key = "LFM2-VL-450M-distill"

    def __init__(self, hf_id: str, adapter_path: str) -> None:
        super().__init__(hf_id)  # base fp16 + processor — identical to the baseline
        from peft import PeftModel
        print(f"  applying LoRA adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.to(dtype=torch.float16, device=self.device).eval()


class FastVLMModel:
    """Apple FastVLM-0.5B — LLaVA-QWen2 architecture with MobileCLIP vision tower.

    apple/FastVLM-0.5B uses trust_remote_code with llava_qwen.py (model_type=llava_qwen2).
    AutoProcessor fails if preprocessor_config.json is absent from the HF hub cache; we
    resolve the snapshot directory locally and load from that path instead.
    """

    model_key = "FastVLM-0.5B"

    def __init__(self, hf_id: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
        from huggingface_hub import snapshot_download
        device = _mps_device()
        print(f"  Loading {hf_id} …")
        # Resolve local snapshot path (avoids preprocessor_config.json hub-lookup issue)
        local_path = snapshot_download(hf_id, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        # Build CLIPImageProcessor with MobileCLIP-L parameters (image_size=1024,
        # mean=0.0, std=1.0 as defined in llava_qwen.py MobileClipVisionTower).
        self.image_processor = CLIPImageProcessor(
            crop_size={"height": 1024, "width": 1024},
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            size={"shortest_edge": 1024},
        )
        self._image_token_index = -200   # IMAGE_TOKEN_INDEX from llava_qwen.py
        self._image_token = "<image>"
        self.device = device

    def _tokenizer_image_token(self, prompt: str) -> torch.Tensor:
        """Tokenize prompt inserting IMAGE_TOKEN_INDEX (-200) where <image> appears."""
        chunks = prompt.split(self._image_token)
        ids: list[int] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                ids.append(self._image_token_index)
            ids.extend(self.tokenizer(chunk, add_special_tokens=(i == 0)).input_ids)
        return torch.tensor([ids], dtype=torch.long)

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        image = Image.open(image_path).convert("RGB")
        suffix = MCQ_PROMPT_SUFFIX if is_mcq else POPE_PROMPT_SUFFIX
        # Build prompt with image token at the start (LLaVA convention)
        prompt = (
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{self._image_token}\n{question}{suffix}"}],
                tokenize=False, add_generation_prompt=True,
            )
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template
            else f"{self._image_token}\n{question}{suffix}"
        )
        input_ids = self._tokenizer_image_token(prompt).to(self.device)
        pixel_values = self.image_processor(images=[image], return_tensors="pt"
                                            )["pixel_values"].to(dtype=torch.float16, device=self.device)
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                images=pixel_values,
                max_new_tokens=32,
                do_sample=False,
                # repetition_penalty only for MCQ: for POPE the suffix contains
                # "Yes" and "No" which the penalty would suppress in the output.
                repetition_penalty=1.2 if is_mcq else 1.0,
            )
        # FastVLM's generate() uses inputs_embeds internally (LLaVA path), so out[0]
        # contains only the generated tokens — do NOT slice off input length.
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # Return first non-empty line so POPE/MCQ scorers see a clean token.
        # FastVLM tends to output the answer letter first then verbose text;
        # can_infer_option requires the letter near the END of the string, so
        # trimming to the first line is essential for correct scoring.
        for line in decoded.splitlines():
            line = line.strip()
            if line:
                return line
        return decoded

    def unload(self) -> None:
        del self.model
        torch.mps.empty_cache()


class MiniCPMV46Model:
    """MiniCPM-V-4.6 (1.3B) on MPS — native transformers 5.x model.

    Uses MiniCPMV4_6ForConditionalGeneration + MiniCPMV4_6Processor.
    No trust_remote_code needed; standard generate() API.
    Chat template adds <think>...</think> preamble via add_generation_prompt=True;
    generated text starts immediately after so out[0][n_in:] is the answer.
    """

    model_key = "MiniCPM-V-4.6"

    def __init__(self, hf_id: str) -> None:
        from transformers import MiniCPMV4_6ForConditionalGeneration, AutoProcessor
        from huggingface_hub import snapshot_download
        device = _mps_device()
        print(f"  Loading {hf_id} …")
        local_path = snapshot_download(hf_id, local_files_only=True)
        self.model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
            local_path, dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(local_path)
        self.device = device

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        image = Image.open(image_path).convert("RGB")
        suffix = MCQ_PROMPT_SUFFIX if is_mcq else POPE_PROMPT_SUFFIX
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question + suffix},
        ]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=[image], text=text, return_tensors="pt").to(self.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        decoded = self.processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        # Return first non-empty line so POPE/MCQ scorers see a clean token
        for line in decoded.splitlines():
            line = line.strip()
            if line:
                return line
        return decoded

    def unload(self) -> None:
        del self.model
        torch.mps.empty_cache()


class Qwen25VLGGUFModel:
    """Qwen2.5-VL-3B Q4_K_M GGUF via llama.cpp/mtmd — the on-device inference path.

    Unlike the other classes (HuggingFace fp16 on MPS), this evaluates the actual
    quantized GGUF bundle (text Q4_K_M + vision mmproj F16) through llama.cpp, so
    P2-1.3 measures the deployed artifact, not an fp16 proxy.

    Runs a local llama-server once with the multimodal model loaded, then answers
    each (image, question) via the OpenAI-compatible /v1/chat/completions endpoint
    with the image inlined as a base64 data URI. --image-min-tokens 1024 is passed
    because Qwen-VL needs >=1024 image tokens for grounding accuracy (llama.cpp
    load warning); omitting it would understate quality.
    """

    model_key = "Qwen2.5-VL-3B-Q4_K_M"

    def __init__(self, model_path: str, mmproj_path: str, port: int = 8081) -> None:
        import subprocess
        import time
        import urllib.request

        server_bin = ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"
        if not server_bin.exists():
            raise RuntimeError(f"llama-server not built at {server_bin}")
        for p in (model_path, mmproj_path):
            if not Path(p).exists():
                raise RuntimeError(f"GGUF not found: {p} (run scripts/convert_qwen25vl_gguf.sh)")

        self.port = port
        print(f"  Starting llama-server (mtmd) on :{port} …")
        self._proc = subprocess.Popen(
            [str(server_bin), "-m", str(model_path), "--mmproj", str(mmproj_path),
             "--port", str(port), "-c", "4096", "-ngl", "999",
             "--image-min-tokens", "1024"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        health = f"http://localhost:{port}/health"
        for _ in range(180):
            try:
                if urllib.request.urlopen(health, timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            self.unload()
            raise RuntimeError("llama-server did not become healthy in time")

        import openai
        self.client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="local")

    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        import base64
        suffix = MCQ_PROMPT_SUFFIX if is_mcq else POPE_PROMPT_SUFFIX
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else (ext or "jpeg")
        resp = self.client.chat.completions.create(
            model="qwen2.5-vl-3b",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                {"type": "text", "text": question + suffix},
            ]}],
            max_tokens=32, temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return text

    def unload(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


_FAMILY_TO_CLASS = {
    "qwen2_5_vl": Qwen25VLModel,
    "qwen2_5_vl_gguf": Qwen25VLGGUFModel,
    "smolvlm": SmolVLMModel,
    "minicpm": MiniCPMVModel,
    "minicpm46": MiniCPMV46Model,
    "lfm2vl": LFM2VLModel,
    "lfm2vl_distill": LFM2VLDistilledModel,
    "fastvlm": FastVLMModel,
}


def load_model(model_key: str) -> MPSModel:
    cfg = MODEL_REGISTRY[model_key]
    cls = _FAMILY_TO_CLASS[cfg["family"]]
    if cfg["family"] == "qwen2_5_vl_gguf":
        return cls(str(ROOT / cfg["model_path"]), str(ROOT / cfg["mmproj_path"]))
    if cfg["family"] == "lfm2vl_distill":
        return cls(cfg["hf_id"], str(ROOT / cfg["adapter_path"]))
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
    # dump_image() returns a list of paths; take the first element
    paths = ds.dump_image(row)
    if isinstance(paths, list):
        return str(paths[0])
    return str(paths)


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
            # Drop base64 image column — it bloats the Excel file beyond xlsx limits
            result_df = result_df.drop(columns=["image"], errors="ignore")

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
            # quality_scores must be empty when status != completed (schema constraint)
            if status != ExperimentStatus.COMPLETED:
                quality_scores = []
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
