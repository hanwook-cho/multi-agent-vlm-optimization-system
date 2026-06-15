# Third-Party Components

This project **uses but does not redistribute** the models, datasets, and tools below.
Model weights and dataset images are fetched by the user from their original sources at
run time (see `.gitignore` — weights, GGUFs, and dataset photos are never committed).

> **Licenses are listed best-effort and may change.** Always verify the current license
> on the source page before redistributing or using a component commercially. This file
> is informational, not legal advice.

## Models

| Model | Used as | Source | License (verify at source) |
|---|---|---|---|
| LFM2-VL-450M | Phase-2 **benchmark** / yardstick | `LiquidAI/LFM2-VL-450M` (HF) | Liquid AI open license (LFM Open License) |
| SmolVLM-500M-Instruct | reference baseline | `HuggingFaceTB/SmolVLM-500M-Instruct` (HF) | Apache-2.0 |
| MiniCPM-V-4.6 | reference baseline | `openbmb/MiniCPM-V-4.6` (HF) | MiniCPM Model License (custom) |
| FastVLM-0.5B | reference baseline | `apple/FastVLM-0.5B` (HF) | Apple ML research license (custom) |
| Qwen2.5-VL-3B-Instruct | Phase-2 **teacher** | `Qwen/Qwen2.5-VL-3B-Instruct` (HF) | Apache-2.0 |
| Qwen2.5-0.5B-Instruct | constructed-student **LM backbone** | `Qwen/Qwen2.5-0.5B-Instruct` (HF) | Apache-2.0 |
| SigLIP (base patch16-224) | constructed-student **vision encoder** | `google/siglip-base-patch16-224` (HF) | Apache-2.0 |
| Qwen2.5-7B-Instruct (GGUF) | **Search Strategist** local LLM | `Qwen/Qwen2.5-7B-Instruct`, GGUF via `bartowski` | Apache-2.0 |

## Datasets

| Dataset | Used for | Source | License (verify at source) |
|---|---|---|---|
| COCO (val2017 / train2017) | Stage-A eval-set images; teacher caption/QA caches | cocodataset.org | Images: Flickr terms; annotations: CC BY 4.0 |
| Flickr30k, Open Images | considered/optional image sources (see ADR-0004) | respective sites | per source |

## Evaluation benchmarks

`POPE`, `RealWorldQA`, and `MMBench_DEV_EN` are accessed through **VLMEvalKit**, which
downloads them from their original hosts. Each benchmark carries its own license/terms —
verify at the benchmark source. RealWorldQA is from xAI; MMBench from the OpenCompass
project; POPE from its authors.

## Libraries & tools

| Component | Role | License |
|---|---|---|
| VLMEvalKit (vendored under `vendor/`, patched) | benchmark data loading + scoring | Apache-2.0 |
| llama.cpp (built separately, not committed) | GGUF inference; Strategist + teacher serving | MIT |
| PyTorch, Transformers, PEFT | training / eval / LoRA | BSD-3 / Apache-2.0 |
| Streamlit, Plotly, pandas | operator console + dashboards | Apache-2.0 / MIT / BSD-3 |
| Pydantic, jsonschema, PyYAML, pytest | schemas, validation, tests | MIT / Apache-2.0 |

## This project's own license

The code in this repository is **Apache-2.0** (see [`LICENSE`](../LICENSE)). That license
applies only to this project's source — not to the third-party weights, datasets, or
benchmarks above, which remain under their own terms.
