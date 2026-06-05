# ADR-0004 — Stage A Evaluation Set Composition

**Date:** 2026-06-05  
**Status:** Accepted  
**Phase:** Phase 0 — Reference Baselines

---

## Context

Phase 1 optimization experiments require a frozen, hash-pinned evaluation set against which all experiments are measured. If the eval set changes between experiments, results are not comparable. This ADR documents what was selected, why, and what the set excludes.

---

## Decisions

### 1. Source: COCO val2017 + stage_a_proxy

| Source | Count | Reason |
|---|---:|---|
| COCO val2017 (seed=42) | 95 | Large, diverse, research-licensed, human-written captions already exist |
| stage_a_proxy img1–5.jpg | 5 | Already used for iPhone and CLIP-score baselines; continuity |
| **Total** | **100** | |

**Why COCO val2017 and not Flickr30k or Open Images:**
- COCO has richer per-image annotations (instance segmentation, captions, keypoints) — more signal for future VQA task design
- COCO val2017 is 5,000 images — small enough to download in full, large enough to sample diversely
- Flickr30k is CC-licensed but older and less diverse
- Open Images is massive (9M images) — unnecessary for a 100-image eval set

**Why val2017 and not train2017:** val2017 is 5,000 images, train2017 is 118k. For a 100-image sample, val2017 gives sufficient diversity without requiring a 19 GB download.

### 2. Size: 100 photos (not 200 as originally planned)

The original plan called for 200 photos with 100 VQA pairs. Scope was halved because:
- 100 VQA pairs would take ~4–5 hours of manual writing; 45 pairs is ~2 hours
- 100 photos × 5 VLMs × inference time is already a multi-hour eval run
- Phase 0's goal is a working eval harness, not exhaustive coverage
- Phase 1 can expand the eval set if needed; the manifest hash scheme ensures old results remain comparable via the frozen hash

### 3. Diversity selection (seed=42)

Images are classified into buckets by dominant COCO supercategory. Target allocation:

| Bucket | Target | COCO supercategories |
|---|---:|---|
| indoor_scene | 20 | furniture, appliance, indoor, electronic, kitchen |
| outdoor_scene | 20 | sports, outdoor |
| person_activity | 20 | person |
| animal | 15 | animal |
| vehicle | 10 | vehicle |
| food | 10 | food |
| **Total** | **95** | |

Random seed is fixed at 42. Changing the seed changes the selection and invalidates the manifest hash.

### 4. Caption set (50 photos)

The first 50 selected COCO images are the caption evaluation set. For each:
- The longest COCO human caption ≥ 40 characters is used as the reference
- All 5 COCO captions are stored in `captions.json` for future use

**CLIP-score upper bound:** Running `compute_clip_score.py` against `captions.json` gives the human-caption CLIP-score baseline — a practical ceiling for what the models should aim for on this image set.

The 5 stage_a_proxy images (img1–5.jpg) are NOT in the caption set — they are used only for continuity with iPhone baselines already measured.

### 5. VQA set (45 photos)

The remaining 45 COCO images are the VQA evaluation set. Each has one question + one expected answer pulled from the **COCO VQA v2** dataset (214,354 human-written Q/A pairs for val2014 images, which share IDs with val2017). No manual writing required. Question type distribution:

| Type | Count | Notes |
|---|---:|---|
| activity | 15 | "What is the person doing?" |
| counting | 10 | "How many X are in the image?" |
| object_presence | 10 | "Is there a Y in the image?" (yes/no) |
| color_attribute | 10 | "What color is the Z?" |
| scene_location | 5 | "Is this indoors or outdoors?" |
| **Total** | **45** | |

Questions use exact-match scoring (same as POPE in Task 2.2). Ambiguous answers avoided: "2" not "two"; "yes"/"no" for binary; simple color names.

VQA pairs are in `vqa_template.json` (TODO fields) → manually filled → renamed to `vqa.json`. Once `vqa.json` exists, re-run `hash_eval_set.py` to update the manifest hash.

### 6. Licensing

COCO is licensed under Creative Commons Attribution 4.0 (images: Flickr ToS). The images themselves cannot be redistributed directly. For the public release (D0.5):
- Redistribute: `manifest.json` (photo IDs + hashes), `captions.json`, `vqa.json`
- Do NOT redistribute: the actual `.jpg` files
- Users fetch images from COCO official download using the IDs in the manifest

### 7. Hash pinning

Every file in `datasets/stage_a/` is SHA-256 hashed. The sorted manifest is itself hashed to produce `manifest_hash`. This hash is stored in `manifest.json` and must be recorded in every `ExperimentConfig.eval_set_hash`.

**Final manifest hash:**
```
e2128ae022b3720375d7c866a037b6d8ec4b399ff92cb59e6065ec9fb7f3e29f
```

---

## Artifacts

| File | Purpose |
|---|---|
| `datasets/stage_a/photos/` | 100 photos (95 COCO + 5 proxy) |
| `datasets/stage_a/captions.json` | 50 COCO reference captions |
| `datasets/stage_a/vqa.json` | 45 VQA pairs (from COCO VQA v2, agreement ≥ 4/10) |
| `datasets/stage_a/manifest.json` | Per-file SHA-256 + manifest hash |
| `datasets/stage_a/selection_log.json` | Audit log of curation decisions |
| `tools/curate_eval_set.py` | Curation script (reproducible with seed=42) |
| `tools/build_vqa_from_coco.py` | VQA pair extraction from COCO VQA v2 |
| `tools/hash_eval_set.py` | Manifest generation script |

---

## Consequences

- All Phase 1 experiments run against this 100-image set, identified by `manifest_hash`
- Adding photos or changing captions requires re-running `hash_eval_set.py` and bumping the version; old experiment results tied to the old hash remain valid
- The VQA set is complete and ready for Phase 1 scoring (sourced from COCO VQA v2; no manual authoring needed)
- CLIP-score on the 50 caption photos can be run immediately (no manual work required)
