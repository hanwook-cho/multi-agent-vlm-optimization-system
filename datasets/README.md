# Datasets

Evaluation sets used for scoring models. Only manifests, captions, and VQA pairs
are committed here. Actual photos are fetched from source datasets by the user
(see redistribution policy below).

## Stage A — proxy eval set (Phase 0, Task 4.5)

`stage_a_proxy/manifest.json` — SHA-256-pinned list of 200 photo IDs
`stage_a_proxy/captions.json` — 100 reference captions (our own, committable)
`stage_a_proxy/vqa.json` — 100 VQA pairs with expected answers (our own, committable)
`stage_a_proxy/photos/` — **gitignored** — fetch from source datasets

Photos are drawn from Flickr30k, COCO Captions, and Open Images (research-use licenses).
To populate `stage_a_proxy/photos/`, run: `python tools/fetch_eval_photos.py` (Phase 0 Week 4).

The manifest SHA-256 is the `dataset_hash` field in every `ExperimentConfig` that uses
this eval set. Changing the eval set requires updating all affected ExperimentConfigs.
