# Tools

Developer-facing utility scripts. Not imported by the main system.
Run directly from the repo root: `python tools/<script>.py`.

Planned scripts:
- `fetch_eval_photos.py` — downloads Stage A photos from Flickr30k/COCO/Open Images
  using the manifest in `datasets/stage_a_proxy/manifest.json` (Phase 0 Week 4)
- `export_schemas.py` — regenerates `schemas/*.schema.json` from Pydantic models
  (useful after editing a model to verify JSON Schema output is correct)
- `inspect_results.py` — pretty-prints MetricsReport records from the metrics DB
- `validate_configs.py` — validates all files in `configs/` against their schemas
