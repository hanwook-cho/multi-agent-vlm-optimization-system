"""B1 rehearsal wiring — `_load_rows` continual-learning replay (CI-safe).

Locks the contract that `DistillSpec.rehearse_data`/`rehearse_frac` actually mix a
second cache into the construction distill set. This was a NO-OP before: the spec
field existed but build_student never passed it to the loader, so the P2-B1 MCQ
negative result was trained without the rehearsal protection its spec claimed.

Pure I/O over tiny temp JSONL caches + touched (empty) image files — `_cache_rows`
only checks image existence, not validity, so no real images or models are needed.
"""

from __future__ import annotations

import json

import runners.build_student as bs


def _make_cache(tmp_path, name: str, n: int, target: str) -> tuple[str, str]:
    """Write an n-row JSONL cache + touched images; return (cache_path, image_dir)."""
    img_dir = tmp_path / f"{name}_imgs"
    img_dir.mkdir()
    lines = []
    for i in range(n):
        fn = f"{name}_{i}.jpg"
        (img_dir / fn).touch()
        lines.append(json.dumps({"image": fn, "prompt": f"q{i}?", "target": target}))
    cache = tmp_path / f"{name}.jsonl"
    cache.write_text("\n".join(lines) + "\n")
    return str(cache), str(img_dir)


def _registry(tmp_path):
    prim = _make_cache(tmp_path, "prim", 20, "GROUND")
    reh = _make_cache(tmp_path, "reh", 8, "MCQ")
    return {"prim": prim, "reh": reh}


def test_no_rehearsal_loads_primary_only(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "DATA_REGISTRY", _registry(tmp_path))
    rows = bs._load_rows("prim", None)
    assert len(rows) == 20
    assert {r["target"] for r in rows} == {"GROUND"}


def test_rehearsal_mixes_in_fraction(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "DATA_REGISTRY", _registry(tmp_path))
    # frac 0.5 of 20 primary = 10 requested, but only 8 rehearse rows exist → +8.
    rows = bs._load_rows("prim", None, rehearse_key="reh", rehearse_frac=0.5)
    assert len(rows) == 28
    targets = [r["target"] for r in rows]
    assert targets.count("GROUND") == 20 and targets.count("MCQ") == 8


def test_rehearsal_fraction_capped_below_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "DATA_REGISTRY", _registry(tmp_path))
    # frac 0.25 of 20 = 5 rehearse rows (pool of 8 has room).
    rows = bs._load_rows("prim", None, rehearse_key="reh", rehearse_frac=0.25)
    assert len(rows) == 25
    assert [r["target"] for r in rows].count("MCQ") == 5


def test_rehearsal_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "DATA_REGISTRY", _registry(tmp_path))
    a = bs._load_rows("prim", None, rehearse_key="reh", rehearse_frac=0.5)
    b = bs._load_rows("prim", None, rehearse_key="reh", rehearse_frac=0.5)
    assert [r["image_path"] for r in a] == [r["image_path"] for r in b]


def test_limit_applied_after_mixing(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "DATA_REGISTRY", _registry(tmp_path))
    rows = bs._load_rows("prim", 12, rehearse_key="reh", rehearse_frac=0.5)
    assert len(rows) == 12  # capped after the mix+shuffle


def test_zero_frac_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "DATA_REGISTRY", _registry(tmp_path))
    rows = bs._load_rows("prim", None, rehearse_key="reh", rehearse_frac=0.0)
    assert len(rows) == 20 and {r["target"] for r in rows} == {"GROUND"}
