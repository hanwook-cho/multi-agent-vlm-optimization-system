"""StudentSpec unit tests (ADR-0012) — content-addressing + validation contract.

These run in CI (no heavy model deps); the actual build_student smoke is a local,
compute-gated check, not a CI test.
"""

from __future__ import annotations

import pytest

from schemas.students import StudentSpec

BASE = {
    "lm": "Qwen/Qwen2.5-0.5B-Instruct",
    "vision": "google/siglip-base-patch16-224",
    "align": {"data": "coco_caption_5k", "steps": 2000},
    "distill": {"data": "qa_balanced_5k"},
}


def test_minimal_spec_applies_defaults():
    s = StudentSpec.model_validate(BASE)
    assert s.init == "scratch"
    assert s.projector.type == "mlp" and s.projector.depth == 2
    assert s.distill.teacher == "Qwen2.5-VL-3B" and s.distill.lora_r == 16
    assert s.eval.benchmarks == ["POPE", "RealWorldQA", "MMBench_DEV_EN"]
    assert len(s.content_hash()) == 64


def test_content_hash_excludes_notes():
    a = StudentSpec.model_validate(BASE)
    b = StudentSpec.model_validate({**BASE, "notes": "agent proposed this on 2026-06-15"})
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_with_a_search_dimension():
    a = StudentSpec.model_validate(BASE)
    b = StudentSpec.model_validate({**BASE, "distill": {"data": "qa_balanced_5k", "lora_r": 32}})
    assert a.content_hash() != b.content_hash()


def test_init_must_be_scratch_or_adapt():
    with pytest.raises(ValueError):
        StudentSpec.model_validate({**BASE, "init": "warmstart"})
    with pytest.raises(ValueError):
        StudentSpec.model_validate({**BASE, "init": "adapt:"})
    # valid adapt form is accepted
    s = StudentSpec.model_validate({**BASE, "init": "adapt:HuggingFaceTB/SmolVLM-500M"})
    assert s.init.startswith("adapt:")
