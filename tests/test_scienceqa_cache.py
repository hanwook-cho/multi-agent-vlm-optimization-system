"""P2-B1 ScienceQA cache — pure formatting contract (CI-safe, no download).

ScienceQA is the MMBench-distribution training source (science/knowledge/reasoning,
natively MCQ). These tests lock the row-formatting contract that turns a ScienceQA
example into a {prompt, target} cache row consumed by build_student.
"""

from __future__ import annotations

from runners.build_scienceqa_cache import format_record, MCQ_TRAIN_SUFFIX


def test_format_wellformed_4choice():
    out = format_record("What is shown?", "", ["cat", "dog", "fish", "bird"], 1)
    assert out is not None
    prompt, target = out
    assert target == "B"
    assert prompt.startswith("What is shown?\nA. cat\nB. dog\nC. fish\nD. bird")
    assert prompt.endswith(MCQ_TRAIN_SUFFIX)


def test_format_prepends_hint():
    prompt, _ = format_record("Which variable?", "An experiment is described.",
                              ["temperature", "mass"], 0)
    assert prompt.startswith("An experiment is described.\nWhich variable?")


def test_format_two_choices_ok():
    out = format_record("True or false?", "", ["yes", "no"], 1)
    assert out is not None and out[1] == "B"


def test_format_rejects_too_few_choices():
    assert format_record("q", "", ["only one"], 0) is None


def test_format_rejects_too_many_choices():
    assert format_record("q", "", ["a", "b", "c", "d", "e"], 2) is None


def test_format_rejects_out_of_range_answer():
    assert format_record("q", "", ["a", "b", "c"], 5) is None
    assert format_record("q", "", ["a", "b"], None) is None
