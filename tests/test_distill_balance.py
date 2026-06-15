"""B1.1 — balanced hard-negative QA parsing helpers (pure functions, CI-safe).

The full qa_balanced cache generation needs the teacher model (compute-gated,
validated by a local pilot); these tests lock the parsing/labelling contract that
turns teacher text into grounded yes/no presence records.
"""

from __future__ import annotations

from services.distillation_pipeline import (
    COCO80,
    _is_yesno,
    _parse_object_list,
    _parse_presence_labels,
)


def test_is_yesno():
    assert _is_yesno("Yes") and _is_yesno("no.") and _is_yesno(" NO ")
    assert not _is_yesno("a red car") and not _is_yesno("two dogs")


def test_parse_object_list_normalizes():
    out = _parse_object_list("a Dog, the Cat\n2 cars, , surfboard.")
    assert "dog" in out and "cat" in out and "surfboard" in out
    # articles stripped, blanks dropped
    assert "a dog" not in out and "" not in out


def test_parse_presence_labels():
    text = "dog: yes\ncar: no\noven : YES\nnonsense line\ntoaster: no."
    labels = _parse_presence_labels(text)
    assert labels == {"dog": True, "car": False, "oven": True, "toaster": False}


def test_coco80_is_the_negative_pool():
    assert len(COCO80) == 80
    assert "oven" in COCO80 and "teddy bear" in COCO80
