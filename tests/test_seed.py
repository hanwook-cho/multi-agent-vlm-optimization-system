"""build_student.seed_everything — reproducibility guard (CI-safe).

Construction runs train a fresh projector from scratch; without a fixed seed,
init variance confounds cross-spec comparisons (the P2-B1 lever runs on 2026-06-17
were degenerate from unlucky inits, not the levers). This locks that seeding makes
the RNGs deterministic.
"""

from __future__ import annotations

import torch

from runners.build_student import seed_everything


def test_seed_makes_torch_deterministic():
    seed_everything(0)
    a = torch.randn(64)
    seed_everything(0)
    b = torch.randn(64)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    seed_everything(0)
    a = torch.randn(64)
    seed_everything(1)
    b = torch.randn(64)
    assert not torch.equal(a, b)


def test_seeds_python_and_numpy():
    import random
    seed_everything(7)
    r = random.random()
    try:
        import numpy as np
        n = float(np.random.rand())
    except Exception:
        n = None
    seed_everything(7)
    assert random.random() == r
    if n is not None:
        import numpy as np
        assert float(np.random.rand()) == n
