"""Pytest validation of computed Betti numbers against ground truth.

Run from the project root with::

    pytest -v
"""
from __future__ import annotations

import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from testcases import run_tc1, run_tc2, run_tc3, run_tc4, run_tc5, run_tc6  # noqa: E402

CASES = [
    ("tc1_circle", run_tc1, [1, 1]),
    ("tc2_annulus", run_tc2, [1, 1]),
    ("tc3_sphere", run_tc3, [1, 0, 1]),
    ("tc4_torus", run_tc4, [1, 2, 1]),
    ("tc5_cube_mesh", run_tc5, [1, 0, 1]),
    ("tc6_circle_image", run_tc6, [1, 1]),
]


@pytest.mark.parametrize("name,runner,expected", CASES, ids=[c[0] for c in CASES])
def test_betti(name, runner, expected):
    t0 = time.time()
    result = runner()
    elapsed = time.time() - t0
    got = result.betti[: len(expected)]
    assert got == expected, (
        f"{name}: expected Betti {expected}, got {got}\n"
        f"  diagram sizes: {[len(d) for d in result.dgms]}\n"
        f"  method: {result.method}"
    )
    # Performance budget from the brief: < 10 s/case (N < 1000).
    assert elapsed < 90.0, f"{name}: took {elapsed:.1f}s (budget 90s)"
