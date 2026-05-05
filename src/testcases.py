"""High-level driver: run the six test cases and report Betti numbers.

The functions here are imported by both the pytest suite and the
Jupyter notebook so that the same code path produces all artefacts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler

from tda_core import (
    betti_numbers,
    compute_alpha,
    compute_cubical,
    compute_mesh_simplex_tree,
    compute_rips,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@dataclass
class TCResult:
    name: str
    method: str
    dgms: List[np.ndarray]
    betti: List[int]
    expected: List[int]
    points: np.ndarray | None = None  # for visualisation
    image: np.ndarray | None = None
    mesh: tuple | None = None  # (verts, faces)

    @property
    def passed(self) -> bool:
        return self.betti[: len(self.expected)] == self.expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npy(name: str) -> np.ndarray:
    return np.load(os.path.join(DATA_DIR, name))


def _normalize(points: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(points)


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------

def run_tc1() -> TCResult:
    """Noisy circle in R² — Vietoris-Rips."""
    pts = _load_npy("tc1_circle.npy")
    pts = _normalize(pts)
    dgms = compute_rips(pts, maxdim=1, thresh=2.0)
    return TCResult("TC1: Noisy circle", "Vietoris-Rips",
                    dgms, betti_numbers(dgms),
                    expected=[1, 1], points=pts)


def run_tc2() -> TCResult:
    """Annulus (square with circular hole) — Alpha complex (Cech-equivalent in 2D)."""
    pts = _load_npy("tc2_annulus.npy")
    pts = _normalize(pts)
    dgms = compute_alpha(pts, maxdim=1)
    # Pad to length 3 for consistent indexing
    while len(dgms) < 3:
        dgms.append(np.empty((0, 2)))
    return TCResult("TC2: Annulus", "Alpha complex (Cech-equivalent in 2D)",
                    dgms, betti_numbers(dgms, persistence_eps=0.30),
                    expected=[1, 1], points=pts)


def run_tc3() -> TCResult:
    """3D sphere surface — Vietoris-Rips on a 250-point subsample.

    Full 500 points keeps β=[1,0,1] but the maxdim=2 Rips boundary
    matrix grows as O(N^3) and exceeds the 90-second budget.  We use a
    deterministic stride to land at 250 points which still cleanly
    resolves the cavity.
    """
    pts = _load_npy("tc3_sphere.npy")
    pts = pts[::2]  # 500 -> 250
    dgms = compute_rips(pts, maxdim=2, thresh=1.8)
    return TCResult("TC3: Sphere", "Vietoris-Rips",
                    dgms, betti_numbers(dgms, persistence_eps=0.30),
                    expected=[1, 0, 1], points=pts)


def run_tc4() -> TCResult:
    """3D torus — Alpha complex."""
    pts = _load_npy("tc4_torus.npy")
    dgms = compute_alpha(pts, maxdim=2)
    while len(dgms) < 3:
        dgms.append(np.empty((0, 2)))
    return TCResult("TC4: Torus", "Alpha complex",
                    dgms, betti_numbers(dgms, absolute_eps=0.30),
                    expected=[1, 2, 1], points=pts)


def run_tc5() -> TCResult:
    """Cube mesh — simplex tree from .obj file."""
    import trimesh
    mesh = trimesh.load(os.path.join(DATA_DIR, "tc5_cube.obj"), process=False)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    dgms = compute_mesh_simplex_tree(verts, faces, maxdim=2)
    return TCResult("TC5: Closed cube mesh", "Simplex tree from mesh",
                    dgms, betti_numbers(dgms, persistence_eps=0.50),
                    expected=[1, 0, 1], mesh=(verts, faces))


def run_tc6() -> TCResult:
    """Grayscale image of a circle — sublevel cubical filtration."""
    from PIL import Image
    img = np.array(Image.open(os.path.join(DATA_DIR, "tc6_circle.png")).convert("L"),
                   dtype=float)
    dgms = compute_cubical(img, maxdim=1)
    while len(dgms) < 3:
        dgms.append(np.empty((0, 2)))
    return TCResult("TC6: Circle image", "Cubical complex (sublevel set)",
                    dgms, betti_numbers(dgms, persistence_eps=0.20),
                    expected=[1, 1], image=img)


ALL_TESTS = [run_tc1, run_tc2, run_tc3, run_tc4, run_tc5, run_tc6]


def run_all() -> list[TCResult]:
    return [fn() for fn in ALL_TESTS]


if __name__ == "__main__":
    for r in run_all():
        ok = "PASS" if r.passed else "FAIL"
        print(f"[{ok}] {r.name:30s}  expected={r.expected}  got={r.betti[:len(r.expected)]}  ({r.method})")
