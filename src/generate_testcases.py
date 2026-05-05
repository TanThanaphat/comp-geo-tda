"""Generate the six standard test-case datasets used in this TDA assignment.

Outputs are saved into the project's ``data/`` directory.
"""
from __future__ import annotations

import os
import numpy as np
from PIL import Image, ImageDraw

RNG = np.random.default_rng(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def tc1_circle(n: int = 100, noise: float = 0.10) -> np.ndarray:
    theta = RNG.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    pts = pts + RNG.normal(0.0, noise, pts.shape)
    return pts.astype(np.float64)


def tc2_annulus(n: int = 200, r_in: float = 0.4, r_out: float = 1.0) -> np.ndarray:
    """Square with a circular hole in the centre."""
    pts = []
    while len(pts) < n:
        p = RNG.uniform(-r_out, r_out, 2)
        d = np.linalg.norm(p)
        if r_in <= d <= r_out:
            pts.append(p)
    return np.asarray(pts, dtype=np.float64)


def tc3_sphere(n: int = 500) -> np.ndarray:
    v = RNG.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float64)


def tc4_torus(n: int = 1000, R: float = 2.0, r: float = 0.7) -> np.ndarray:
    """Uniform sample on a torus surface (rejection sampling)."""
    out = []
    while len(out) < n:
        u = RNG.uniform(0, 2 * np.pi)
        v = RNG.uniform(0, 2 * np.pi)
        w = RNG.uniform(0, 1)
        if w <= (R + r * np.cos(u)) / (R + r):
            x = (R + r * np.cos(u)) * np.cos(v)
            y = (R + r * np.cos(u)) * np.sin(v)
            z = r * np.sin(u)
            out.append([x, y, z])
    return np.asarray(out, dtype=np.float64)


def tc5_cube_obj(path: str) -> None:
    """A closed, watertight cube mesh (.obj) — 8 verts, 12 triangles."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = np.array([
        [1, 2, 3], [1, 3, 4],
        [5, 7, 6], [5, 8, 7],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 8], [3, 8, 4],
        [4, 8, 5], [4, 5, 1],
    ])
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")


def tc6_circle_image(path: str, size: int = 256) -> None:
    """Grayscale image of a thin ring on white background."""
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    cx = cy = size // 2
    R_out = size // 3
    R_in = R_out - 6
    draw.ellipse([cx - R_out, cy - R_out, cx + R_out, cy + R_out], fill=0)
    draw.ellipse([cx - R_in, cy - R_in, cx + R_in, cy + R_in], fill=255)
    img.save(path)


def main() -> None:
    np.save(os.path.join(DATA_DIR, "tc1_circle.npy"), tc1_circle())
    np.save(os.path.join(DATA_DIR, "tc2_annulus.npy"), tc2_annulus())
    np.save(os.path.join(DATA_DIR, "tc3_sphere.npy"), tc3_sphere())
    np.save(os.path.join(DATA_DIR, "tc4_torus.npy"), tc4_torus())
    tc5_cube_obj(os.path.join(DATA_DIR, "tc5_cube.obj"))
    tc6_circle_image(os.path.join(DATA_DIR, "tc6_circle.png"))
    print("Generated TC1-TC6 datasets in", os.path.abspath(DATA_DIR))


if __name__ == "__main__":
    main()
