"""Render persistence diagrams + barcodes + input visualisations.

Outputs go into ``figures/`` so the report and the notebook can embed
them.  Each test case produces one combined figure ``tcN_summary.png``.
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tda_core import bottleneck, plot_barcode, plot_diagram  # noqa: E402
from testcases import ALL_TESTS  # noqa: E402

FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _draw_input(ax, result):
    if result.points is not None:
        pts = result.points
        if pts.shape[1] == 2:
            ax.scatter(pts[:, 0], pts[:, 1], s=8)
            ax.set_aspect("equal")
        else:
            ax.remove()
            ax = result._ax3d  # populated below
    elif result.image is not None:
        ax.imshow(result.image, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    elif result.mesh is not None:
        verts, faces = result.mesh
        ax.remove()
        ax = result._ax3d
    ax.set_title("Input data")
    return ax


def render_case(result):
    fig = plt.figure(figsize=(13, 4))
    if (result.points is not None and result.points.shape[1] == 3) or result.mesh is not None:
        ax_in = fig.add_subplot(1, 3, 1, projection="3d")
        if result.points is not None:
            pts = result.points
            sample = pts[np.random.default_rng(0).choice(len(pts),
                                                          size=min(800, len(pts)),
                                                          replace=False)]
            ax_in.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=4)
        else:
            verts, faces = result.mesh
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            tri = Poly3DCollection(verts[faces], alpha=0.4, edgecolor="k")
            ax_in.add_collection3d(tri)
            ax_in.set_xlim(verts[:, 0].min(), verts[:, 0].max())
            ax_in.set_ylim(verts[:, 1].min(), verts[:, 1].max())
            ax_in.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        ax_in.set_title("Input")
    else:
        ax_in = fig.add_subplot(1, 3, 1)
        if result.points is not None:
            ax_in.scatter(result.points[:, 0], result.points[:, 1], s=8)
            ax_in.set_aspect("equal")
        elif result.image is not None:
            ax_in.imshow(result.image, cmap="gray")
            ax_in.set_xticks([]); ax_in.set_yticks([])
        ax_in.set_title("Input")

    ax_dgm = fig.add_subplot(1, 3, 2)
    plot_diagram(result.dgms, title="Persistence diagram", ax=ax_dgm)

    ax_bc = fig.add_subplot(1, 3, 3)
    plot_barcode(result.dgms, title="Barcode", ax=ax_bc)

    fig.suptitle(f"{result.name}  —  Betti {result.betti[: len(result.expected)]}  "
                 f"(expected {result.expected})  via {result.method}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def main() -> None:
    timings = []
    for i, runner in enumerate(ALL_TESTS, start=1):
        t0 = time.time()
        result = runner()
        elapsed = time.time() - t0
        timings.append((result.name, elapsed, result.passed))
        fig = render_case(result)
        out = os.path.join(FIG_DIR, f"tc{i}_summary.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[{('PASS' if result.passed else 'FAIL'):4s}] {result.name:30s} "
              f"{elapsed:6.2f}s  -> {out}")

    # Cross-comparison: bottleneck distance between TC1 (circle) H1 and TC2 (annulus) H1
    from testcases import run_tc1, run_tc2
    r1, r2 = run_tc1(), run_tc2()
    bd = bottleneck(r1.dgms[1], r2.dgms[1])
    print(f"\nBottleneck distance H1(TC1 circle) vs H1(TC2 annulus): {bd:.4f}")

    with open(os.path.join(FIG_DIR, "timings.txt"), "w") as f:
        f.write("name,seconds,pass\n")
        for n, t, ok in timings:
            f.write(f"{n},{t:.3f},{ok}\n")
        f.write(f"\nbottleneck H1(circle vs annulus): {bd:.4f}\n")


if __name__ == "__main__":
    main()
