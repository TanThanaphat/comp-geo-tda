"""Core TDA routines: filtrations, Betti numbers, plotting helpers.

The module wraps three back-ends:

* **Ripser** — Vietoris-Rips persistence on Euclidean point clouds.
* **GUDHI** — Alpha complexes (point clouds), Simplex Trees (meshes),
  Cubical complexes (images / sublevel filtrations).
* **Persim** — diagram comparison metrics.

Each ``compute_*`` function returns a list of three ``numpy`` arrays
``[H0, H1, H2]`` whose rows are ``(birth, death)`` pairs.  ``np.inf`` in
the death column is the *essential* class (a feature that never dies).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Persistence computation
# ---------------------------------------------------------------------------

def compute_rips(points: np.ndarray, maxdim: int = 2,
                 thresh: float | None = None) -> List[np.ndarray]:
    """Vietoris-Rips persistence using the ripser library."""
    from ripser import ripser
    kw = dict(maxdim=maxdim)
    if thresh is not None:
        kw["thresh"] = thresh
    return ripser(points, **kw)["dgms"]


def compute_alpha(points: np.ndarray, maxdim: int = 2) -> List[np.ndarray]:
    """Alpha-complex persistence via GUDHI.

    Diagram values are returned in the *Rips-equivalent* radius scale
    (``sqrt`` of the alpha filtration value) so that they are directly
    comparable to Vietoris-Rips diagrams.
    """
    import gudhi as gd
    ac = gd.AlphaComplex(points=points)
    st = ac.create_simplex_tree()
    pers = st.persistence(persistence_dim_max=True)
    dgms: list[list[tuple[float, float]]] = [[] for _ in range(maxdim + 1)]
    for dim, (b, d) in pers:
        if dim > maxdim:
            continue
        b = float(np.sqrt(b))
        d = float(np.sqrt(d)) if d != float("inf") else np.inf
        dgms[dim].append((b, d))
    return [np.array(arr) if arr else np.empty((0, 2)) for arr in dgms]


def compute_mesh_simplex_tree(vertices: np.ndarray, faces: np.ndarray,
                              maxdim: int = 2) -> List[np.ndarray]:
    """Persistence of a mesh through a GUDHI simplex tree.

    Each simplex is inserted with a filtration value equal to its
    longest edge (a Rips-style filtration restricted to the mesh
    skeleton).  This keeps Betti numbers stable while staying
    geometrically meaningful.
    """
    import gudhi as gd
    st = gd.SimplexTree()
    for v_idx, v in enumerate(vertices):
        st.insert([int(v_idx)], filtration=0.0)
    for tri in faces:
        i, j, k = (int(t) for t in tri)
        e_ij = float(np.linalg.norm(vertices[i] - vertices[j]))
        e_jk = float(np.linalg.norm(vertices[j] - vertices[k]))
        e_ik = float(np.linalg.norm(vertices[i] - vertices[k]))
        st.insert([i, j], filtration=e_ij)
        st.insert([j, k], filtration=e_jk)
        st.insert([i, k], filtration=e_ik)
        st.insert([i, j, k], filtration=max(e_ij, e_jk, e_ik))
    pers = st.persistence(persistence_dim_max=True)
    dgms: list[list[tuple[float, float]]] = [[] for _ in range(maxdim + 1)]
    for dim, (b, d) in pers:
        if dim > maxdim:
            continue
        dgms[dim].append((float(b), float(d) if d != float("inf") else np.inf))
    return [np.array(arr) if arr else np.empty((0, 2)) for arr in dgms]


def compute_cubical(image: np.ndarray, maxdim: int = 2) -> List[np.ndarray]:
    """Sublevel-set persistence of a 2D grayscale image (cubical complex)."""
    import gudhi as gd
    cc = gd.CubicalComplex(top_dimensional_cells=image)
    cc.persistence()
    dgms: list[list[tuple[float, float]]] = [[] for _ in range(maxdim + 1)]
    for dim, (b, d) in cc.persistence():
        if dim > maxdim:
            continue
        dgms[dim].append((float(b), float(d) if d != float("inf") else np.inf))
    return [np.array(arr) if arr else np.empty((0, 2)) for arr in dgms]


# ---------------------------------------------------------------------------
# Betti numbers from a persistence diagram
# ---------------------------------------------------------------------------

def betti_numbers(dgms: Sequence[np.ndarray], persistence_eps: float = 0.20,
                  absolute_eps: float | None = None) -> List[int]:
    """Estimate Betti numbers by counting *significant* features.

    A feature ``(b, d)`` is significant when its persistence ``d - b``
    exceeds the cutoff.  The cutoff is either ``absolute_eps`` (when
    given) or ``persistence_eps`` times the largest finite persistence
    seen *across all dimensions* of the diagram — this global scale
    prevents short H0 merge bars from being mistaken for components.
    Essential classes (``d == inf``) always count.
    """
    if absolute_eps is None:
        finite_persistences = []
        for dgm in dgms:
            if len(dgm) == 0:
                continue
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite):
                finite_persistences.append((finite[:, 1] - finite[:, 0]).max())
        global_max = max(finite_persistences) if finite_persistences else 1.0
        thresh = persistence_eps * global_max
    else:
        thresh = absolute_eps

    bettis: list[int] = []
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            bettis.append(0)
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        infinite_count = int(np.sum(~np.isfinite(dgm[:, 1])))
        # For H0 the Betti number in the limit is just the number of
        # connected components, i.e. the number of essential bars.
        if dim == 0:
            bettis.append(max(infinite_count, 1))
            continue
        if len(finite) == 0:
            bettis.append(infinite_count)
            continue
        persistence = finite[:, 1] - finite[:, 0]
        bettis.append(int(np.sum(persistence > thresh)) + infinite_count)
    return bettis


# ---------------------------------------------------------------------------
# Diagram comparison
# ---------------------------------------------------------------------------

def bottleneck(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    from persim import bottleneck as _b
    return float(_b(dgm_a, dgm_b))


def wasserstein(dgm_a: np.ndarray, dgm_b: np.ndarray, p: int = 2) -> float:
    from persim import wasserstein as _w
    return float(_w(dgm_a, dgm_b))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diagram(dgms: Sequence[np.ndarray], title: str = "", ax=None):
    """Plot a persistence diagram (H0, H1, H2)."""
    from persim import plot_diagrams
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    plot_diagrams(list(dgms), show=False, ax=ax)
    ax.set_title(title)
    return ax


def plot_barcode(dgms: Sequence[np.ndarray], title: str = "", ax=None,
                 max_features_per_dim: int = 30):
    """Plot a persistence barcode."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    y = 0
    yticks = []
    yticklabels = []
    finite_max = 0.0
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        order = np.argsort(-(dgm[:, 1] - dgm[:, 0]))
        for idx in order[:max_features_per_dim]:
            b, d = dgm[idx]
            if not np.isfinite(d):
                d = np.nanmax(dgm[np.isfinite(dgm[:, 1])][:, 1]) * 1.1 if np.any(np.isfinite(dgm[:, 1])) else b + 1.0
            ax.hlines(y, b, d, colors=colors[dim % 3], linewidth=2)
            finite_max = max(finite_max, d)
            y += 1
        yticks.append(y - 0.5)
        yticklabels.append(f"H{dim}")
    ax.set_xlim(0, finite_max * 1.05 if finite_max else 1.0)
    ax.set_xlabel("filtration value")
    ax.set_yticks([])
    ax.set_title(title)
    handles = [plt.Line2D([], [], color=colors[i], lw=2, label=f"H{i}")
               for i in range(len(dgms))]
    ax.legend(handles=handles, loc="lower right")
    return ax
