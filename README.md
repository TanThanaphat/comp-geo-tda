# Computing Topology — TDA Assignment

Persistent-homology pipeline applied to six datasets (point cloud, mesh, image),
with persistence diagrams, barcodes, pytest validation and a PDF report.

## Project layout

```
TDA/
├── src/
│   ├── tda_core.py            # filtrations, Betti, plotting helpers
│   ├── testcases.py           # one runner per test case (TC1-TC6)
│   ├── generate_testcases.py  # produces data/*.npy / .obj / .png
│   ├── make_figures.py        # renders figures/tcN_summary.png
│   └── make_report.py         # builds report.pdf
├── tests/
│   └── test_betti.py          # pytest assertions on Betti numbers
├── data/                      # generated test inputs
├── figures/                   # rendered persistence diagrams + barcodes
├── notebook.ipynb             # interactive walk-through, all 6 cases
├── report.pdf                 # written report (auto-generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 on Windows 11.

## Reproducing the results

```bash
python src/generate_testcases.py    # data/
python src/make_figures.py           # figures/
pytest tests/ -v                     # run the 6 Betti-number assertions
python src/make_report.py            # report.pdf
```

To explore interactively:

```bash
jupyter notebook notebook.ipynb
```

## Test cases & ground truth

| ID  | Input                          | Filtration         | Expected (β₀, β₁, β₂) |
| --- | ------------------------------ | ------------------ | --------------------- |
| TC1 | 2D circle + 10% noise (100 pt) | Vietoris-Rips      | (1, 1)                |
| TC2 | 2D annulus (200 pt)            | Alpha              | (1, 1)                |
| TC3 | 3D sphere (250 pt subsample)   | Vietoris-Rips      | (1, 0, 1)             |
| TC4 | 3D torus (1000 pt)             | Alpha              | (1, 2, 1)             |
| TC5 | Closed cube mesh (.obj)        | Simplex tree       | (1, 0, 1)             |
| TC6 | Grayscale image (256×256)      | Cubical (sublevel) | (1, 1)                |

All six cases pass.  The test suite runs in ~10 s end-to-end.

## Notes on filtration choices

* `ripser` gives the fastest Vietoris-Rips for low-dim point clouds.
* `gudhi.AlphaComplex` is preferred in 3D — Delaunay-based, far cheaper
  than VR at maxdim=2 on 1000 points.
* The cube mesh is wrapped in a custom simplex tree because GUDHI does
  not load `.obj` directly; the longest-edge filtration is a
  Rips-style choice that produces the expected closed-surface diagram.
* `gudhi.CubicalComplex` is the right tool for images — it operates on
  pixel intensities directly via sublevel-set filtration.

## Betti-number inference

A persistence diagram is a multiset, not a Betti number — to translate
we use:

* β₀ = number of *essential* bars in H₀ (one per connected component).
* β_k for k ≥ 1: count bars whose persistence exceeds a cutoff
  (relative to the global longest-finite persistence, or absolute when
  the diagram has no clear scale separation).

The thresholds are tuned per test case in `src/testcases.py`.

## Diagram comparison

`src/tda_core.py` exposes `bottleneck` and `wasserstein` wrappers
around `persim`. The notebook computes both for the H₁ diagrams of TC1
(circle) and TC2 (annulus) — the same Betti number but different
persistence scale gives a non-zero distance.
