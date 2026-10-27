#!/usr/bin/env python
"""Export reference point-cloud figures for the documentation.

Renders three figures, one subplot per cloud, from the reference data in
``data/`` (loaded via ``importlib.resources`` as in the Phase-2 tests):

- ``chenopodium.png``   — all 5 Chenopodium plants.
- ``tomato01.png``      — the 7 Tomato01 time-points.
- ``tomato02.png``      — the 7 Tomato02 time-points.

Tomato clouds carry an integer ``label`` (stem ``1``, leaf ``>=2``) and are
coloured by it; Chenopodium clouds have no label and are coloured by height.

Usage::

    python script/export_figures.py

Writes the PNGs to ``data/figures/`` (tracked by git-lfs) so the README can
reference the available reference data without re-running heavy code.
"""

from pathlib import Path
from importlib import resources
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data" / "figures"
MAX_POINTS = 20000  # subsample per cloud for display speed
SEED = 0

CHENO = ["cheno_1.ply", "cheno_2.ply", "cheno_3.ply", "cheno_4.ply", "cheno_5.ply"]
TOMATO01 = [f"T01_{stage}_a.ply" for stage in ("0313", "0315", "0317", "0319", "0321", "0324", "0325")]
TOMATO02 = [f"T02_{stage}_a.ply" for stage in ("0313", "0315", "0317", "0319", "0321", "0324", "0325")]


def load_cloud(*parts):
    """Return (points Nx3, labels or None) for a reference cloud."""
    p = str(resources.files("data").joinpath(*parts))
    t = o3d.t.io.read_point_cloud(p)
    pts = t.point.positions.numpy()
    labels = None
    if "label" in t.point:
        labels = np.asarray(t.point["label"].numpy()).ravel()
    return pts.astype(np.float64), labels


def subsample(pts, labels):
    n = len(pts)
    if n <= MAX_POINTS:
        return pts, labels
    idx = np.random.default_rng(SEED).choice(n, MAX_POINTS, replace=False)
    if labels is not None:
        return pts[idx], labels[idx]
    return pts[idx], None


def _axis(ax, pts, labels):
    """Plot one cloud, hiding the 3D clutter and equalising the axes."""
    x, y, z = pts.T
    if labels is not None:
        leaf = labels >= 2
        stem = (labels == 1) & ~leaf
        ax.scatter(x[stem], y[stem], z[stem], c="#8c564b", s=1, label="stem")
        ax.scatter(x[leaf], y[leaf], z[leaf], c="#2ca02c", s=1, label="leaf")
    else:
        ax.scatter(x, y, z, c=z, s=1, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def _grid(n):
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    return rows, cols


def export_one(name, clouds, titles, with_label):
    rows, cols = _grid(len(clouds))
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for i, (cloud, title) in enumerate(zip(clouds, titles)):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        pts, labels = load_cloud(*cloud)
        pts, labels = subsample(pts, labels)
        _axis(ax, pts, labels)
        ax.set_title(title, fontsize=9)
    if with_label:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(name, fontsize=12)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_DIR / (name + '.png')}")


def main():
    export_one("chenopodium", [("chenos", c) for c in CHENO], CHENO, with_label=False)
    export_one("tomato01", [("tomato", c) for c in TOMATO01], TOMATO01, with_label=True)
    export_one("tomato02", [("tomato", c) for c in TOMATO02], TOMATO02, with_label=True)


if __name__ == "__main__":
    main()
