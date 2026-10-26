"""Downsample Pheno4D tomato point clouds to <=MAXPTS points.

Reproduces the downsampling used for the tomato evaluation clouds in
data/tomato/. Follows the article's method: remove soil, then increase the
minimum spacing between points (voxel grid, one representative point per
occupied cell) so each cloud has at most MAXPTS points. Labels are preserved
(1 = stem, >=2 = individual leaf-blade instances).

Usage:
    python downsample_tomato.py [--source /path/to/Pheno4D.zip] [--out data/tomato]
        [--maxpts 500000] [--plants Tomato01 Tomato02]
        [--dates 0313 0315 0317 0319 0321 0324 0325]
"""
import argparse
import io
import os
import zipfile

import numpy as np
import pandas as pd

DEFAULT_PLANTS = ["Tomato01", "Tomato02"]
DEFAULT_DATES = ["0313", "0315", "0317", "0319", "0321", "0324", "0325"]


def write_binary_ply(path, xyz, label):
    n = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property int label\nend_header\n".format(n=n)
    )
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<i4")])
    rec["x"], rec["y"], rec["z"], rec["label"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], label
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def voxel_downsample(xyz, label, target):
    """One representative (first) point per occupied voxel, up to target points."""
    mins = xyz.min(0)
    size = 0.02
    while True:
        vid = np.floor((xyz - mins) / size)
        keys = vid[:, 0] * 1_000_000 + vid[:, 1] * 1000 + vid[:, 2]
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        start = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
        if len(start) <= target:
            break
        size *= 1.05
    reps = order[start]
    return xyz[reps], label[reps]


def load_points(source, plant, date):
    member = "Pheno4D/{0}/T{1}_{2}_a.txt".format(plant, plant[-2:], date)
    if os.path.isdir(source):
        raw = open(os.path.join(source, member), "rb").read()
    else:
        with zipfile.ZipFile(source) as z:
            raw = z.read(member)
    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None,
                     names=["x", "y", "z", "label"])
    m = df["label"].values != 0
    xyz = df[["x", "y", "z"]].values[m].astype(np.float64)
    label = df["label"].values[m].astype(np.int32)
    return xyz, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/tmp/Pheno4D.zip",
                    help="Path to Pheno4D.zip or an extracted Pheno4D directory.")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--maxpts", type=int, default=500_000)
    ap.add_argument("--plants", nargs="+", default=DEFAULT_PLANTS)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for plant in args.plants:
        for d in args.dates:
            xyz, label = load_points(args.source, plant, d)
            if len(xyz) > args.maxpts:
                xyz, label = voxel_downsample(xyz, label, args.maxpts)
            out = os.path.join(args.out, "T{0}_{1}_a.ply".format(plant[-2:], d))
            write_binary_ply(out, xyz, label)
            print("{0}: {1} pts -> {2} ({3} pts)".format(
                os.path.basename(out), len(xyz), out, len(xyz)))


if __name__ == "__main__":
    main()
