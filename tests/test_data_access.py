"""Reference clouds are resolvable as importlib resources and load correctly.

Point counts mirror the manifest in ``data/README.md``; asserting them guards
against accidental LFS/regeneration drift.
"""

import numpy as np
import open3d as o3d
import pytest

from conftest import data_path

CHENOS = {
    "cheno_1.ply": 55918,
    "cheno_2.ply": 52493,
    "cheno_3.ply": 32295,
    "cheno_4.ply": 72026,
    "cheno_5.ply": 52671,
}

TOMATO = {
    "T01_0313_a.ply": 497195,
    "T01_0315_a.ply": 487964,
    "T01_0317_a.ply": 467721,
    "T01_0319_a.ply": 493840,
    "T01_0321_a.ply": 497999,
    "T01_0324_a.ply": 483818,
    "T01_0325_a.ply": 496151,
    "T03_0313_a.ply": 296197,
    "T03_0315_a.ply": 471805,
    "T03_0317_a.ply": 498207,
    "T03_0319_a.ply": 469141,
    "T03_0321_a.ply": 468684,
    "T03_0324_a.ply": 472892,
    "T03_0325_a.ply": 494395,
}


@pytest.mark.parametrize("name,n", CHENOS.items())
def test_cheno_cloud_resolvable_and_loads(name, n):
    path = data_path("chenos", name)
    assert path.is_file()
    pcd = o3d.t.io.read_point_cloud(str(path))
    assert pcd.point.positions.shape[0] == n


@pytest.mark.parametrize("name,n", TOMATO.items())
def test_tomato_cloud_semantics(name, n):
    path = data_path("tomato", name)
    assert path.is_file()
    pcd = o3d.t.io.read_point_cloud(str(path))
    assert pcd.point.positions.shape[0] == n
    labels = pcd.point["label"].numpy()
    assert labels.min() >= 1  # soil (label 0) removed during downsampling
    assert (labels == 1).sum() > 0  # stem present
    assert labels.max() > 1  # at least one leaf-blade instance
