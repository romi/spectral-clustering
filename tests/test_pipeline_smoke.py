"""End-to-end smoke tests of the spectral segmentation on real reference data.

The full CLI pipelines are monolithic and slow (they persist many plots and
run thousands of energy-optimisation iterations), so these smoke tests run the
core stages on a small subsample of a reference cloud: build the Riemannian
graph, check connectivity, and verify the spectral clustering yields a
stem/leaf split. Assertions target structure, not exact values.
"""

import numpy as np
import networkx as nx
import open3d as o3d
import pytest

import spectral_clustering.similarity_graph as sgk
from spectral_clustering.point_cloud_graph import PointCloudGraph

from conftest import data_path

# One cloud per pipeline, as per the plan ("keep it light").
PIPELINE_CLOUDS = [
    pytest.param(("tomato", "T01_0313_a.ply"), id="tomato"),
    pytest.param(("chenos", "cheno_1.ply"), id="chenopodium"),
]


def _subsample(cloud, n=2000, seed=0):
    """Randomly subsample a reference cloud to keep the smoke test fast."""
    pcd = o3d.t.io.read_point_cloud(str(data_path(*cloud)))
    pts = np.asarray(pcd.point.positions.numpy(), dtype=np.float64)
    idx = np.random.default_rng(seed).choice(len(pts), n, replace=False)
    sub = o3d.geometry.PointCloud()
    sub.points = o3d.utility.Vector3dVector(pts[idx])
    return sub


@pytest.mark.parametrize("cloud", PIPELINE_CLOUDS)
def test_riemannian_graph_is_connected(cloud):
    G = sgk.create_riemannian_graph(_subsample(cloud), method="knn", nearest_neighbors=8)
    assert G.number_of_nodes() == 2000
    assert nx.is_connected(G)


@pytest.mark.parametrize("cloud", PIPELINE_CLOUDS)
def test_spectral_clustering_splits_stem_and_leaf(cloud):
    sub = _subsample(cloud)
    G = sgk.create_riemannian_graph(sub, method="knn", nearest_neighbors=8)
    pcg = PointCloudGraph(G)
    pcg.pcd = sub
    pcg.compute_graph_eigenvectors()
    pcg.compute_gradient_of_fiedler_vector(method="by_fiedler_weight")
    clusters_centers = pcg.clustering_by_kmeans_using_gradient_norm(
        export_in_labeled_point_cloud=True, number_of_clusters=4
    )
    assert len(clusters_centers) >= 2  # stem vs leaf split
