#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Méthode de segmentation telle qu'expliquée dans l'article
# Hétroy-Wheeler, Casella, and Boltcheva, “Segmentation of Tree Seedling Point Clouds into Elementary Units.”


import networkx as nx
import numpy as np
import open3d as o3d

import spectral_clustering.similarity_graph as SGk
from spectral_clustering.dijkstra_segmentation import actugraph
from spectral_clustering.dijkstra_segmentation import initdijkstralitt
from spectral_clustering.dijkstra_segmentation import segmdijkstralitt


def affichesegmlitt(pcd, G, segmentdict, c):
    """Displays a 3D point cloud and its corresponding graph representation.

    This function visualizes a 3D point cloud along with its graph structure.
    It creates a graph representation where nodes correspond to the points in
    the point cloud, and edges are defined based on the provided segment
    dictionary. The visualization allows the user to see the structure of
    the graph overlayed on the point cloud.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        The 3D point cloud to be visualized.
    G : networkx.Graph
        The base graph structure associated with the point cloud.
    segmentdict : dict[int, list[int]]
        A dictionary where each key is a segment index and the value is a list
        of point indices representing a path in the graph.
    c : int
        The number of graph segments to visualize. Segments in the range
        [1, c-1] from the segment dictionary are visualized.

    """
    Gaffichage = nx.Graph()
    pts = np.array(pcd.points)
    N = len(pcd.points)
    for i in range(N):
        Gaffichage.add_node(i, pos=pts[i])
    for i in range(1, c):
        Gaffichage.add_path(segmentdict[i])
    edgelist = Gaffichage.edges
    print(Gaffichage.edges)
    cloption = o3d.visualization.RenderOption()
    graph = o3d.geometry.LineSet()
    graph.points = o3d.utility.Vector3dVector(pts)
    graph.lines = o3d.utility.Vector2iVector(edgelist)
    o3d.visualization.draw_geometries([graph, pcd])


def sortienuagelitt(pcd, G, segmentdict, c):
    """Sorts segments and labels points based on the shortest paths in a graph.

    This function processes a point cloud and a graph representing segments,
    calculates the shortest paths from specified segments to all other points,
    and labels each point in the point cloud according to the segment it is
    associated with or assigns a default label if no segment is reachable.
    It outputs a classified point cloud with these labels.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        A point cloud object representing the spatial data. It is expected
        to have points that need to be classified.
    G : networkx.classes.multidigraph.MultiDiGraph
        A graph containing nodes and edges representing segments. The nodes
        correspond to points or locations in the graph, and the edges have
        associated weights used for shortest path calculations.
    segmentdict : dict
        A dictionary where keys are segment identifiers (int or str) and
        values are lists of points or segment nodes. These represent specific
        paths or segments in the graph for label assignment.
    c : int
        A default label assigned to points that do not belong to any segment
        or do not have a reachable shortest path.

    Notes
    -----
    The function saves the resulting classified point cloud data as a text file named `pcdclassifdijkstra2.txt`.

    Raises
    ------
    ValueError
        If certain values could not be processed or matched during
        the segment identification process in the segment dictionary.
    """
    label = []
    chemintot = []
    N = len(pcd.points)
    for Seg, chemin in segmentdict.items():
        chemintot = chemintot + chemin
    # Transformation de la liste obtenue en set
    setsegments = set(chemintot)
    length, path = nx.multi_source_dijkstra(G, setsegments, weight='weight')
    for p in range(N):
        if p in length:
            # Sélection du chemin qui concerne le point d'intérêt
            j = 1
            ind = -1
            if length[p] == 0:
                parrive = p
            else:
                parrive = path[p][0]
            while ind == -1 and j < c:
                try:
                    ind = segmentdict[j].index(parrive)
                except ValueError:
                    ind = -1
                j = j + 1
            j = j - 1
            label = label + [j]
        else:
            label = label + [c]
    label = np.asarray(label)
    label = np.asarray(label.reshape(np.asarray(pcd.points).shape[0], 1), dtype=np.float64)
    pcdtabclassif = np.concatenate([np.asarray(pcd.points), label], axis=1)
    # FIXME: this should be a parameter:
    np.savetxt('pcdclassifdijkstra2.txt', pcdtabclassif, delimiter=",")


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("Data/arabi_densep_clean_segm.ply")
    r = 8
    G = SGk.create_riemannian_graph(pcd, method='knn', nearest_neighbors=r)
    # SGk.drawGraphO3D(pcd, G)
    G = actugraph(G)
    segmsource = initdijkstralitt(G)
    c = 13
    segmentdict = segmdijkstralitt(G, segmsource, c)
    affichesegmlitt(pcd, G, segmentdict, c)
    sortienuagelitt(pcd, G, segmentdict)
