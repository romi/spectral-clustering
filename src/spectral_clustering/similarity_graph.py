#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import networkx as nx
import numpy as np
import open3d as o3d


# p est le nuage de points pcd, r_nn le seuil pour la détermination des connexions
# fonction permettant la création d'un graphe grâce à la librairie networkx
def create_riemannian_graph(pcd, method='knn', nearest_neighbors=1, radius=1.):
    """Generate a similarity graph from a point cloud.

    Parameters
    ----------
    p : o3d.o3d.geometry.PointCloud
        Point cloud on which to compute the similarity graph
    method : str
        Search method : 'knn' or 'radius'
    nearest_neighbors : int
        Number of nearest neighbors for the 'knn' method
    radius : float
        Radius for the 'radius' method

    Returns
    -------
    networkx.Graph
        Similarity graph computed on the point cloud
    """
    N = len(pcd.points)
    # définition d'un arbre KD contenant tous les points
    tree = o3d.geometry.KDTreeFlann(pcd)

    pcd.estimate_normals()

    # Prise des points sous forme de tableau ndarray
    pts = np.array(pcd.points)
    normals = np.array(pcd.normals)

    # Déclaration d'un graph networkx
    G = nx.Graph()

    # On insère chaque point du nuage de points dans le graphe avec un numéro et le trio de coordonnées (pos) en attributs
    for i in range(N):
        G.add_node(i, pos=pts[i], normal=normals[i])

    # Construction des edges du graphe à partir d'un seuil
    # On part de la structure de nuage de points en KDTree
    # Cette structure dans open3d dispose de fonctions pour seuil, KNN, RKNN
    for i in range(N):
        if method == 'radius':
            [k, idxs, _] = tree.search_radius_vector_3d(pts[i], radius)
        elif method == 'knn':
            [k, idxs, _] = tree.search_knn_vector_3d(pts[i], nearest_neighbors)
        for idx in idxs:
            d = np.sqrt(np.square(pts[i][0] - pts[idx][0]) + np.square(pts[i][1] - pts[idx][1]) + np.square(
                pts[i][2] - pts[idx][2]))
            if d != 0:
                w = 1 / d
                G.add_edge(i, idx, weight=w)
                G.edges[(i, idx)]["distance"] = d

    return G


def create_connected_riemannian_graph(point_cloud, method='knn', nearest_neighbors=1, radius=1.):
    """Create a connected Riemannian graph from a point cloud using a specified method.

    This function takes a point cloud and generates a Riemannian graph based on the
    provided method and parameters. If the generated graph is not connected, it
    extracts the largest connected component from the graph, saves the corresponding
    points, and regenerates the graph. The final graph and updated point cloud are
    returned.

    Parameters
    ----------
    point_cloud : o3d.geometry.PointCloud
        Input point cloud data.
    method : str, optional
        Method to generate the Riemannian graph. Options are 'knn' (k-nearest neighbors)
        or other methods, by default 'knn'.
    nearest_neighbors : int, optional
        Number of nearest neighbors to use when creating the graph (if method is 'knn'),
        by default 1.
    radius : float, optional
        Radius parameter to define connectivity (if applicable to the method),
        by default 1.0.

    Returns
    -------
    networkx.Graph
        The connected Riemannian graph.
    o3d.geometry.PointCloud
        The updated point cloud corresponding to the connected graph.
    """
    G = create_riemannian_graph(pcd=point_cloud, method=method, nearest_neighbors=nearest_neighbors, radius=radius)
    pcd2 = point_cloud
    if nx.is_connected(G) is False:
        print('not connected')
        largest_cc = max(nx.connected_components(G), key=len)
        # creating the new pcd point clouds
        coords = np.zeros((len(largest_cc), 3))
        i = 0
        for node in largest_cc:
            coords[i, :] = G.nodes[node]['pos']
            i += 1
        path = os.getcwd()
        print(path)
        np.savetxt(path + '/New_pcd_connected.txt', coords, delimiter=' ', fmt='%f')

        pcd2 = o3d.io.read_point_cloud(path + "/New_pcd_connected.txt", format='xyz')

        G = create_riemannian_graph(pcd2, method=method, nearest_neighbors=nearest_neighbors, radius=radius)
        pcd = pcd2

    return G, pcd2


def add_label_from_pcd_file(G, file="Data/peuplier_seg.txt"):
    """Assigns clustering labels to a graph `G` from a specified Point Cloud Data (PCD) file.

    The function reads a given file containing point cloud data where clustering labels are
    extracted from the fourth column (index 3) of the file data. These labels are set as a new
    attribute `clustering_labels` of the graph `G` and are also assigned to individual nodes
    within the graph. This assignment links clustering information from the point cloud data
    to the graph structure.

    Parameters
    ----------
    G : networkx.Graph
        A graph representation of the dataset where each node will be assigned a clustering label
        based on the point cloud data loaded from the input file.
    file : str, optional
        File path to the PCD (Point Cloud Data) file in text format. By default, the path
        points to a specific `.txt` dataset: "/Users/katiamirande/PycharmProjects/
        Spectral_clustering_0/Data/peuplier_seg.txt".

    Notes
    -----
    The function expects that the input data in the provided file contains at least four columns,
    where the fourth column (index 3) corresponds to the clustering labels. The labels are expected
    to be integers. The function specifically reshapes these labels before assigning them to the
    graph.

    """
    label = np.loadtxt(file, delimiter=',')
    l = label[:, 3].astype('int32')
    G.clustering_labels = l.reshape((len(l), 1))
    j = 0
    for n in G.nodes:
        G.nodes[n]['clustering_labels'] = l[j]
        j += 1


def add_label_from_separate_file(G,
                                 file="Data/rawLabels_Arabido_0029.txt"):
    """Adds clustering labels from a separate file to the graph `G`.

    This function reads a file containing clustering labels, processes it into an
    integer array, and assigns these labels both as a graph-level attribute and
    node-level attributes. The labels for nodes are assigned sequentially based
    on the order of nodes in the graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph to which clustering labels will be assigned.
    file : str, optional
        Path to the file containing clustering labels. The file is expected to
        contain comma-separated values. Defaults to
        "
        Data/rawLabels_Arabido_0029.txt".

    Notes
    -----
    - The labels in the file are expected to align with the order of nodes in the
      graph `G`.
    - The clustering labels are stored as an attribute for the graph (`clustering_labels`)
      and as an attribute for each node (`G.nodes[n]['clustering_labels']`).
    """
    label = np.loadtxt(file, delimiter=',')
    l = label.astype('int32')
    G.clustering_labels = l.reshape((len(l), 1))
    j = 0
    for n in G.nodes:
        G.nodes[n]['clustering_labels'] = l[j]
        j += 1


# affichage via open3D
# En entrée : p nuage de points
def draw_graph_open3d(pcd, G):
    """Visualize a graph overlaid on a 3D point cloud using o3d.

    This function creates a line set object from the provided graph edges
    and associates it with the nodes represented by the points in the
    point cloud. The resulting visualization is displayed in an Open3D
    viewer to provide an interactive 3D representation of the graph.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        A 3D point cloud object where each point is treated as a node of
        the graph.
    G : networkx.Graph
        A graph object, containing edges as tuples of integers
        representing connections between the indices of points in the
        provided point cloud.

    """
    graph = o3d.geometry.LineSet()
    graph.points = pcd.points
    graph.lines = o3d.utility.Vector2iVector(G.edges)
    o3d.visualization.draw_geometries([graph])


# affichage du graphe via CellComplex
# en entrée : la matrice d'adjacence (matrice de similarité) et le nuage de points importé/lu via open3D
def draw_graph_cellcomplex(pcd, G, pcd_to_superimpose):
    """Draws a graph cell complex visualization using the provided point cloud, graph and an additional point cloud for superimposition.

    The function generates a 3D representation of the graph with edges and vertices colored and styled
    using predefined colormaps and glyph configurations, and displays it using VTK.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud or numpy.ndarray
        The primary point cloud used to define the coordinates of the graph vertices.
        It can either be an `o3d.geometry.PointCloud` object or a numpy array of
        point coordinates.
    G : networkx.Graph
        The input graph whose nodes and edges will be visualized. Nodes must
        correspond to elements in the `pcd` parameter, and edges define the
        connections to be drawn.
    pcd_to_superimpose : o3d.geometry.PointCloud or numpy.ndarray
        An additional point cloud that will be visualized as a superimposed set of
        points on the graph. This provides a way to compare or highlight a separate
        dataset in relation to the graph cell complex.
    """
    if type(pcd) is o3d.geometry.PointCloud:
        pcdtab = np.asarray(pcd.points)
    else:
        pcdtab = pcd
    # s, t = np.meshgrid(np.arange(len(pcdtab)), np.arange(len(pcdtab)))
    # sources = s[simatrix > 0]
    # targets = t[simatrix > 0]
    # sources, targets = sources[sources < targets], targets[sources < targets]

    from cellcomplex.property_topomesh.creation import edge_topomesh
    from cellcomplex.property_topomesh.visualization.vtk_actor_topomesh import VtkActorTopomesh
    from cellcomplex.property_topomesh.visualization.vtk_tools import vtk_display_actors
    from cellcomplex.property_topomesh.analysis import compute_topomesh_property
    from cellcomplex.property_topomesh.creation import vertex_topomesh

    topomesh = edge_topomesh(np.array([e for e in G.edges if e[0] != e[1]]),
                             dict(zip(np.asarray([n for n in G.nodes]), pcdtab)))

    compute_topomesh_property(topomesh, 'length', 1)

    edge_actor = VtkActorTopomesh()
    edge_actor.set_topomesh(topomesh, 1, property_name='length')
    edge_actor.line_glyph = 'tube'
    edge_actor.update(colormap="cool")

    vertex_actor = VtkActorTopomesh()
    vertex_actor.set_topomesh(topomesh, 0)
    # vertex_actor.point_glyph = 'point'
    vertex_actor.point_glyph = 'sphere'
    vertex_actor.glyph_scale = 2
    vertex_actor.update(colormap="Reds")

    point_cloud_actor = VtkActorTopomesh(vertex_topomesh(pcd_to_superimpose), 0)
    point_cloud_actor.point_glyph = 'point'
    point_cloud_actor.update(colormap="Blues")

    vtk_display_actors([vertex_actor.actor, edge_actor.actor, point_cloud_actor.actor], background=(0.9, 0.9, 0.9))


def export_eigenvectors_on_pointcloud(pcd, keigenvec, k, filename='vecteurproprecol.txt'):
    """Export eigenvectors associated with a specific eigenvalue to a file for visualization alongside the point cloud data.

    This function computes a new array combining the coordinates of the points in the point cloud
    and the eigenvectors corresponding to a specific eigenvalue. The combined data is then saved
    to a CSV file for potential visualization purposes, e.g., in a tool like CloudCompare.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The input point cloud from which coordinates are extracted.
    keigenvec : numpy.ndarray
        The matrix of eigenvectors where each column corresponds to an eigenvalue.
    k : int
        The index of the eigenvalue whose eigenvectors will be used.
    filename : str, optional
        The name of the output file where the combined data will be saved. Default is
        'vecteurproprecol.txt'.
    """
    # Le facteur multiplicatif est présent uniquement pour pouvoir éventuellement mieux afficher les couleurs/poids dans CloudCompare
    #
    label = keigenvec[:, k]
    size = label.shape[0]
    label = np.asarray(label.reshape(size, 1), dtype=np.float64)
    pcd = np.array(pcd.points)
    pcdtabvecteurpropre = np.concatenate([pcd, label], axis=1)
    np.savetxt(filename, pcdtabvecteurpropre, delimiter=',')


def export_pointcloud_on_eigenvectors_3d(keigenvec, vec1, vec2, vec3, filename='espacespec.txt'):
    """Export a 3D point cloud using specific eigenvector indices.

    This function extracts points from the given eigenvector matrix based on
    specified indices and exports the resulting 3D coordinates into a text file
    in a tabular format. The saved file consists of points where each row
    contains three values representing a single point in 3D space. The points
    are separated by commas for easier parsing.

    Parameters
    ----------
    keigenvec : ndarray
        A 2D array where each column represents an eigenvector and rows
        represent points in the dataset. It is expected to have at least
        three columns to extract 3D coordinates.
    vec1 : int
        Index of the eigenvector to use as the x-coordinate.
    vec2 : int
        Index of the eigenvector to use as the y-coordinate.
    vec3 : int
        Index of the eigenvector to use as the z-coordinate.
    filename : str, optional
        The name of the file to save the output. The default is
        'espacespec.txt', and the data in the file will be saved
        in a comma-separated value format.
    """
    pts = keigenvec[:, [vec1, vec2, vec3]]
    pts = pts.reshape(keigenvec.shape[0], 3)
    np.savetxt(filename, pts, delimiter=',')
