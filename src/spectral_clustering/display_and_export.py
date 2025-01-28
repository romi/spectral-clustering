#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d


def export_quotient_graph_attribute_on_point_cloud(qg, attribute, name='', directory="."):
    """Exports attributes from a quotient graph to a point cloud.

    Parameters
    ----------
    qg : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph containing point cloud and attribute data.
    attribute : str
        The attribute from the quotient graph nodes to be exported.
    name : str, optional
        A suffix to append to the output file name.
    directory : str or pathlib.Path, optional
        A path to the output directory. Defaults to the current working directory.
    """
    labels_from_qg = np.zeros((len(qg.point_cloud_graph), 4))
    i = 0
    G = qg.point_cloud_graph
    for n in G.nodes:
        labels_from_qg[i, 0:3] = G.nodes[n]['pos']
        labels_from_qg[i, 3] = qg.nodes[G.nodes[n]['quotient_graph_node']][attribute]
        i += 1
    directory = Path(directory)
    fname = 'pcd_' + attribute + name + '.txt'
    np.savetxt(directory / fname, labels_from_qg, delimiter=",")


def export_clustering_labels_on_point_cloud(pcd_g, filename="pcd_clustered.txt"):
    """Exports clustering labels of a graph to a point cloud.

    Parameters
    ----------
    pcd_g : spectral_clustering.pointcloudgraph.PointCloudGraph
        The graph containing nodes coordinates and clustering labels.
    filename : str or pathlib.Path, optional
        The file name for the output.
    """
    pcd_clusters = np.concatenate([pcd_g.nodes_coords, pcd_g.clustering_labels], axis=1)
    np.savetxt(filename, pcd_clusters, delimiter=",")
    print("Export du nuage avec les labels de cluster")


def export_anything_on_point_cloud(pcd_g, attribute, filename="pcd_attribute.txt"):
    """Exports a specified attribute to the point cloud.

    Parameters
    ----------
    pcd_g : spectral_clustering.pointcloudgraph.PointCloudGraph
        The graph containing nodes coordinates.
    attribute : ndarray
        An array representing the attribute values to export.
    filename : str or pathlib.Path, optional
        The name of the output file.
    """
    if attribute.ndim == 1:
        attribute = attribute[:, np.newaxis]
    pcd_attribute = np.concatenate([pcd_g.nodes_coords, attribute], axis=1)
    np.savetxt(filename, pcd_attribute, delimiter=",")
    print("Export du nuage avec les attributs demandés")


def export_gradient_of_fiedler_vector_on_pointcloud(pcd_g, filename="pcd_vp2_grad.txt"):
    """Exports the gradient of the Fiedler vector to a point cloud.

    Parameters
    ----------
    pcd_g : spectral_clustering.pointcloudgraph.PointCloudGraph
        The graph containing nodes coordinates and fiedler gradient data.
    filename : str, optional
        The file name for the output.
    """
    pcd_vp2_grad = np.concatenate([pcd_g.nodes_coords, pcd_g.gradient_on_fiedler], axis=1)
    np.savetxt(filename, pcd_vp2_grad, delimiter=",")
    print("Export du nuage avec gradient du vecteur propre 2")


def export_fiedler_vector_on_pointcloud(G, filename="pcd_vp2.txt"):
    """Exports the Fiedler vector to a point cloud.

    Parameters
    ----------
    G : spectral_clustering.pointcloudgraph.PointCloudGraph
        The graph containing nodes coordinates and Fiedler vector data.
    filename : str or pathlib.Path, optional
        The file name for the output.
    """
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    np.savetxt(filename, pcd_vp2, delimiter=",")
    print("Export du nuage avec le vecteur propre 2")


def export_some_graph_attributes_on_point_cloud(pcd_g, graph_attribute='quotient_graph_node',
                                                filename='graph_attribute.txt'):
    """Exports specific graph attributes to a point cloud.

    Parameters
    ----------
    pcd_g : spectral_clustering.pointcloudgraph.PointCloudGraph
        The graph containing nodes and the requested attribute.
    graph_attribute : str, optional
        The name of the graph attribute to export.
    filename : str or pathlib.Path, optional
        The file name for the output.
    """
    G = pcd_g
    new_classif = np.asarray(list((dict(G.nodes(data=graph_attribute)).values())))
    new_classif = new_classif[:, np.newaxis]
    export_anything_on_point_cloud(G, attribute=new_classif, filename=filename)


def display_and_export_graph_of_fiedler_vector(pcd_g, filename="fiedler_vector.png", sorted_by_fiedler_vector=True):
    """Displays and exports a plot of the Fiedler vector.

    Parameters
    ----------
    pcd_g : spectral_clustering.pointcloudgraph.PointCloudGraph
        The graph containing Fiedler vector data.
    filename : str or pathlib.Path, optional
        The file name for saving the output plot.
    sorted_by_fiedler_vector : bool, optional
        Whether to sort the data by the Fiedler vector values.
    """
    vp2 = pcd_g.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([pcd_g.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_sort_by_vp2 = pcd_vp2[pcd_vp2[:, 3].argsort()]
    figure = plt.figure(0)
    figure.clf()
    figure.gca().set_title("fiedler vector")
    if sorted_by_fiedler_vector:
        figure.gca().scatter(range(len(pcd_vp2_sort_by_vp2)), pcd_vp2_sort_by_vp2[:, 3], color='blue')
    else:
        figure.gca().scatter(range(len(pcd_vp2)), pcd_vp2[:, 3], color='blue')
    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du vecteur propre 2")


def display_and_export_graph_of_gradient_of_fiedler_vector(pcd_g, filename="Gradient_of_fiedler_vector.png",
                                                           sorted_by_fiedler_vector=True, sorted_by_gradient=False):
    """Displays and exports a plot of the gradient of the Fiedler vector.

    Parameters
    ----------
    pcd_g : spectral_clustering.pointcloudgraph.PointCloudGraph
        The graph containing Fiedler vector gradient data.
    filename : str or pathlib.Path, optional
        The file name for saving the output plot.
    sorted_by_fiedler_vector : bool, optional
        Whether to sort the data by the Fiedler vector values.
    sorted_by_gradient : bool, optional
        Whether to sort the data by the gradient values.
    """
    vp2 = pcd_g.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([pcd_g.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_grad_vp2 = np.concatenate([pcd_vp2, pcd_g.gradient_on_fiedler], axis=1)
    pcd_vp2_grad_vp2_sort_by_vp2 = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 3].argsort()]
    pcd_vp2_grad_vp2_sort_by_grad = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 4].argsort()]
    figure = plt.figure(1)
    figure.clf()
    figure.gca().set_title("Gradient of fiedler vector")
    if sorted_by_fiedler_vector and not sorted_by_gradient:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_vp2[:, 4], color='blue')
    elif not sorted_by_fiedler_vector and not sorted_by_gradient:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2)), pcd_vp2_grad_vp2[:, 4], color='blue')
    if sorted_by_gradient:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_grad[:, 4],
                             color='blue')
    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du gradient")


def display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name='quotient_graph_matplotlib',
                                                 data_on_nodes='intra_class_node_number', data=True,
                                                 attributekmeans4clusters=False, round=False,
                                                 directory="."):
    """Generates a matplotlib visualization for the given quotient graph and exports it to a file.

    Visualizes the input graph using NetworkX's drawing functions with matplotlib. The appearance of the
    graph, including node colors, labels, and sizes, can be customized based on the given parameters.
    The plot is saved to the specified file.

    Parameters
    ----------
    qg : spectral_clustering.quotientgraph.QuotientGraph
        The input graph to be visualized and exported.
    node_sizes : int, optional
        Size of the nodes in the plot, by default ``20``.
    name : str, optional
        The name of the file where the graph visualization plot is saved.
    data_on_nodes : str, optional
        The name of the attribute on the nodes to be displayed as labels, by default
        'intra_class_node_number'.
    data : bool, optional
        Determines whether to use node-specific data attributes in visualization, by default ``True``.
    attributekmeans4clusters : bool, optional
        If ``True``, the visualization integrates clustering-specific attributes and uses k-means
        related coloring, by default ``False``.
    round : bool, optional
        If ``True``, rounds the node attribute values for display, by default ``False``.
    directory : str or pathlib.Path, optional
        A path to the output directory. Defaults to the current working directory.
    """
    figure = plt.figure(0)
    figure.clf()
    graph_layout = nx.kamada_kawai_layout(qg)
    colormap = 'jet'

    if attributekmeans4clusters and data:
        labels_from_attributes = dict(qg.nodes(data=data_on_nodes))
        # Rounding the data to allow an easy display
        for dict_value in labels_from_attributes:
            labels_from_attributes[dict_value] = round(labels_from_attributes[dict_value], 2)
        node_color_from_attribute = dict(qg.nodes(data='seed_colors')).values()
        node_color = [qg.nodes[i]['kmeans_labels'] / 4 for i in qg.nodes()]
        nx.drawing.nx_pylab.draw_networkx(qg,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=True,
                                          node_size=node_sizes,
                                          node_color=node_color_from_attribute,
                                          labels=labels_from_attributes,
                                          cmap=plt.get_cmap(colormap))

    elif attributekmeans4clusters is False and data is False:
        nx.drawing.nx_pylab.draw_networkx(qg,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=False,
                                          node_size=node_sizes,
                                          node_color="r",
                                          cmap=plt.get_cmap(colormap))

    elif attributekmeans4clusters is False and data:
        labels_from_attributes = dict(qg.nodes(data=data_on_nodes))
        # Rounding the data to allow an easy display
        for dict_value in labels_from_attributes:
            if data_on_nodes != 'semantic_label':
                if round is True:
                    labels_from_attributes[dict_value] = round(labels_from_attributes[dict_value], 2)
        nx.drawing.nx_pylab.draw_networkx(qg,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=True,
                                          node_size=node_sizes,
                                          node_color="r",
                                          labels=labels_from_attributes,
                                          cmap=plt.get_cmap(colormap))

    # nx.drawing.nx_pylab.draw_networkx_edge_labels(quotient_graph, pos=graph_layout, font_size=20, font_family="sans-sherif")

    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    directory = Path(directory)
    fname = name + '_' + data_on_nodes + '.png'
    figure.savefig(directory / fname)
    print("Export du graphe quotient matplotlib")


def display_gradient_vector_field(pcd_g, normalized=True, scale=1., filename="gradient_vectorfield_3d.png"):
    """Display the gradient vector field of a 3D topology.

    This function visualizes the gradient vector field of a provided topology using
    arrows as glyphs in a 3D representation. It supports configurable vector
    normalization, glyph scaling, and output filename for saving the visualization
    as an image.

    Parameters
    ----------
    pcd_g : spectral_clustering.pointcloudgraph.PointCloudGraph
        The topology graph object, containing nodes with associated coordinates and gradient data.
    normalized : bool, optional
        Specifies whether the vectors should be normalized. If ``True``, the scaled
        direction gradient on the Fiedler vector is used. Otherwise, the original
        gradient is scaled with the direction gradient on the Fiedler vector.
    scale : float, optional
        Factor by which the arrow glyphs representing the gradient vectors should
        be scaled. Defaults to ``1.0``.
    filename : str or pathlib.Path, optional
        The name of the output image file where the visualization will be saved.
        Defaults to ``'gradient_vectorfield_3d.png'``.
    """
    from cellcomplex.property_topomesh.creation import vertex_topomesh
    from cellcomplex.property_topomesh.visualization.vtk_actor_topomesh import VtkActorTopomesh
    from visu_core.vtk.display import vtk_display_actors, vtk_save_screenshot_actors

    n_points = pcd_g.nodes_coords.shape[0]

    topomesh = vertex_topomesh(dict(zip(range(n_points), pcd_g.nodes_coords)))

    if normalized:
        vectors = pcd_g.direction_gradient_on_fiedler_scaled
    if normalized is False:
        vectors = pcd_g.gradient_on_fiedler * pcd_g.direction_gradient_on_fiedler_scaled

    topomesh.update_wisp_property('vector', 0, dict(zip(range(n_points), vectors)))

    actors = []

    vector_actor = VtkActorTopomesh(topomesh, degree=0, property_name='vector')
    vector_actor.vector_glyph = 'arrow'
    vector_actor.glyph_scale = scale
    vector_actor.update(colormap='Reds', value_range=(0, 0))
    actors += [vector_actor]

    # Change of background
    ren, _, _ = vtk_display_actors(actors, background=(0.9, 0.9, 0.9))
    cam = ren.GetActiveCamera()
    vtk_save_screenshot_actors(actors, image_filename=filename, camera=cam)


def draw_quotientgraph_cellcomplex(pcd, qg, pcd_g, color_attribute='quotient_graph_node',
                                   filename='graph_and_quotientgraph_3d.png'):
    """Draws and visualizes a cell complex derived from a quotient graph in 3D visualization.

    The method creates a topomesh representation of the input quotient graph, assigns vertex colors based on
    specified attributes, calculates edge properties (e.g., length), and renders the visualization.
    It allows customization of glyph type, color maps, and visual output filename.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud or numpy.ndarray
        Point cloud data, either as an Open3D PointCloud object or a NumPy array containing
        point coordinates. If provided as Open3D PointCloud, the points are extracted and used
        as input.
    qg : spectral_clustering.quotientgraph.QuotientGraph
        Quotient graph describing the cell complex structure. The graph should contain nodes
        and edges associated with the geometric representation of the input data.
    pcd_g : spectral_clustering.pointcloudgraph.PointCloudGraph
        Graph specifying additional properties such as positions ('pos') and the visualization
        attributes for nodes in 3D. This graph is used to represent the original structure.
    color_attribute : str, optional
        Name of the property to be used for coloring the vertex glyphs. Defaults to
        ``'quotient_graph_node'``. It can be changed to any node attribute present in the input
        quotient graph.
    filename : str or pathlib.Path, optional
        Path and name of the file where the rendered 3D visualization will be saved as a
        screenshot. Defaults to ``'graph_and_quotientgraph_3d.png'``.
    """
    if type(pcd) is o3d.geometry.PointCloud:
        pcdtab = np.asarray(pcd.points)
    else:
        pcdtab = pcd
    # s, t = np.meshgrid(np.arange(len(pcdtab)), np.arange(len(pcdtab)))
    # sources = s[simatrix > 0]
    # targets = t[simatrix > 0]
    # sources, targets = sources[sources < targets], targets[sources < targets]

    from cellcomplex.property_topomesh.creation import vertex_topomesh, edge_topomesh
    from cellcomplex.property_topomesh.analysis import compute_topomesh_property
    from cellcomplex.property_topomesh.visualization.vtk_actor_topomesh import VtkActorTopomesh
    from visu_core.vtk.display import vtk_display_actors, vtk_save_screenshot_actors

    topomesh = edge_topomesh(np.array([e for e in qg.edges if e[0] != e[1]]),
                             dict(zip(np.asarray([n for n in qg.nodes]), pcdtab)))

    if color_attribute == 'quotient_graph_node':
        topomesh.update_wisp_property('quotient_graph_node', 0, dict(zip([n for n in qg.nodes], [n for n in qg.nodes])))
    else:
        topomesh.update_wisp_property(color_attribute, 0,
                                      dict(
                                          zip([n for n in qg.nodes], [qg.nodes[n][color_attribute] for n in qg.nodes])))

    compute_topomesh_property(topomesh, 'length', 1)

    actors = []

    edge_actor = VtkActorTopomesh()
    edge_actor.set_topomesh(topomesh, 1, property_name='length')
    edge_actor.line_glyph = 'tube'
    edge_actor.glyph_scale = 0.33
    edge_actor.update(colormap="gray")
    actors += [edge_actor]

    vertex_actor = VtkActorTopomesh(topomesh, 0, property_name=color_attribute)
    # vertex_actor.point_glyph = 'point'
    vertex_actor.point_glyph = 'sphere'
    vertex_actor.glyph_scale = 2
    vertex_actor.update(colormap="jet")
    actors += [vertex_actor]

    graph_topomesh = vertex_topomesh(dict(zip([n for n in pcd_g.nodes], [pcd_g.nodes[n]['pos'] for n in pcd_g.nodes])))
    graph_topomesh.update_wisp_property(color_attribute, 0, dict(
        zip([n for n in pcd_g.nodes], [pcd_g.nodes[n][color_attribute] for n in pcd_g.nodes])))
    point_cloud_actor = VtkActorTopomesh(graph_topomesh, 0, property_name=color_attribute)
    point_cloud_actor.point_glyph = 'point'
    point_cloud_actor.update(colormap="jet")
    actors += [point_cloud_actor]

    ren, _, _ = vtk_display_actors(actors, background=(0.9, 0.9, 0.9))
    cam = ren.GetActiveCamera()
    vtk_save_screenshot_actors(actors, image_filename=filename, camera=cam)


def draw_quotientgraph_matplotlib_3D(nodes_coords_moy, qg):
    """Draws a 3D visualization of a quotient graph using Matplotlib.

    The function takes in the 3D coordinates of nodes and a quotient graph object, and visualizes the
    nodes and edges of the graph in three dimensions.

    Parameters
    ----------
    nodes_coords_moy : numpy.ndarray
        A 2D numpy array of shape `(n, 3)`, where n represents the number of graph nodes.
        Each row corresponds to the 3D coordinates of a node.
    qg : spectral_clustering.quotientgraph.QuotientGraph
        A quotient graph to be visualized.

    Notes
    -----
    The function uses a matplotlib 3D scatter plot to render the nodes of the graph
    and draws lines connecting the nodes to represent edges. Node coordinates are
    specified in the `nodes_coords_moy` parameter, while edge connectivity is
    derived from the `QG` graph structure.
    """
    from mpl_toolkits.mplot3d import Axes3D

    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        for i in range(nodes_coords_moy.shape[0]):
            xi = nodes_coords_moy[i, 0]
            yi = nodes_coords_moy[i, 1]
            zi = nodes_coords_moy[i, 2]

            ax.scatter(xi, yi, zi, s=10 ** 2, edgecolors='k', alpha=0.7)

        for i, j in enumerate(qg.edges()):
            corresp = dict(zip(qg.nodes, range(len(qg.nodes))))
            x = np.array((nodes_coords_moy[corresp[j[0]], 0], nodes_coords_moy[corresp[j[1]], 0]))
            y = np.array((nodes_coords_moy[corresp[j[0]], 1], nodes_coords_moy[corresp[j[1]], 1]))
            z = np.array((nodes_coords_moy[corresp[j[0]], 2], nodes_coords_moy[corresp[j[1]], 2]))
            ax.plot(x, y, z, c='black', alpha=0.5)
        ax.view_init(elev=30)
        ax.set_axis_off()
        plt.show()
