#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import sklearn.cluster as skc


def segment_a_node_using_attribute(quotientgraph, quotient_node_to_work, attribute_to_work_with='direction_gradient',
                                   method='OPTICS'):
    """Perform clustering on a specified node of the QuotientGraph, updating the associated nodes
    of the PointCloudGraph accordingly.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        The QuotientGraph object containing the graph data.
    quotient_node_to_work : int
        The node in the QuotientGraph to cluster.
    attribute_to_work_with : str, optional
        The attribute of the PointCloudGraph nodes to use for clustering. Default is 'direction_gradient'.
    method : str, optional
        The clustering method to use. Supported methods are 'KMEANS' and 'OPTICS'. Default is 'OPTICS'.

    Notes
    -----
    - The function clusters the nodes of the PointCloudGraph that are part of the given 'quotient_node_to_work'.
    - Updates the `quotient_graph_node` attribute of each PointCloudGraph node to reflect the new clustering.

    See Also
    --------
    sklearn.cluster.KMeans : K-means clustering algorithm.
    sklearn.cluster.OPTICS : OPTICS clustering algorithm for density-based clustering.
    """
    G = quotientgraph.point_cloud_graph
    nw = quotient_node_to_work

    # make list of nodes inside the quotient graph node to work with
    list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == nw]

    # Use of a function to segment
    X = np.zeros((len(list_of_nodes), 3))

    for i in range(len(list_of_nodes)):
        X[i] = G.nodes[list_of_nodes[i]][attribute_to_work_with]

    if method == 'KMEANS':
        clustering = skc.KMeans(n_clusters=2, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(X)
    if method == 'OPTICS':
        clustering = skc.OPTICS(min_samples=100).fit(X)

    # Instead of having 0 or 1, this gives a number not already used in the quotient graph to name the nodes
    clustering_labels = clustering.labels_[:, np.newaxis] + max(quotientgraph.nodes) + 1

    # Integration of the new labels in the quotient graph and updates

    for i in range(len(list_of_nodes)):
        G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]


def segment_several_nodes_using_attribute(quotientgraph, list_quotient_node_to_work=[], algo='OPTICS',
                                          para_clustering_meth=100, attribute='direction_gradient'):
    """Perform clustering on multiple specified nodes of the QuotientGraph, updating the class of the underlying
    nodes in the associated PointCloudGraph.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        The QuotientGraph object containing the graph data.
    list_quotient_node_to_work : list of int, optional
        A list of node indices in the QuotientGraph to be clustered. Default is an empty list, which means no nodes are clustered.
    algo : {'kmeans', 'OPTICS'}, optional
        The clustering algorithm to use. Supported values are:
        - 'kmeans': K-means clustering,
        - 'OPTICS': Density-based clustering.
        Default is 'OPTICS'.
    para_clustering_meth : int, optional
        The parameter needed for the chosen clustering algorithm:
        - For 'OPTICS', it specifies the minimum sample size (`min_samples`) to form a cluster.
        - For 'kmeans', it specifies the number of clusters (`n_clusters`) to generate.
        Default is 100.
    attribute : str, optional
        The node attribute from the PointCloudGraph to be used for clustering. Default is 'direction_gradient'.

    Notes
    -----
    - The function selects the nodes from the PointCloudGraph corresponding to the specified nodes in
      `list_quotient_node_to_work` from the QuotientGraph.
    - The clustering results are assigned as new values for the `quotient_graph_node` attribute of the
      PointCloudGraph nodes.
    - The updated clustering is applied to the PointCloudGraph associated with the QuotientGraph.
      Only the PointCloudGraph is directly modified.
    """

    lw = list_quotient_node_to_work
    G = quotientgraph.point_cloud_graph
    # make list of nodes inside the different quotient graph nodes to work with
    list_of_nodes = []
    for qnode in lw:
        list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
        list_of_nodes += list_of_nodes_each

    # Use of a function to segment
    X = np.zeros((len(list_of_nodes), 3))

    for i in range(len(list_of_nodes)):
        X[i] = G.nodes[list_of_nodes[i]][attribute]

    if algo == 'OPTICS':
        clustering = skc.OPTICS(min_samples=para_clustering_meth).fit(X)
    elif algo == 'kmeans':
        clustering = skc.KMeans(n_clusters=para_clustering_meth, init='k-means++', n_init=20, max_iter=300,
                                tol=0.0001).fit(X)

    # Selection of max value for 'quotient_graph_node' in G
    q_g = nx.get_node_attributes(G, 'quotient_graph_node')
    q_g_values = list(q_g.values())

    # Instead of having 0 or 1, this gives a number not already used in the quotient graph to name the nodes
    clustering_labels = clustering.labels_[:, np.newaxis] + max(q_g_values) + 2

    # Integration of the new labels in the quotient graph and updates
    for i in range(len(list_of_nodes)):
        G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]

    quotientgraph.point_cloud_graph = G


def segment_each_cluster_by_optics(quotientgraph, list_leaves=[], leaves_out=True,
                                   attribute_to_work='direction_gradient'):
    """Perform clustering on the nodes of the QuotientGraph, updating the corresponding nodes
    in the associated PointCloudGraph based on the specified attribute.

    Each node of the QuotientGraph is processed and re-clustered based on the values of a specified
    attribute in the associated PointCloudGraph. The clustering is performed using an external
    function, which updates the `quotient_graph_node` attribute for those nodes. The QuotientGraph
    is updated accordingly after the clustering.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        The QuotientGraph object containing the graph structure and associated PointCloudGraph data.
    list_leaves : list of int, optional, default=[]
        A list of node indices in the QuotientGraph to be excluded from clustering
        when `leaves_out` is True.
    leaves_out : bool, optional, default=True
        If True, end nodes (or leaves) of the QuotientGraph in `list_leaves` are excluded
        from the clustering process. If False, all nodes are included regardless of `list_leaves`.
    attribute_to_work : str, optional, default='direction_gradient'
        The attribute in the PointCloudGraph associated with the QuotientGraph to be
        used for re-clustering the nodes.

    Notes
    -----
    - The function updates the `quotient_graph_node` attribute for each node in the PointCloudGraph
      to reflect the new clustering results.
    - The QuotientGraph structure to align with the updated clustering.
    - Only nodes in the QuotientGraph with an `intra_class_node_number` greater than 100
      are processed for clustering.
    - The function delegates the clustering task to `segment_several_nodes_using_attribute`,
      which performs clustering using the specified attribute and method.

    See Also
    --------
    segment_several_nodes_using_attribute : Perform clustering on multiple specified nodes of the QuotientGraph.


    """
    # Put the "end" nodes in a list
    for n in quotientgraph:
        if leaves_out:
            if n not in list_leaves and quotientgraph.nodes[n]['intra_class_node_number'] > 100:
                segment_several_nodes_using_attribute(quotientgraph=quotientgraph, list_quotient_node_to_work=[n],
                                                      attribute=attribute_to_work)
        elif leaves_out is False and quotientgraph.nodes[n]['intra_class_node_number'] > 100:
            segment_several_nodes_using_attribute(quotientgraph=quotientgraph, list_quotient_node_to_work=[n],
                                                  attribute=attribute_to_work)

    quotientgraph.rebuild(quotientgraph.point_cloud_graph)
