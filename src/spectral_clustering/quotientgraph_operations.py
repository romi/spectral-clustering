#!/usr/bin/env python
# -*- coding: utf-8 -*-

import operator

import networkx as nx
import numpy as np

from spectral_clustering.display_and_export import display_and_export_quotient_graph_matplotlib
from spectral_clustering.display_and_export import export_some_graph_attributes_on_point_cloud


def delete_small_clusters(quotientgraph, min_number_of_element_in_a_quotient_node=50):
    """
    Merges small clusters in a quotient graph with their largest neighboring cluster. This function modifies the
    quotient graph in place by removing clusters with fewer intra-class nodes than a specified threshold. The
    nodes of these removed clusters are reassigned to their neighboring cluster with the highest number of
    intra-class nodes. The edges and attributes of the remaining clusters are updated accordingly.

    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        A graph representing the quotient graph structure that holds clusters as nodes,
        defined by the aggregated properties such as number of intra-class nodes, intra-class edge weights,
        or inter-class edge properties.
    min_number_of_element_in_a_quotient_node: int, optional
        Minimum threshold specifying the number of intra-class elements required for a cluster to persist.
        Clusters with a count below this threshold are merged with their most significant neighboring cluster.
        Default is ``50``.
    """
    G = quotientgraph.point_cloud_graph
    nodes_to_remove = []
    for u in quotientgraph.nodes:
        if quotientgraph.nodes[u]['intra_class_node_number'] < min_number_of_element_in_a_quotient_node:
            adjacent_clusters = [n for n in quotientgraph[u]]
            max_number_of_nodes_in_adjacent_clusters = 0
            for i in range(len(adjacent_clusters)):
                if quotientgraph.nodes[adjacent_clusters[i]][
                    'intra_class_node_number'] > max_number_of_nodes_in_adjacent_clusters:
                    max_number_of_nodes_in_adjacent_clusters = quotientgraph.nodes[adjacent_clusters[i]][
                        'intra_class_node_number']
                    new_cluster = adjacent_clusters[i]
            # Opération de fusion du petit cluster avec son voisin le plus conséquent.
            # Mise à jour des attributs de la grande classe et suppression de la petite classe
            quotientgraph.nodes[new_cluster]['intra_class_node_number'] += quotientgraph.nodes[u][
                'intra_class_node_number']
            quotientgraph.nodes[new_cluster]['intra_class_edge_weight'] += (
                        quotientgraph.nodes[u]['intra_class_edge_weight']
                        + quotientgraph.edges[new_cluster, u][
                            'inter_class_edge_weight'])
            quotientgraph.nodes[new_cluster]['intra_class_edge_number'] += (
                        quotientgraph.nodes[u]['intra_class_edge_number']
                        + quotientgraph.edges[new_cluster, u][
                            'inter_class_edge_number'])

            # Mise à jour du lien avec le PointCloudGraph d'origine
            for v in G.nodes:
                if G.nodes[v]['quotient_graph_node'] == u:
                    G.nodes[v]['quotient_graph_node'] = new_cluster

            # Mise à jour des edges
            for i in range(len(adjacent_clusters)):
                if quotientgraph.has_edge(new_cluster, adjacent_clusters[i]) is False:
                    quotientgraph.add_edge(new_cluster, adjacent_clusters[i],
                                           inter_class_edge_weight=quotientgraph.edges[u, adjacent_clusters[i]][
                                               'inter_class_edge_weight'],
                                           inter_class_edge_number=quotientgraph.edges[u, adjacent_clusters[i]][
                                               'inter_class_edge_number'])
                elif quotientgraph.has_edge(new_cluster, adjacent_clusters[i]) and new_cluster != adjacent_clusters[i]:
                    quotientgraph.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                        quotientgraph.edges[u, adjacent_clusters[i]]['inter_class_edge_weight']
                    quotientgraph.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                        quotientgraph.edges[u, adjacent_clusters[i]]['inter_class_edge_number']
            nodes_to_remove.append(u)

    quotientgraph.remove_nodes_from(nodes_to_remove)


def update_quotient_graph_attributes_when_node_change_cluster(quotientgraph, old_cluster, new_cluster, node_to_change):
    """Updates the attributes of a quotient graph after a node changes its cluster.

    This function modifies all the relevant attributes of the quotient graph
    and ensures consistency after a node transitions from one cluster to another.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph object whose attributes need to be updated. It contains
        the graph structure as well as attribute data.
    old_cluster : int
        The original cluster to which the node belonged.
    new_cluster : int
        The new cluster to which the node is assigned.
    node_to_change : int
        The node that transitions from one cluster to another.
    """

    G = quotientgraph.point_cloud_graph
    # update Quotient Graph attributes
    # Intra_class_node_number
    # print(old_cluster)
    # print(new_cluster)
    quotientgraph.nodes[old_cluster]['intra_class_node_number'] += -1
    quotientgraph.nodes[new_cluster]['intra_class_node_number'] += 1

    # intra_class_edge_weight, inter_class_edge_number
    for ng in G[node_to_change]:
        if old_cluster != G.nodes[ng]['quotient_graph_node'] and new_cluster == G.nodes[ng]['quotient_graph_node']:
            quotientgraph.nodes[new_cluster]['intra_class_edge_weight'] += G.edges[ng, node_to_change]['weight']
            quotientgraph.nodes[new_cluster]['intra_class_edge_number'] += 1
            quotientgraph.edges[old_cluster, new_cluster]['inter_class_edge_weight'] += - G.edges[ng, node_to_change][
                'weight']
            quotientgraph.edges[old_cluster, new_cluster]['inter_class_edge_number'] += - 1

        if old_cluster == G.nodes[ng]['quotient_graph_node'] and new_cluster != G.nodes[ng]['quotient_graph_node']:
            quotientgraph.nodes[old_cluster]['intra_class_edge_weight'] += -G.edges[ng, node_to_change]['weight']
            quotientgraph.nodes[old_cluster]['intra_class_edge_number'] += -1
            cluster_adj = G.nodes[ng]['quotient_graph_node']
            if quotientgraph.has_edge(new_cluster, cluster_adj) is False:
                quotientgraph.add_edge(new_cluster, cluster_adj, inter_class_edge_weight=None,
                                       inter_class_edge_number=None)
            quotientgraph.edges[new_cluster, cluster_adj]['inter_class_edge_weight'] += G.edges[ng, node_to_change][
                'weight']
            quotientgraph.edges[new_cluster, cluster_adj]['inter_class_edge_number'] += 1

        if old_cluster != G.nodes[ng]['quotient_graph_node'] and new_cluster != G.nodes[ng]['quotient_graph_node']:
            cluster_adj = G.nodes[ng]['quotient_graph_node']
            if quotientgraph.has_edge(new_cluster, cluster_adj) is False:
                quotientgraph.add_edge(new_cluster, cluster_adj,
                                       inter_class_edge_weight=G.edges[ng, node_to_change]['weight'],
                                       inter_class_edge_number=1)
            else:
                quotientgraph.edges[new_cluster, cluster_adj]['inter_class_edge_weight'] += G.edges[ng, node_to_change][
                    'weight']
                quotientgraph.edges[new_cluster, cluster_adj]['inter_class_edge_number'] += 1
            if new_cluster != old_cluster:
                quotientgraph.edges[old_cluster, cluster_adj]['inter_class_edge_weight'] += - \
                    G.edges[ng, node_to_change]['weight']
                quotientgraph.edges[old_cluster, cluster_adj]['inter_class_edge_number'] += - 1


def check_connectivity_of_modified_cluster(pointcloudgraph, old_cluster, new_cluster, node_to_change):
    """Checks the connectivity of the old cluster after a node has been reassigned to a new cluster.

    When a node is moved from one cluster to another in the distance-based graph, this function verifies
    whether the old cluster of the node remains connected or gets split into multiple disconnected
    parts.

    Parameters
    ----------
    pointcloudgraph : spectral_clustering.pointcloudgraph.PointCloudGraph
        The graph representing the point cloud and its connections.
    old_cluster : int
        The original cluster identifier of the node.
    new_cluster : int
        The new cluster identifier of the node.
    node_to_change : int
        The node that has been reassigned to a different cluster.

    Returns
    -------
    bool
        ``True`` if the old cluster remains connected after the node is reassigned, ``False`` otherwise.
    """
    G = pointcloudgraph
    # Build a dictionary which associate each node of the PointCloudGraph with its cluster/node in the QuotientGraph
    d = dict(G.nodes(data='quotient_graph_node', default=None))
    # Extract points of the old_cluster
    points_of_interest = [key for (key, value) in d.items() if value == old_cluster]
    # Handling the case where this cluster has completely disappeared
    if not points_of_interest:
        result = True
    else:
        # Creation of the subgraph and test
        subgraph = G.subgraph(points_of_interest)
        result = nx.is_connected(subgraph)

    return result


# def rebuilt_topological_quotient_graph


def collect_quotient_graph_nodes_from_pointcloudpoints(quotient_graph, list_of_points=[]):
    """Collect unique quotient graph nodes corresponding to the given list of point cloud points.

    This function retrieves the quotient graph nodes associated with the provided points
    in the point cloud graph. It extracts the nodes from the quotient graph associated
    with each given point in the graph and eliminates duplicates to return a unique
    list of quotient graph nodes.

    Parameters
    ----------
    quotient_graph : spectral_clustering.quotientgraph.QuotientGraph
        A graph-like data structure with a `point_cloud_graph` attribute. The
        `point_cloud_graph` is expected to have nodes with a `quotient_graph_node`
        attribute.
    list_of_points : list, optional
        A list of specific point cloud node identifiers whose associated quotient
        graph nodes need to be collected. Default is an empty list.

    Returns
    -------
    list
        A list of unique quotient graph nodes corresponding to the input point
        cloud points.
    """
    G = quotient_graph.point_cloud_graph
    listi = []
    for node in list_of_points:
        listi.append(G.nodes[node]['quotient_graph_node'])

    list_of_clusters = list(set(listi))

    return list_of_clusters


def collect_quotient_graph_nodes_from_pointcloudpoints_majority(quotient_graph, list_of_points=[]):
    """Collects quotient graph nodes from a point cloud.

    This function determines the quotient graph nodes that are most associated
    with the points within a given list of points using a majority rule. It
    iterates through the nodes in the quotient graph, evaluates point
    membership, and decides the inclusion of a quotient graph node into the
    result based on the majority vote.

    Parameters
    ----------
    quotient_graph : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph object containing a point cloud graph with defined
        nodes. Each node in the graph must include a 'quotient_graph_node'
        attribute.
    list_of_points : list, optional
        A list of points that are used to determine the majority association
        with the quotient graph's nodes. If not provided, it defaults to an
        empty list.

    Returns
    -------
    list
        A list of unique quotient graph nodes from the point cloud graph that
        have a majority of association with the provided list of points.
    """
    G = quotient_graph.point_cloud_graph
    listi = []

    for qgnode in quotient_graph.nodes:
        countyes = 0
        countno = 0
        list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qgnode]
        for n in list_of_nodes_each:
            if n in list_of_points:
                countyes += 1
            else:
                countno += 1
        if countyes > countno:
            listi.append(qgnode)

    list_of_clusters = list(set(listi))

    return list_of_clusters


def compute_quotient_graph_mean_attribute_from_points(G, QG, attribute='clustering_labels'):
    """Computes the mean of a specified attribute from nodes in the input graph `G` and assigns
    it to the corresponding nodes in the quotient graph `QG`.

    This function calculates the average value of a given attribute for all nodes in the
    original graph `G` that are represented by a single node in the quotient graph `QG`.
    The resulting mean is stored as a new attribute in the corresponding node in `QG`.

    Parameters
    ----------
    G : spectral_clustering.pointcloudgraph.PointCloudGraph
        The original graph containing the detailed node information including attributes
        relevant to the computation.
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph derived from the original graph. Each node in `QG` represents
        a group of nodes from `G`.
    attribute : str, optional
        The node attribute in `G` from which the mean is to be computed. The default value
        is 'clustering_labels'.

    Notes
    -----
    The function assumes that:
    1. Each node in `G` has the specified `attribute`.
    2. Each node in `G` has a secondary attribute 'quotient_graph_node' which specifies
       the corresponding node in `QG`.
    3. Groups of nodes in `G` are mapped to nodes in `QG` through the
       'quotient_graph_node' attribute.

    The resulting mean for each node in `QG` will be stored under a new attribute with
    the name defined as `{attribute}_mean`.
    """
    for n in QG.nodes:
        moy = 0
        list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        for e in list_of_nodes:
            moy += G.nodes[e][attribute]
        QG.nodes[n][attribute + '_mean'] = moy / len(list_of_nodes)


def merge_similar_class_QG_nodes(QG, attribute='viterbi_class', export=True):
    """Merges similar quotient graph (QG) nodes based on the specified attribute and updates the graph structure.

    This function iteratively merges nodes in the quotient graph (QG) that share the same value of the specified
    attribute. Each merge operation updates the QG and its associated point cloud graph, recalculating the
    'similarity_class' for affected edges until no edges exhibit similarity. Optionally, the function can export
    specific attributes to a file and display/export the modified graph.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph to be processed, which contains node attributes and maintains
        a reference to the associated point cloud graph.
    attribute : str, optional
        The node attribute used to evaluate similarity between nodes. By default, 'viterbi_class'.
    export : bool, optional
        If set to True, exports graph attributes and visualizations of the modified QG.
        Defaults to True.

    Returns
    -------
    spectral_clustering.quotientgraph.QuotientGraph
        The modified quotient graph (QG) with merged similar nodes and updated attributes.
    """
    G = QG.point_cloud_graph
    energy_similarity = 0
    for e in QG.edges():
        v1 = QG.nodes[e[0]][attribute]
        v2 = QG.nodes[e[1]][attribute]
        if v1 == v2:
            QG.edges[e]['similarity_class'] = 1
        else:
            QG.edges[e]['similarity_class'] = 0
        energy_similarity += QG.edges[e]['similarity_class']

    energy_per_edges = nx.get_edge_attributes(QG, 'similarity_class')
    edge_to_delete = max(energy_per_edges.items(), key=operator.itemgetter(1))[0]
    energy_max = energy_per_edges[edge_to_delete]
    while energy_max != 0:
        n1 = QG.nodes[edge_to_delete[0]]['intra_class_node_number']
        n2 = QG.nodes[edge_to_delete[1]]['intra_class_node_number']
        viterbi = QG.nodes[edge_to_delete[0]][attribute]
        list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == edge_to_delete[1]]
        for j in range(len(list_of_nodes)):
            G.nodes[list_of_nodes[j]]['quotient_graph_node'] = edge_to_delete[0]

        QG = nx.contracted_nodes(QG, edge_to_delete[0], edge_to_delete[1], self_loops=False)
        QG.point_cloud_graph = G
        QG.nodes[edge_to_delete[0]]['intra_class_node_number'] = n1 + n2
        QG.nodes[edge_to_delete[0]][attribute] = viterbi
        for e in QG.edges(edge_to_delete[0]):
            v1 = QG.nodes[e[0]][attribute]
            v2 = QG.nodes[e[1]][attribute]
            if v1 == v2:
                QG.edges[e]['similarity_class'] = 1
            else:
                QG.edges[e]['similarity_class'] = 0

        energy_per_edges = nx.get_edge_attributes(QG, 'similarity_class')
        edge_to_delete = max(energy_per_edges.items(), key=operator.itemgetter(1))[0]
        energy_max = energy_per_edges[edge_to_delete]
    # QG.rebuild(G=G)
    if export is True:
        export_some_graph_attributes_on_point_cloud(pcd_g=QG.point_cloud_graph,
                                                    graph_attribute='quotient_graph_node',
                                                    filename='Merge_after_viterbi.txt')
        display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20,
                                                     name="merge_quotient_graph_viterbi", data_on_nodes='viterbi_class',
                                                     data=True, attributekmeans4clusters=False)
    return QG


def merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[1]):
    """Merge nodes of a quotient graph (QG) based on a specified attribute and classes.

    This function processes a quotient graph and modifies it by merging nodes of the same
    specified class based on their attributes. It calculates similarities between nodes
    and iteratively merges those with a similarity class of 1, updating the properties
    of the resulting graph. The node merging process ensures that the point cloud graph
    associated with the quotient graph is also updated correctly.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph that contains nodes and edges to be processed. The graph must
        already include the attributes necessary for merging, such as intra-class node
        numbers and node classes.
    attribute : str, optional
        The node attribute of the quotient graph used to determine which nodes can
        be merged. Defaults to 'viterbi_class'.
    viterbiclass : list, optional
        A list of class values used for similarity calculations and determining
        whether nodes should be merged. Defaults to [1].

    Returns
    -------
    spectral_clustering.quotientgraph.QuotientGraph
        The updated quotient graph after merging nodes of the specified class.
    """
    G = QG.point_cloud_graph
    energy_similarity = 0
    for e in QG.edges():
        v1 = QG.nodes[e[0]][attribute]
        v2 = QG.nodes[e[1]][attribute]
        if v1 == v2 and v1 in viterbiclass:
            QG.edges[e]['similarity_class'] = 1
        else:
            QG.edges[e]['similarity_class'] = 0
        energy_similarity += QG.edges[e]['similarity_class']

    energy_per_edges = nx.get_edge_attributes(QG, 'similarity_class')
    edge_to_delete = max(energy_per_edges.items(), key=operator.itemgetter(1))[0]
    energy_max = energy_per_edges[edge_to_delete]
    while energy_max == 1:
        n1 = QG.nodes[edge_to_delete[0]]['intra_class_node_number']
        n2 = QG.nodes[edge_to_delete[1]]['intra_class_node_number']
        viterbi = QG.nodes[edge_to_delete[0]][attribute]
        list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == edge_to_delete[1]]
        for j in range(len(list_of_nodes)):
            G.nodes[list_of_nodes[j]]['quotient_graph_node'] = edge_to_delete[0]

        QG = nx.contracted_nodes(QG, edge_to_delete[0], edge_to_delete[1], self_loops=False)
        QG.point_cloud_graph = G
        QG.nodes[edge_to_delete[0]]['intra_class_node_number'] = n1 + n2
        QG.nodes[edge_to_delete[0]][attribute] = viterbi
        for e in QG.edges(edge_to_delete[0]):
            v1 = QG.nodes[e[0]][attribute]
            v2 = QG.nodes[e[1]][attribute]
            if v1 == v2 and v1 in viterbiclass:
                QG.edges[e]['similarity_class'] = 1
            else:
                QG.edges[e]['similarity_class'] = 0

        energy_per_edges = nx.get_edge_attributes(QG, 'similarity_class')
        edge_to_delete = max(energy_per_edges.items(), key=operator.itemgetter(1))[0]
        energy_max = energy_per_edges[edge_to_delete]
    # QG.rebuild(G=G)

    return QG


def quotient_graph_compute_direction_mean(quotient_graph):
    """Computes the mean direction gradient for each node in a quotient graph.

    This function iterates through nodes in the given `quotient_graph` and computes
    the mean of a specified property, `direction_gradient`, over corresponding
    subsets of nodes in the associated point cloud graph. The computed means are
    stored as a new property, `dir_gradient_mean`, in the nodes of the quotient graph.

    Parameters
    ----------
    quotient_graph : spectral_clustering.quotientgraph.QuotientGraph
        The input quotient graph, which is expected to contain a property,
        `point_cloud_graph`. Each node in the `quotient_graph` corresponds to
        a set of nodes in the associated `point_cloud_graph`, and the function
        computes and assigns a mean value for its `direction_gradient` property.
    """
    G = quotient_graph.point_cloud_graph
    for l in quotient_graph:
        quotient_graph.nodes[l]['dir_gradient_mean'] = 0
        list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l]
        for n in list_of_nodes_in_qnode:
            quotient_graph.nodes[l]['dir_gradient_mean'] += G.nodes[n]['direction_gradient']
        quotient_graph.nodes[l]['dir_gradient_mean'] /= len(list_of_nodes_in_qnode)


def quotient_graph_compute_direction_standard_deviation(quotient_graph, mean='dir_gradient_mean'):
    """Computes the standard deviation of direction gradients for a quotient graph node.

    This function calculates the standard deviation of the direction gradients for each
    quotient graph node. It processes each node in the quotient graph by iterating through
    its associated nodes in the original graph. For each associated node, a directional
    gradient is computed, and the mean and standard deviation of the gradients are updated
    accordingly in the quotient graph node attributes.

    Parameters
    ----------
    quotient_graph : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph object, which contains nodes and associated attributes. It
        includes a point cloud graph (point_cloud_graph) and information about the
        directional gradients of each node.
    mean : str, optional
        The attribute name in the quotient graph node describing the mean vector for
        the directional gradient. Defaults to 'dir_gradient_mean'.
    """
    G = quotient_graph.point_cloud_graph
    for l in quotient_graph:
        quotient_graph.nodes[l]['dir_gradient_angle_mean'] = 0
        quotient_graph.nodes[l]['dir_gradient_stdv'] = 0
        list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l]
        for n in list_of_nodes_in_qnode:
            v1 = quotient_graph.nodes[l][mean]
            v2 = G.nodes[n]['direction_gradient']
            G.nodes[n]['dir_gradient_angle'] = abs(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
            quotient_graph.nodes[l]['dir_gradient_angle_mean'] += G.nodes[n]['dir_gradient_angle']
        quotient_graph.nodes[l]['dir_gradient_angle_mean'] /= len(list_of_nodes_in_qnode)
        for n in list_of_nodes_in_qnode:
            v1 = quotient_graph.nodes[l][mean]
            v2 = G.nodes[n]['direction_gradient']
            quotient_graph.nodes[l]['dir_gradient_stdv'] += np.power(
                quotient_graph.nodes[l]['dir_gradient_angle_mean'] - G.nodes[n]['dir_gradient_angle'], 2)
        quotient_graph.nodes[l]['dir_gradient_stdv'] /= len(list_of_nodes_in_qnode)


def transfer_quotientgraph_infos_on_riemanian_graph(QG, info='viterbi_class'):
    """Transfers information from a quotient graph to a Riemannian graph based on a specified attribute.

    The function replicates a specified attribute from the nodes of the quotient graph to the nodes
    of the Riemannian graph. Each node in the Riemannian graph receives the value of the attribute
    from its corresponding node in the quotient graph. The correspondence is determined by a mapping
    stored in each node of the Riemannian graph.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph containing the original information that will be transferred.
    info : str, optional
        The attribute key in the quotient graph's nodes whose values are to be transferred to
        the nodes of the Riemannian graph. Default is 'viterbi_class'.

    Raises
    ------
    KeyError
        If the `quotient_graph_node` key is missing in any node's attributes in the
        Riemannian graph or if the specified `info` key is missing in the corresponding
        nodes in the quotient graph.

    Notes
    -----
    This function assumes correspondence between the nodes of the quotient graph and
    the nodes of the Riemannian graph is established via the 'quotient_graph_node'
    key in each Riemannian graph node.
    """
    G = QG.point_cloud_graph
    for n in G.nodes:
        G.nodes[n][info] = QG.nodes[G.nodes[n]['quotient_graph_node']][info]


def calculate_leaf_quotients(QG, list_leaves, list_of_linear, root_point_riemanian):
    """Calculates leaf quotients for nodes and edges in a graph, updating specific attributes for each.

    The function processes a quotient graph `QG` and updates node and edge attributes based on the paths
    between leaf nodes in `list_leaves` and a designated `root_point_riemanian`. For each leaf, the function
    determines the shortest path (via Dijkstra's algorithm) from the leaf to the root, either through specific
    nodes in `list_of_linear` or the entire graph.
    It updates the 'leaf_quotient_n' and 'leaf_quotient_e' attributes of nodes and edges respectively along the path,
    storing how many times each is involved in such paths.
    The function also manages the 'useful_path' attribute for edges.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph represented as a NetworkX graph object. It contains nodes and edges with
        specific attributes that are updated during the computation.
    list_leaves : list
        A list of leaf nodes from which paths to the root node will be calculated.
    list_of_linear : list
        A list of intermediate nodes that may be included in the subgraph for calculating paths.
    root_point_riemanian : Any
        The root point in the Riemannian space, used to identify the destination node within the quotient
        graph.

    """
    G = QG.point_cloud_graph
    for e in QG.edges:
        QG.edges[e]['leaf_quotient_e'] = 0
        QG.edges[e]['useful_path'] = 50
    for n in QG.nodes:
        QG.nodes[n]['leaf_quotient_n'] = 0

    for leaf in list_leaves:
        print(leaf)
        sub_qg = nx.subgraph(QG, list_of_linear + [leaf] + [G.nodes[root_point_riemanian]["quotient_graph_node"]])
        if nx.has_path(sub_qg, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"]):
            path = nx.dijkstra_path(sub_qg, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"],
                                    weight='distance_centroides')
        else:
            print('else')
            path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"],
                                    weight='distance_centroides')
            print(path)
        for n in path:
            QG.nodes[n]['leaf_quotient_n'] += 1
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            QG.edges[e]['leaf_quotient_e'] += 1
            QG.edges[e]['useful_path'] = 0
