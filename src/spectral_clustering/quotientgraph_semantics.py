#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy as copy
import operator

import networkx as nx
import numpy as np
import scipy as sp

from spectral_clustering.dijkstra_segmentation import initdijkstra
from spectral_clustering.display_and_export import display_and_export_quotient_graph_matplotlib
from spectral_clustering.display_and_export import export_quotient_graph_attribute_on_point_cloud
from spectral_clustering.quotientgraph_operations import calculate_leaf_quotients
from spectral_clustering.quotientgraph_operations import compute_quotient_graph_mean_attribute_from_points
from spectral_clustering.quotientgraph_operations import quotient_graph_compute_direction_mean
from spectral_clustering.quotientgraph_operations import quotient_graph_compute_direction_standard_deviation
from spectral_clustering.quotientgraph_operations import transfer_quotientgraph_infos_on_riemanian_graph
from spectral_clustering.split_and_merge import create_subgraphs_to_work
from spectral_clustering.split_and_merge import resegment_nodes_with_elbow_method


def define_leaves_by_topo(quotientgraph):
    """
    Defines the leaves of a quotient graph based on topological structure and selects
    a root node based on specific node attributes.

    This function identifies "end" nodes (leaves) of the graph and computes a
    mean gradient value for each leaf node based on the associated nodes' gradient
    attributes from the base graph. The root node is randomly selected from the
    leaves, although it could optionally be chosen as the node with the highest
    mean gradient.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph structure representing a simplified version of the
        point cloud graph. Nodes in this graph contain attributes, and the graph
        is connected to the underlying detailed structure of the point cloud
        graph.
    """
    G = quotientgraph.point_cloud_graph
    # Put the "end" nodes in a list
    list_leaves = [x for x in quotientgraph.nodes() if quotientgraph.degree(x) == 1]

    # Pick the node with the most average norm gradient (most likely to be part of the stem)
    max_norm = 0
    max_norm_node = -1
    for l in list_leaves:
        quotientgraph.nodes[l]['norm_gradient_mean'] = 0
        list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l]
        for n in list_of_nodes_in_qnode:
            quotientgraph.nodes[l]['norm_gradient_mean'] += G.nodes[n]['norm_gradient']
        quotientgraph.nodes[l]['norm_gradient_mean'] /= len(list_of_nodes_in_qnode)
        if quotientgraph.nodes[l]['norm_gradient_mean'] > max_norm:
            max_norm = quotientgraph.nodes[l]['norm_gradient_mean']
            max_norm_node = l

    # Finally I chose randomly the root among the leaves but I could choose the max_norm_node
    root = np.random.choice(list_leaves)
    # cluster again the branches by following a path
    # to_cluster = []
    # for n in self:
    #    if n not in list_leaves:
    #       to_cluster.append(n)

    # self.segment_several_nodes_using_attribute(G=self.point_cloud_graph, list_quotient_node_to_work=to_cluster,
    #                                          algo='OPTICS', para_clustering_meth=100, attribute='direction_gradient')
    # self.delete_small_clusters(min_number_of_element_in_a_quotient_node=100)


def define_semantic_classes(quotientgraph):
    """Assigns semantic labels to nodes in a quotient graph.

    Given a quotient graph, this function classifies its nodes into three semantic classes:
    'leaf', 'stem', and 'petiole'.
    Nodes with a single adjacency are labeled as 'leaf'.
    The node with the highest number of neighbors is labeled as 'stem'.
    All other nodes are classified as 'petiole'.
    The function directly modifies the quotient graph by adding a 'semantic_label' attribute with the
    appropriate class to each node.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        A quotient graph where each node represents an abstract entity, and its
        edges define connections between these entities.
    """
    # leaves : one adjacency
    list_leaves = [x for x in quotientgraph.nodes() if quotientgraph.degree(x) == 1]

    # the node with the most number of neighborhood is the stem
    degree_sorted = sorted(quotientgraph.degree, key=lambda x: x[1], reverse=True)
    stem = [degree_sorted[0][0]]

    # the rest is defined as branches
    for n in quotientgraph.nodes:
        if n in list_leaves:
            quotientgraph.nodes[n]['semantic_label'] = 'leaf'
        if n in stem:
            quotientgraph.nodes[n]['semantic_label'] = 'stem'
        elif n not in list_leaves and n not in stem:
            quotientgraph.nodes[n]['semantic_label'] = 'petiole'


def define_semantic_scores(quotientgraph, method='similarity_dist'):
    """Defines semantic scores for nodes in the quotient graph based on specific methods.

    This function computes the semantic scores for each node in the quotient graph
    based on either a condition-based logic or a similarity distance approach. The
    method determines how the scores are calculated and applied to the nodes. Scores
    are computed for three categories: `leaf`, `petiole`, and `stem`. The method
    'includes intermediate computation and smoothstep/identity transformations for
    distances or direct logical conditions.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        A quotient graph object which contains nodes and related descriptors
        such as planarity, linearity, silhouette, and degree. The graph structure
        and attributes are utilized to compute semantic scores for each node.
    method : str, optional
        The scoring method to be used. Defaults to 'similarity_dist'.
        Options include:

          - 'condition_list': Evaluates descriptive conditions on graph node properties.
          - 'similarity_dist': Computes similarity based on a reference score vector
            using Euclidean distances and applies smoothstep or identity transformations.
    """
    quotientgraph.compute_local_descriptors(quotientgraph.point_cloud_graph, method='all_qg_cluster', data='coords')
    quotientgraph.compute_silhouette()
    degree_sorted = sorted(quotientgraph.degree, key=lambda x: x[1], reverse=True)
    stem = [degree_sorted[0][0]]

    if method == 'condition_list':
        for n in quotientgraph.nodes:
            score_vector_leaf = [quotientgraph.nodes[n]['planarity'] > 0.5, quotientgraph.nodes[n]['linearity'] < 0.5,
                                 quotientgraph.degree(n) == 1]
            score_vector_petiole = [quotientgraph.nodes[n]['planarity'] < 0.5,
                                    quotientgraph.nodes[n]['linearity'] > 0.5, quotientgraph.degree(n) == 2,
                                    quotientgraph.nodes[n]['silhouette'] > 0.2]
            score_vector_stem = [quotientgraph.nodes[n]['planarity'] < 0.5, quotientgraph.nodes[n]['linearity'] > 0.5,
                                 quotientgraph.degree(n) == degree_sorted[0][1]]
            quotientgraph.nodes[n]['score_leaf'] = score_vector_leaf.count(True) / len(score_vector_leaf)
            quotientgraph.nodes[n]['score_petiole'] = score_vector_petiole.count(True) / len(score_vector_petiole)
            quotientgraph.nodes[n]['score_stem'] = score_vector_stem.count(True) / len(score_vector_stem)

    if method == 'similarity_dist':
        def smoothstep(x):
            if x <= 0:
                res = 0
            elif x >= 1:
                res = 1
            else:
                res = 3 * pow(x, 2) - 2 * pow(x, 3)

            return res

        def identity(x):
            if x <= 0:
                res = 0
            elif x >= 1:
                res = 1
            else:
                res = x

            return res

        score_vector_leaf_ref = [0.81, 0, -0.1]
        score_vector_linea_ref = [0, 1, 0.3]
        for n in quotientgraph.nodes:
            # score_vector_n = sk.preprocessing.normalize(np.asarray([QG.nodes[n]['planarity'], QG.nodes[n]['linearity']]).reshape(1,len(score_vector_leaf_ref)), norm='l2')
            score_vector_n = [quotientgraph.nodes[n]['planarity'], quotientgraph.nodes[n]['linearity'],
                              quotientgraph.nodes[n]['silhouette']]
            print(score_vector_n)
            d1_leaf = sp.spatial.distance.euclidean(score_vector_leaf_ref, score_vector_n)
            # print(d1_leaf)
            d2_linea = sp.spatial.distance.euclidean(score_vector_linea_ref, score_vector_n)
            # print(d2_leaf)
            f1 = 1 - d1_leaf
            f2 = 1 - d2_linea
            quotientgraph.nodes[n]['score_leaf'] = identity(f1)
            quotientgraph.nodes[n]['score_petiole'] = identity(f2)


def minimum_spanning_tree_quotientgraph_semantics(quotientgraph):
    """
    Compute the minimum spanning tree (MST) of a quotient graph with semantic-based
    weight adjustments.

    This function processes a quotient graph by assigning semantic weights to its
    edges based on the semantic labels of the connected nodes. Edges connecting
    nodes with the same semantic label are given a higher weight than those
    connecting nodes with different semantic labels. Subsequently, the function
    computes the MST of the adjusted quotient graph using Kruskal's algorithm.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph on which the MST and weight adjustments should be
        computed. The graph must have nodes with a 'semantic_label' property and
        edges that need a 'semantic_weight' property. Additionally, it must
        incorporate attributes for the point cloud graph used in computation.

    Returns
    -------
    QG_t2 : networkx.Graph
        The resultant minimum spanning tree of the quotient graph, where weights
        are determined by semantic-based criteria.
    """
    define_semantic_classes(quotientgraph)
    for (u, v) in quotientgraph.edges:
        if quotientgraph.nodes[u]['semantic_label'] == quotientgraph.nodes[v]['semantic_label']:
            quotientgraph.edges[u, v]['semantic_weight'] = 50
        else:
            quotientgraph.edges[u, v]['semantic_weight'] = 1
    compute_quotient_graph_mean_attribute_from_points(quotientgraph.point_cloud_graph, quotientgraph,
                                                      attribute='quotient_graph_node')
    QG_t2 = nx.minimum_spanning_tree(quotientgraph, algorithm='kruskal', weight='semantic_weight')

    return QG_t2


def determination_main_stem(QG, list_of_linear_QG_nodes, stemroot, list_leaves, angle_to_stop=45,
                            new_viterbi_class_number=3):
    """
    Determines the main stem of a Quotient Graph (QG) starting from a stem root, facilitated
    by directional and angular constraints. Updates the viterbi classification for the identified
    stem nodes in both the QG and the associated point cloud graph.

    The function iteratively identifies the stem path by minimizing energy values (dot products
    of edge vectors). The traversal stops either when the energy threshold (`angle_to_stop`) is
    exceeded or if the next node is not part of the predefined QG linear nodes. Additionally,
    the function assigns the provided viterbi class number to all nodes in the identified stem.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The input Quotient Graph where directional and viterbi classifications are computed.
    list_of_linear_QG_nodes : list
        A list of node identifiers in the Quotient Graph representing the linear structural
        nodes of interest.
    stemroot : int or str
        Identifier of the starting node for the stem determination process in the Quotient Graph.
    list_leaves : list
        List of leaf node identifiers in the Quotient Graph used for directional computation.
    angle_to_stop : float, optional
        Angular threshold (in degrees) to determine when to stop the stem traversal.
        Defaults to ``45`` degrees.
    new_viterbi_class_number : int, optional
        The viterbi classification number assigned to identified stem nodes in both the Quotient
        Graph and the point cloud graph. Defaults to ``3``.
    """
    QG.compute_direction_info(list_leaves=list_leaves)
    G = QG.point_cloud_graph
    energy_to_stop = 1 - np.cos(np.radians(angle_to_stop))
    keepgoing = True
    node_stem = stemroot
    list_stem = []
    list_stem.append(node_stem)
    while keepgoing:
        list_energies = []
        list_nodes = []
        print(node_stem)
        for n in QG[node_stem]:
            if n not in list_stem:
                list_energies.append(QG.edges[(n, node_stem)]['energy_dot_product'])
                list_nodes.append(n)
        node = list_energies.index(min(list_energies))
        node_stem = list_nodes[node]
        if list_nodes:
            if min(list_energies) > energy_to_stop or node_stem not in list_of_linear_QG_nodes:
                keepgoing = False
            else:
                list_stem.append(node_stem)
        else:
            keepgoing is False

    for n in list_stem:
        QG.nodes[n]['viterbi_class'] = new_viterbi_class_number
        list_gnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        for i in list_gnode:
            G.nodes[i]['viterbi_class'] = new_viterbi_class_number


def determination_main_stem_shortest_paths(QG, list_of_linear_QG_nodes):
    """Determines the shortest paths for the main stem using the quotient graph (QG).

    This function identifies and processes key nodes in the quotient graph (QG) and
    marks their corresponding nodes in the point cloud graph with a specific class.
    It uses the Sub-Riemannian graph derived from the QG to compute the traversed
    clusters and identifies the main stem's shortest paths within the graph structure.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph representing the higher-level graph structure, which includes
        nodes and edges used for the analysis.
    list_of_linear_QG_nodes : list
        A list of nodes from the quotient graph that defines the linear paths to focus on.

    Returns
    -------
    list
        A deduplicated list of quotient graph nodes traversed by the shortest paths of the main stem.
    """
    sub_riemanian = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=list_of_linear_QG_nodes)
    G = QG.point_cloud_graph
    segmsource, ptsource, ptarrivee = initdijkstra(sub_riemanian)
    list_clusters_qg_traversed = []
    for i in segmsource:
        n = G.nodes[i]['quotient_graph_node']
        list_clusters_qg_traversed.append(n)
    final_list_stem = list(set(list_clusters_qg_traversed))
    for n in final_list_stem:
        QG.nodes[n]['viterbi_class'] = 3
        list_gnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        for i in list_gnode:
            G.nodes[i]['viterbi_class'] = 3
    return final_list_stem


def determination_main_stem_shortest_paths_improved(QG, ptsource, list_of_linear_QG_nodes, angle_to_stop=45,
                                                    minimumpoint=5, classnumberstem=3, classnumberanomaly=3):
    """
    Determines the main 'stem' paths on a quotient graph using an improved algorithm,
    identifying clusters and marking anomalies based on vector directions and energy
    thresholds.

    The function processes a quotient graph and a list of linear quotient graph nodes
    to identify the main linear stems. It applies directional filtering based on the
    mean and variance of directional information stored in the graph vertices. The
    stems are further refined using angle-based thresholds, and anomalies are noted
    and classified accordingly. Additionally, this function transfers classifications
    from the quotient graph to the Riemannian graph.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph (QG) that contains the structure and properties used to
        compute the main stems and anomaly points.
    ptsource : int
        A starting point or source node for the shortest path calculation on the sub-Riemannian graph.
    list_of_linear_QG_nodes : list[int]
        A list of nodes in the quotient graph that are considered linear for the purposes of this computation.
    angle_to_stop : float, optional
        The angle threshold to stop splitting paths, in degrees. Default is ``45``.
    minimumpoint : int, optional
        Minimum number of traversals through a cluster node for it to be considered part of the main stem.
        Default is ``5``.
    classnumberstem : int, optional
        The classification number assigned to the nodes belonging to the main stem.
        Default is ``3``.
    classnumberanomaly : int, optional
        The classification number assigned to nodes considered anomalies.
        Default is ``3``.

    Returns
    -------
    list[int]
        A list of quotient graph nodes identified as the refined main stem,
        including the source point and the final cleaned list.
    """
    import copy
    sub_riemanian = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=list_of_linear_QG_nodes)
    G = QG.point_cloud_graph
    energy_to_stop = 1 - np.cos(np.radians(angle_to_stop))

    dict = nx.single_source_dijkstra_path_length(sub_riemanian, ptsource, weight='weight')
    ptarrivee = max(dict.items(), key=operator.itemgetter(1))[0]
    segmsource = nx.dijkstra_path(sub_riemanian, ptsource, ptarrivee, weight='weight')
    list_clusters_qg_traversed = []
    for i in segmsource:
        n = G.nodes[i]['quotient_graph_node']
        list_clusters_qg_traversed.append(n)

    def count_to_dict(lst):
        return {k: lst.count(k) for k in lst}

    dict = count_to_dict(list_clusters_qg_traversed)
    list_stem2 = []
    for key, value in dict.items():
        if value > minimumpoint:
            list_stem2.append(key)
    list_stem1 = list(dict.fromkeys(list_clusters_qg_traversed))

    quotient_graph_compute_direction_mean(QG)
    quotient_graph_compute_direction_standard_deviation(QG)
    final_list_stem = copy.deepcopy(list_stem2)
    for node in list_stem2:
        if QG.nodes[node]['dir_gradient_angle_mean'] < np.cos(np.radians(angle_to_stop)):
            final_list_stem.remove(node)
            QG.nodes[node]['viterbi_class'] = classnumberanomaly

    nodi = 0
    final_list_stem_clean = []
    while nodi <= (len(final_list_stem) - 2):
        e = (final_list_stem[nodi], final_list_stem[nodi + 1])
        v1 = QG.nodes[e[0]]['dir_gradient_mean']
        v2 = QG.nodes[e[1]]['dir_gradient_mean']
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        energy = 1 - dot
        print(e)
        print(energy)
        if energy > energy_to_stop:
            final_list_stem_clean = final_list_stem[:nodi + 1]
            nodi = len(final_list_stem) + 2
        else:
            final_list_stem_clean = final_list_stem
            nodi += 1

    final_list_stem_clean.insert(0, G.nodes[ptsource]['quotient_graph_node'])
    for n in final_list_stem_clean:
        QG.nodes[n]['viterbi_class'] = classnumberstem

    transfer_quotientgraph_infos_on_riemanian_graph(QG=QG, info='viterbi_class')

    return final_list_stem_clean


def stem_detection_with_quotite_leaves(QG, list_leaves3, list_apex, list_of_linear, root_point_riemanian,
                                       new_class_stem=3):
    """
    Detects and classifies the main stem of a graph based on leaf quotients and other metrics.

    This function identifies the main stem of an input graph using attributes like
    leaf quotients, apex nodes, and linear node structures. The identified stem
    is classified and its attributes are exported for further use. The function
    works by analyzing subgraphs, calculating paths, and determining connections
    between nodes to identify the main stem.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        A quotient graph representation of the point cloud graph.
    list_leaves3 : list
        A list of end-leaf nodes in the quotient graph.
    list_apex : list
        A list of apex nodes to consider. If empty, list_leaves3 is used by default.
    list_of_linear : list
        A list of nodes representing a linear path or structure in the graph.
    root_point_riemanian : int
        Node id corresponding to the root point in the Riemannian graph representation.
    new_class_stem : int, optional
        An integer value used to classify the main stem nodes after detection.
        Default is ``3``.

    Returns
    -------
    list
        A list of nodes forming the identified and classified main stem of the graph.

    Notes
    -----
    - The function depends heavily on networkx for graph operations such as
      subgraph creation, path calculations, and attribute manipulations.
    - Subgraphs are dynamically created to analyze and validate paths.
    - Detected stem nodes are marked with the provided classification value in
      the quotient graph `viterbi_class` attribute.
    - The exported attributes can be used for further semantic analysis or visualization.
    """
    calculate_leaf_quotients(QG, list_leaves3, list_of_linear, root_point_riemanian)
    G = QG.point_cloud_graph
    list_length = []
    if list_apex:
        list_end = list_apex
    else:
        list_end = list_leaves3

    for leaf in list_end:
        sub_qg = nx.subgraph(QG, list_of_linear + [leaf] + [G.nodes[root_point_riemanian]["quotient_graph_node"]])
        display_and_export_quotient_graph_matplotlib(qg=sub_qg, node_sizes=20,
                                                     name="sub_graphsclass_feuilles_sur_noeuds" + str(leaf),
                                                     data_on_nodes='viterbi_class', data=True,
                                                     attributekmeans4clusters=False)

        if nx.has_path(sub_qg, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"]):
            path = nx.dijkstra_path(sub_qg, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"],
                                    weight='distance_centroides')
        else:
            path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"],
                                    weight='distance_centroides')
        pathleafquantity = []
        paths = []
        for p in path:
            pathleafquantity.append(QG.nodes[p]['leaf_quotient_n'])
            paths.append(path)
        print(pathleafquantity)
        print(set(pathleafquantity))
        lengthpath = sum(list(set(pathleafquantity)))
        list_length.append(lengthpath)
        print(path)
        print(lengthpath)

    leafend = list_end[list_length.index(max(list_length))]
    # path_final = paths[]

    list_stem_full = nx.dijkstra_path(QG, G.nodes[root_point_riemanian]["quotient_graph_node"], leafend,
                                      weight='distance_centroides')
    list_stem = list_stem_full
    print('list_stem')
    print(list_stem)
    previous = 0
    """
    maxim = QG.nodes[G.nodes[root_point_riemanian]["quotient_graph_node"]]['leaf_quotient_n']
    for n in list_stem_full:
        if maxim >= QG.nodes[n]['leaf_quotient_n']:
            maxim = QG.nodes[n]['leaf_quotient_n']
            prev = n
        else:
            list_stem.pop(prev)
    """
    del list_stem[-1]
    for n in list_stem:
        # if n not in list_leaves3:
        QG.nodes[n]["viterbi_class"] = new_class_stem

    export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_')
    return list_stem


def differenciate_apex_limb(QG, attribute_class='viterbi_class', number_leaves_limb=1, new_apex_class=4):
    """Differentiate the apex limb based on the given attribute class and number of leaves.

    This function updates the provided graph by reassigning the specified attribute
    class for nodes that meet specific conditions. It identifies nodes (limbs) in the
    graph based on the attribute value and further modifies their attribute class if
    certain criteria regarding local extremum of the Fiedler vector are satisfied.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        A graph-like object with nodes carrying specific attributes. Typically,
        this should be a custom graph object that includes attributes such as
        `number_of_local_Fiedler_extremum` in its nodes.
    attribute_class : str, optional
        The attribute class in the graph's node dictionary to check and update.
        Default is ``'viterbi_class'``.
    number_leaves_limb : int, optional
        The value of `attribute_class` used to identify the nodes of interest
        (limbs). Only nodes where the specific attribute class has this value
        will be processed. Default is ``1``.
    new_apex_class : int, optional
        The new value for the attribute class to assign to the nodes that satisfy
        the conditions. Default is ``4``.
    """
    QG.count_local_extremum_of_Fiedler()
    list_limbs = [x for x, y in QG.nodes(data=True) if y[attribute_class] == number_leaves_limb]
    for l in list_limbs:
        if QG.nodes[l]['number_of_local_Fiedler_extremum'] > 1:
            QG.nodes[l][attribute_class] = new_apex_class


def topology_control(quotient_graph, attribute_class_control='viterbi_class', class_stem=3, class_petiols=5,
                     class_limb=1, class_apex=4, error_norm=10, error_dir=11):
    """Adjusts node attributes in a quotient graph based on the class control rules.

    This function modifies the node attributes in the provided quotient graph
    (QG) according to specific rules associated with node classifications such as
    stems, petioles, limbs, and apex. Adjustments are made to enforce corrections
    for nodes incorrectly classified with high degrees or inappropriate connections.
    The corrections are applied by reassigning specific error classification values.

    Parameters
    ----------
    quotient_graph : spectral_clustering.quotientgraph.QuotientGraph
        A NetworkX graph where each node has attributes, including the
        classification attribute used for controlling topology.
    attribute_class_control : str, optional
        Name of the node attribute that represents the class to control.
        The default is ``'viterbi_class'``.
    class_stem : int, optional
        Numerical identifier representing the class of stem nodes.
        The default is ``3``.
    class_petiols : int, optional
        Numerical identifier representing the class of petiole nodes.
        The default is ``5``.
    class_limb : int, optional
        Numerical identifier representing the class of limb nodes.
        The default is ``1``.
    class_apex : int, optional
        Numerical identifier representing the class of apex nodes.
        The default is ``4``.
    error_norm : int, optional
        Numerical identifier used to reclassify nodes that violate the
        connection rules for limb or apex nodes. The default is ``10``.
    error_dir : int, optional
        Numerical identifier used to reclassify nodes with high degree
        that violate the class petioles rules. The default is ``11``.
    """
    QG = quotient_graph
    # check leaves connected to stem
    stem = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == class_stem]
    for n in QG[stem[0]]:
        if QG.nodes[n][attribute_class_control] == class_limb or QG.nodes[n][attribute_class_control] == class_apex:
            QG.nodes[n][attribute_class_control] = error_norm

    petiols = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == class_petiols]
    for n in petiols:
        if QG.degree(n) >= 3:
            QG.nodes[n][attribute_class_control] = error_dir


def treat_topology_error(quotient_graph, attribute_class_control='viterbi_class', error=5, way_to_treat='direction',
                         number_of_cluster_tested=20):
    """
    Handles errors in the topological attributes of a quotient graph by identifying problematic nodes
    and resegmenting them using a clustering approach. This method applies an elbow method for clustering
    to ensure appropriate grouping of nodes based on specified attributes.

    Parameters
    ----------
    quotient_graph : spectral_clustering.quotientgraph.QuotientGraph
        The input quotient graph containing nodes with labeled attributes to analyze and correct.
    attribute_class_control : str, optional
        The attribute in quotient_graph's nodes that is examined to identify nodes with errors.
        Default is ``'viterbi_class'``.
    error : int, optional
        The specific error value in the attribute_class_control being targeted for treatment.
        Default is ``5``.
    way_to_treat : {'direction', 'norm'}, optional, default='direction'
        Specifies the method to use for segmenting and treating erroneous nodes.
        If ``'direction'``, the 'direction_gradient' attribute is used for resegmentation.
        If ``'norm'``, the 'norm_gradient' attribute is used.
    number_of_cluster_tested : int, optional
        The maximum number of clusters to test when applying the elbow method for clustering the nodes.
        Default is ``20``.

    Notes
    -----
    The method makes use of the resegment_nodes_with_elbow_method, which performs clustering on nodes
    based on their attributes. The attributes ‘direction_gradient’ or ‘norm_gradient’ are utilized
    depending on the `way_to_treat` parameter to guide the resegmentation process.
    """
    QG = quotient_graph
    nodes_to_treat = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == error]

    if way_to_treat == 'direction':
        resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=nodes_to_treat,
                                          number_of_cluster_tested=number_of_cluster_tested,
                                          attribute='direction_gradient', number_attribute=3, standardization=False)
    elif way_to_treat == 'norm':
        resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=nodes_to_treat,
                                          number_of_cluster_tested=number_of_cluster_tested,
                                          attribute='norm_gradient', number_attribute=1, standardization=False,
                                          numer=1000)


def weight_with_semantic(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_linear=0,
                         class_apex=4,
                         weight_limb_limb=10000, weight_apex_apex=10000, weight_apex_mainstem=10000,
                         weight_limb_apex=10000, weight_limb_mainstem=10000, weight_linear_mainstem=0,
                         weight_linear_linear=0,
                         weight_linear_limb=0, weight_linear_apex=0):
    """Assigns semantic weights to the edges of a graph based on node classifications.

    This function iterates over all edges in the provided graph `QG` and assigns a weight
    to each edge depending on the classification of its connected nodes. The classification
    of the nodes is determined using the `class_attribute` parameter. Different combinations
    or sets of node categories (e.g., `class_limb`, `class_apex`) have corresponding weights
    that are specified as input arguments to the function.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The graph whose edges will be assigned weights based on the semantic relationships
        between connected nodes.
    class_attribute : str, optional
        The node attribute in `QG` used to determine the category of the nodes.
        Default is ``'viterbi_class'``.
    class_limb : Any, optional
        A value representing the "limb" classification of a node.
        Default is ``1``.
    class_mainstem : Any, optional
        A value representing the "mainstem" classification of a node.
        Default is ``3``.
    class_linear : Any, optional
        A value representing the "linear" classification of a node.
        Default is ``0``.
    class_apex : Any, optional
        A value representing the "apex" classification of a node.
        Default is ``4``.
    weight_limb_limb : float, optional
        The weight assigned to edges where both connected nodes are classified as "limb".
        Default is ``10000``.
    weight_apex_apex : float, optional
        The weight assigned to edges where both connected nodes are classified as "apex".
        Default is ``10000``.
    weight_apex_mainstem : float, optional
        The weight assigned to edges where one node is classified as "apex" and the other as "mainstem".
        Default is ``10000``.
    weight_limb_apex : float, optional
        The weight assigned to edges where one node is classified as "limb" and the other as "apex".
        Default is ``10000``.
    weight_limb_mainstem : float, optional
        The weight assigned to edges where one node is classified as "limb" and the other as "mainstem".
        Default is ``10000``.
    weight_linear_mainstem : float, optional
        The weight assigned to edges where one node is classified as "linear" and the other as "mainstem".
        Default is ``0``.
    weight_linear_linear : float, optional
        The weight assigned to edges where both connected nodes are classified as "linear".
        Default is ``0``.
    weight_linear_limb : float, optional
        The weight assigned to edges where one node is classified as "linear" and the other as "limb".
        Default is ``0``.
    weight_linear_apex : float, optional
        The weight assigned to edges where one node is classified as "linear" and the other as "apex".
        Default is ``0``.

    Notes
    -----
    - The function modifies the provided graph `QG` in place. It adds a new edge attribute
      named `weight_sem_paths` to store the computed weights.
    - If node classifications for a given edge do not match any of the specified categories
    """
    for e in QG.edges():
        QG.edges[e]['weight_sem_paths'] = np.nan
        c1 = QG.nodes[e[0]][class_attribute]
        c2 = QG.nodes[e[1]][class_attribute]
        l = [c1, c2]
        if set(l) == {class_limb, class_limb}:
            QG.edges[e]['weight_sem_paths'] = weight_limb_limb
        if set(l) == {class_limb, class_mainstem}:
            QG.edges[e]['weight_sem_paths'] = weight_limb_mainstem
        if set(l) == {class_limb, class_apex}:
            QG.edges[e]['weight_sem_paths'] = weight_limb_apex
        if set(l) == {class_limb, class_linear}:
            QG.edges[e]['weight_sem_paths'] = weight_linear_limb

        if set(l) == {class_apex, class_apex}:
            QG.edges[e]['weight_sem_paths'] = weight_apex_apex
        if set(l) == {class_apex, class_mainstem}:
            QG.edges[e]['weight_sem_paths'] = weight_apex_mainstem
        if set(l) == {class_apex, class_linear}:
            QG.edges[e]['weight_sem_paths'] = weight_linear_apex

        if set(l) == {class_linear, class_linear}:
            QG.edges[e]['weight_sem_paths'] = weight_linear_linear
        if set(l) == {class_linear, class_mainstem}:
            QG.edges[e]['weight_sem_paths'] = weight_linear_mainstem


def shortest_paths_from_limbs_det_petiol(QG, root_point_riemanian, class_attribute='viterbi_class',
                                         weight='weight_sem_paths', class_limb=1, class_linear=0, class_mainstem=3,
                                         new_class_petiol=5):
    """
    Computes shortest paths from limb nodes to the root point in the Riemannian tree graph and updates
    node and edge attributes accordingly.

    This function identifies limb nodes in the graph based on their classification attribute and calculates
    the shortest path from these limb nodes to a specified root point in the Riemannian tree graph.
    During the path computation, nodes classified as linear are reclassified to a new specified petiol class.
    Additionally, edges along these paths are marked as part of a useful shortest path.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The input graph representing the point cloud or Riemannian tree structure.
    root_point_riemanian : Any
        The key of the root node in the graph from which shortest paths are computed.
    class_attribute : str, optional
        The node attribute in the graph used for classification.
        Default is ``'viterbi_class'``.
    weight : str, optional
        The edge attribute used to calculate shortest paths' weights
        Default is ``'weight_sem_paths'``.
    class_limb : int, optional
        The classification value of nodes considered as limb nodes
        Default is ``1``.
    class_linear : int, optional
        The classification value of nodes considered linear in the graph
        Default is ``0``.
    class_mainstem : int, optional
        The classification value of nodes considered as the main stem in the graph
        Default is ``3``.
    new_class_petiol : int, optional
        The classification value to which linear nodes are reclassified during path traversal
        Default is ``5``.

    Notes
    -----
    The function updates the graph `QG` in place by modifying node and edge attributes.
    """
    G = QG.point_cloud_graph

    list_limb = [x for x, y in QG.nodes(data=True) if y[class_attribute] == class_limb]
    for leaf in list_limb:
        path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"], weight=weight)
        for n in path:
            if QG.nodes[n][class_attribute] == class_linear:
                QG.nodes[n][class_attribute] = new_class_petiol
        for i in range(len(path) - 1):
            if QG.has_edge(path[i], path[i + 1]):
                e = (path[i], path[i + 1])
            else:
                e = (path[i + 1], path[i])
            QG.edges[e]['useful_path_shortest'] = True


def maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5,
                         class_apex=4, class_branch=6,
                         weight_petiol_petiol=0, weight_petiol_apex=10000, weight_branch_limb=10000):
    """Adjusts edge weights in a graph based on semantic class relationships.

    This function considers semantic relationships between different graph node
    classes and modifies the edge weight accordingly. It specifically targets
    edges connecting nodes with defined class combinations, and assigns a new
    weight to them based on the given parameters.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The input graph where the semantic weights of edges will be modified.
        Nodes of the graph should include the attribute specified in
        `class_attribute` to determine their semantic class.
    class_attribute : str
        The name of the node attribute that represents the semantic class of the node.
        Default is ``'viterbi_class'``.
    class_limb : int, optional
        Represents the integer code for the 'limb' semantic class.
        Default: `1`
    class_mainstem : int, optional
        Represents the integer code for the 'mainstem' semantic class.
        Default: `3`
    class_petiol : int, optional
        Represents the integer code for the 'petiol' semantic class.
        Default: `5`
    class_apex : int, optional
        Represents the integer code for the 'apex' semantic class.
        Default: `4`
    class_branch : int, optional
        Represents the integer code for the 'branch' semantic class.
        Default: `6`
    weight_petiol_petiol : int, optional
        The weight representing edges between two 'petiol' class nodes.
        Default: `0`.
    weight_petiol_apex : int, optional
        The weight representing edges connecting a 'petiol' class node to an 'apex' class node.
        Default: `10000`.
    weight_branch_limb : int, optional
        The weight representing edges connecting a 'branch' class node to a 'limb' class node.
        Default: `10000`.

    Notes
    -------
    This function modifies the input graph in-place by adding or updating the 'weight_sem_paths' attribute
    on edges based on their connected nodes' semantic classes.
    """
    for e in QG.edges():
        c1 = QG.nodes[e[0]][class_attribute]
        c2 = QG.nodes[e[1]][class_attribute]
        l = [c1, c2]
        if set(l) == {class_branch, class_limb}:
            QG.edges[e]['weight_sem_paths'] = weight_branch_limb


def shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class',
                                        weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3,
                                        new_class_branch=6):
    """Shortest paths calculation and attribute mutation for apex-derived branches.

    This function calculates shortest paths in a point-cloud graph (QG) from nodes classified
    as "apex" to the specified root point, utilizing a Riemannian metric. It then updates
    node attributes by reclassifying nodes belonging to a specified class and sets edge attributes
    for edges on these paths as useful.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph representing the point-cloud structure. It should contain node
        attributes, including those specified in `class_attribute`, and edge attributes for
        weights used in shortest path computation.
    root_point_riemanian : int or string
        The identifier of the root node within the quotient graph. This is the target node for
        computing shortest paths using Dijkstra's algorithm.
    class_attribute : str, optional
        The name of the node attribute used to determine and reclassify nodes for processing.
        Default is ``'viterbi_class'``.
    weight : str, optional
        The edge attribute name representing edge weights for the shortest path computation.
        Default is ``'weight_sem_paths'``.
    class_apex : int, optional
        The integer value of the node attribute used to identify "apex" nodes.
        Default is ``4``.
    class_branch : int, optional
        The integer value of the node attribute for nodes belonging to the original branch class
        which will be reclassified.
        Default is ``0``.
    class_mainstem : int, optional
        This parameter exists but is not actively used in the function logic. Typically might
        relate to other classifications within the graph structure.
        Default is ``3``.
    new_class_branch : int, optional
        The new classification value assigned to nodes that belonged to the `class_branch` group
        encountered along the shortest paths.
        Default is ``6``.
    """
    G = QG.point_cloud_graph
    list_apex = [x for x, y in QG.nodes(data=True) if y[class_attribute] == class_apex]
    for e in QG.edges():
        QG.edges[e]['useful_path_shortest'] = False
    for leaf in list_apex:
        path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"], weight=weight)
        print(path)
        for n in path:
            if QG.nodes[n][class_attribute] == class_branch:
                QG.nodes[n][class_attribute] = new_class_branch
        for i in range(len(path) - 1):
            if QG.has_edge(path[i], path[i + 1]):
                e = (path[i], path[i + 1])
            else:
                e = (path[i + 1], path[i])
            QG.edges[e]['useful_path_shortest'] = True


def merge_remaining_clusters(quotientgraph, remaining_clusters_class=0, class_attribute='viterbi_class'):
    """
    Merges clusters in a given quotient graph based on inter-class metadata and updates the point cloud graph
    accordingly. The function operates on nodes belonging to the specified 'remaining_clusters_class' and involves
    reassigning nodes in the point cloud graph to new clusters after merging.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph in which clusters are to be merged. This is a networkx graph object containing nodes
        and edges, along with associated metadata.
    remaining_clusters_class : int, optional
        The class value used to identify nodes in the quotient graph that belong to the target cluster to
        be merged. Default is ``0``.
    class_attribute : str, optional
        The attribute name in the quotient graph's node metadata used to distinguish clusters. Nodes with the
        specified 'remaining_clusters_class' value for this attribute will be targeted for merging. Default
        is ``'viterbi_class'``.

    Returns
    -------
    spectral_clustering.quotientgraph.QuotientGraph
        Updated quotient graph after merging clusters. The graph includes revised node and edge metadata based on
        the merging process.

    Notes
    -----
    - Clusters are merged based on the most frequent inter-class links (metadata). If the most frequent linked
      cluster also belongs to the target class to be merged, the second-most frequent linked cluster is chosen.
      This ensures proper merging behavior.
    - Updates are propagated to the point cloud graph (`point_cloud_graph`) within the quotient graph. Nodes in
      the point cloud graph corresponding to the merged clusters are reassigned to the new cluster.
    - Edge attributes 'useful_path_shortest' are preserved during the merging process. In cases of conflicting data
      between nodes being merged, priority is given to attributes of the original edges in the quotient graph.

    """
    list_clusters = [x for x, y in quotientgraph.nodes(data=True) if y[class_attribute] == remaining_clusters_class]
    print(list_clusters)
    G = quotientgraph.point_cloud_graph

    for u in list_clusters:
        count = quotientgraph.compute_quotientgraph_metadata_on_a_node_interclass(u)
        print(count)
        max_class_o = max(count, key=count.get)
        max_class = max_class_o
        # deal with possibility of max_class_ another class to merge. In that case, second best linked is chosent
        # if only linked to another class to merge, merge with this one.
        while max_class in list_clusters and bool(count) and len(count) > 1:
            count.pop(max_class)
            max_class = max(count, key=count.get)
        if bool(count) is False or len(count) == 1:
            max_class = max_class_o

        new_cluster = max_class
        print(new_cluster)
        list_G_nodes_to_change = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == u]
        for g in list_G_nodes_to_change:
            G.nodes[g]['quotient_graph_node'] = new_cluster

        dict1 = {}
        for e in quotientgraph.edges(new_cluster):
            dict1[e] = quotientgraph.edges[e]['useful_path_shortest']
        quotientgraph = nx.contracted_nodes(quotientgraph, new_cluster, u, self_loops=False)
        quotientgraph.point_cloud_graph = G
        dict2 = {}
        for e in quotientgraph.edges(new_cluster):
            dict2[e] = quotientgraph.edges[e]['useful_path_shortest']
            if e in dict1.keys():
                quotientgraph.edges[e]['useful_path_shortest'] = dict1[e]
        print(dict1)
        print(dict2)

    return quotientgraph


def remove_edges_useful_paths(QG):
    """Removes edges that are not marked as useful from the input graph.

    This function takes a graph as input and creates a deep copy of it. It iterates
    through all edges in the copied graph, and if an edge is marked as not part of a
    useful path (as indicated by the 'useful_path_shortest' attribute), it removes
    that edge from the original graph.

    Parameters
    ----------
    QG : spectral_clustering.quotientgraph.QuotientGraph
        Input graph from which edges not marked as useful will be removed. The graph
        must contain an attribute 'useful_path_shortest' on edges indicating whether
        the edge belongs to a useful shortest path or not.
    """
    FG = copy.deepcopy(QG)
    d = FG.edges()
    for e in d:
        if FG.edges[e]['useful_path_shortest'] is False:
            QG.remove_edge(*e)
