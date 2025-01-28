#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random

import networkx as nx
from spectral_clustering.display_and_export import display_and_export_quotient_graph_matplotlib
from spectral_clustering.display_and_export import export_quotient_graph_attribute_on_point_cloud
from treex import Tree
from treex.analysis.hidden_markov import viterbi
from treex.simulation import read_object
from treex.simulation import save_object


def read_pointcloudgraph_into_treex(pointcloudgraph):
    """Reads a point cloud graph object and processes it into a tree diagram before saving and re-loading it.

    This function performs the following operations:
    1. Takes the input `pointcloudgraph` object.
    2. Saves the object with specific attributes using serialization.
    3. Reads the serialized object back into a tree representation.
    4. Returns the re-loaded tree object.

    Parameters
    ----------
    pointcloudgraph : spectral_clustering.pointcloudgraph.PointCloudGraph
        The input graph object representing a point cloud. This input should
        hold sufficient attributes required for further processing into a
        spanning tree format.

    Returns
    -------
    object
        The deserialized tree object created from the saved graph. This object
        includes the processed attributes stored during the serialization step.
    """
    mst = pointcloudgraph
    save_object(mst, 'test_spanning_tree_attributes.p')
    st_tree = read_object('test_spanning_tree_attributes.p')
    return st_tree


def increment_spanning_tree(st_tree, root, t, list_of_nodes, list_att):
    """Recursively increments a spanning tree by traversing and adding nodes from a given root node in a graph.

    The function processes the graph `st_tree`, traverses neighbors of the `root` node, and adds their
    attributes and subtrees to the spanning tree `t`.

    Parameters
    ----------
    st_tree : networkx.Graph
        The graph representing the original structure where traversal
        begins.
    root : Any
        The root node from which the function starts traversing neighbors.
    t : Tree
        The spanning tree being constructed or modified by appending
        subtrees and attributes.
    list_of_nodes : list
        A list that tracks nodes already visited to avoid revisiting
        and infinite loops during the traversal.
    list_att : list
        A list of attributes to be copied from the nodes in the original
        graph `st_tree` to the new spanning tree `t`.
    """
    for neighbor in st_tree.neighbors(root):
        if neighbor not in list_of_nodes:
            s = Tree()
            s.add_attribute_to_id('nx_label', neighbor)
            for att in list_att:
                s.add_attribute_to_id(att, st_tree.nodes[neighbor][att])
            list_of_nodes.append(neighbor)
            increment_spanning_tree(st_tree, neighbor, s, list_of_nodes, list_att)
            t.add_subtree(s)


def build_spanning_tree(st_tree, root_label, list_att=['planarity', 'linearity', 'scattering']):
    """Builds a spanning tree from the given graph structure and initializes it with specific attributes for the root node.

    The spanning tree is constructed iteratively, starting from the root label and adding its connected nodes while
    copying their attributes.

    Parameters
    ----------
    st_tree : networkx.Graph
        The input graph from which the spanning tree is to be constructed. It
        should be a NetworkX graph object with node attributes that will be used
        in the resulting spanning tree.
    root_label : Any
        The label of the root node for the spanning tree. This must exist as a
        node in `st_tree`.
    list_att : list of str, optional
        A list of node attributes to be copied from the original graph `st_tree`
        to the spanning tree. By default, this list is ['planarity', 'linearity',
        'scattering'].

    Returns
    -------
    Tree
        The constructed spanning tree with the attributes `nx_label` and any
        additional attributes from `list_att` added to each node.

    """
    t = Tree()
    t.add_attribute_to_id('nx_label', root_label)
    for att in list_att:
        t.add_attribute_to_id(att, st_tree.nodes[root_label][att])
    list_of_nodes = [root_label]
    increment_spanning_tree(st_tree, root_label, t, list_of_nodes, list_att)
    return t


def add_attributes_to_spanning_tree(st_tree, t, list_att=['planarity', 'linearity', 'scattering']):
    """    Adds specific attributes to a spanning tree structure from a graph.

    The function updates attributes of nodes in the spanning tree using attribute
    values present in the original graph. For each node in the tree, a corresponding
    node from the original graph is identified, and specified attributes are copied
    over. The user can define which attributes to copy via the `list_att` parameter.

    Parameters
    ----------
    st_tree : networkx.Graph
        The spanning tree represented as a NetworkX Graph, containing nodes
        with attributes that are to be copied.
    t : Tree
        A class instance representing another tree object. This tree provides
        methods like `dict_of_ids`, `list_of_ids`, and `add_attribute_to_id`.
        It is used to retrieve the list of node IDs and update their attributes.
    list_att : list of str, optional
        List of attribute names to be added to the spanning tree. By default,
        the attributes 'planarity', 'linearity', and 'scattering' are copied.
    """
    dict = t.dict_of_ids()
    for node in t.list_of_ids():
        st_node = dict[node]['attributes']['nx_label']
        for att in list_att:
            t.add_attribute_to_id(att, st_tree.nodes[st_node][att], node)


def add_viterbi_results_to_quotient_graph(quotientgraph, t, list_semantics=['leaf', 'stem', 'NSP']):
    """Adds Viterbi classification results to the nodes of a quotient graph.

    This function processes the Viterbi classification results stored in a tree-like
    data structure and assigns them to corresponding nodes in a quotient graph. It
    updates the quotient graph nodes with new attributes related to their Viterbi
    classification.

    Parameters
    ----------
    quotientgraph : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph whose nodes are to be updated with Viterbi classification
        results.
    t : object
        A tree-like data structure that contains node identifiers and corresponding
        attributes, including `nx_label` and `viterbi_type`.
    list_semantics : list of str, optional
        A list of strings specifying the semantics to process for Viterbi classifications.
        By default, it includes ['leaf', 'stem', 'NSP'].

    Returns
    -------
    None
        The function updates the quotientgraph in-place, adding a `viterbi_class`
        attribute to its nodes.
    """
    dict = t.dict_of_ids()
    for n in t.list_of_ids():
        qg_node = dict[n]['attributes']['nx_label']
        quotientgraph.nodes[qg_node]['viterbi_class'] = dict[n]['attributes']['viterbi_type']


def create_observation_list(t, list_obs=['planarity', 'linearity', 'scattering'], name='observations'):
    """
    Creates a new attribute in the given tree structure, which contains a list of specific observation
    values from a predefined list of attributes for each node.

    This function iterates through each node in the tree's list of IDs, extracts the requested
    attributes from a dictionary of node information, and stores them as a new attribute in the tree
    structure.

    Parameters
    ----------
    t : Tree
        The tree structure to which the new observation attribute will be added. It must provide
        access to its nodes and their associated attributes.
    list_obs : list of str, optional
        A list of attribute names to be extracted from each node. Defaults to ['planarity',
        'linearity', 'scattering'] if not specified.
    name : str, optional
        The name of the new attribute to be added to each node. Defaults to 'observations' if not
        specified.
    """
    dict = t.dict_of_ids()
    for node in t.list_of_ids():
        obs = []
        for att in list_obs:
            obs.append(dict[node]['attributes'][att])
        t.add_attribute_to_id(name, obs, node)


def viterbi_workflow(minimum_spanning_tree,
                     quotient_graph,
                     root=8,
                     observation_list_import=['planarity2', 'linearity', 'intra_class_node_number'],
                     initial_distribution=[1, 0],
                     transition_matrix=[[0.2, 0.8], [0, 1]],
                     parameters_emission=[[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]]):
    """
    Executes the Viterbi algorithm on input graphs and their corresponding data, enabling the
    classification and visualization of node attributes based on observed and derived metrics.
    The function processes a Minimum Spanning Tree (MST) and a Quotient Graph, preparing and
    embedding Viterbi results into these structures, while exporting their graphical and numerical
    details for further applications.

    Parameters
    ----------
    minimum_spanning_tree : object
        A data structure representing the Minimum Spanning Tree (MST) of a graph.
    quotient_graph : spectral_clustering.quotientgraph.QuotientGraph
        The quotient graph to which Viterbi results are added after computation.
    root : int, optional
        The root node of the spanning tree, default is 8.
    observation_list_import : list of str, optional
        List of attribute names to be considered during tree building and observations,
        default is ['planarity2', 'linearity', 'intra_class_node_number'].
    initial_distribution : list of float, optional
        The initial state probability distribution for the Viterbi algorithm,
        default is [1, 0].
    transition_matrix : list of list of float, optional
        The state transition probabilities for the Hidden Markov Model,
        default is [[0.2, 0.8], [0, 1]].
    parameters_emission : list of list of list of float, optional
        Parameters for Gaussian emission probabilities, where each sublist corresponds
        to the mean and standard deviation of the Gaussian distributions for a given state,
        default is [[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]].
    """
    st_tree = read_pointcloudgraph_into_treex(pointcloudgraph=minimum_spanning_tree)
    rt = root
    t = build_spanning_tree(st_tree, rt, list_att=observation_list_import)
    create_observation_list(t, list_obs=observation_list_import)
    #########################################################################
    initial_distribution = initial_distribution
    # initial_distribution = [1, 0, 0]
    transition_matrix = transition_matrix
    # transition_matrix = [[0.2, 0, 0.8], [0, 0.8, 0.2], [0, 0.8, 0.2]]
    # transition_matrix = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
    # insertion bruit
    # transition_matrix = [[0.2, 0.7, 0.1], [0, 0.9, 0.1], [0.3, 0.3, 0.4]]
    continuous_obs = True

    if continuous_obs:  # observations are Gaussian
        parameterstot = parameters_emission

        def gen_emission(k, parameters):  # Gaussian emission
            return random.gauss(parameters[k][0], parameters[k][1])

        def pdf_emission_dim1(x, moy, sd):  # Gaussian emission
            return 1.0 / (sd * math.sqrt(2 * math.pi)) * math.exp(
                -1.0 / (2 * sd ** 2) * (x - moy) ** 2)

        def pdf_emission_dimn(x, k, parameterstot):
            p = 1
            if len(parameterstot) == 1:
                p = pdf_emission_dim1(x, moy=parameterstot[0][k][0], sd=parameterstot[0][k][1])
                return p
            else:
                for i in range(len(parameterstot[0])):
                    p *= pdf_emission_dim1(x[i], moy=parameterstot[k][i][0], sd=parameterstot[k][i][1])
                return p

    viterbi(t, 'observations', initial_distribution, transition_matrix, pdf_emission_dimn, parameterstot)

    add_viterbi_results_to_quotient_graph(minimum_spanning_tree, t, list_semantics=['leaf', 'stem', 'NSP'])
    add_viterbi_results_to_quotient_graph(quotient_graph, t, list_semantics=['leaf', 'stem', 'NSP'])
    # display_and_export_quotient_graph_matplotlib(quotient_graph=QG_t, node_sizes=20, filename="quotient_graph_observation", data_on_nodes='observations', data=True, attributekmeans4clusters = False)
    display_and_export_quotient_graph_matplotlib(qg=minimum_spanning_tree, node_sizes=20,
                                                 name="quotient_graph_viterbi", data_on_nodes='viterbi_class',
                                                 data=True, attributekmeans4clusters=False)
    export_quotient_graph_attribute_on_point_cloud(quotient_graph, attribute='viterbi_class')


######### Main

if __name__ == '__main__':
    #################################### toy example
    QG_toy = nx.Graph()
    mutige = 0.3
    sigmatige = 0.1
    mupet = 0.8
    sigmapet = 0.2
    muf = 0.8
    sigmaf = 0.3
    QG_toy.add_nodes_from([
        (1, {"observation": random.gauss(mutige, sigmatige)}),
        (2, {"observation": random.gauss(mupet, sigmapet)}),
        (3, {"observation": random.gauss(muf, sigmaf)}),
        (4, {"observation": random.gauss(mupet, sigmapet)}),
        (5, {"observation": random.gauss(muf, sigmaf)}),
        (6, {"observation": random.gauss(mupet, sigmapet)}),
        (7, {"observation": random.gauss(muf, sigmaf)}),
        (8, {"observation": random.gauss(mupet, sigmapet)}),
        (9, {"observation": random.gauss(muf, sigmaf)})
    ])

    QG_toy.add_edges_from([(1, 2), (2, 3), (1, 4), (4, 5), (1, 6), (6, 7), (1, 8), (8, 9)])
    #################################

    st_tree = read_pointcloudgraph_into_treex(pointcloudgraph=QG_toy)
    rt = 8
    # t = build_spanning_tree(st_tree, rt, list_att=['observation'])

    t = build_spanning_tree(st_tree, rt, list_att=['planarity2', 'linearity', 'intra_class_node_number'])

    # def create_observation_list(t, list_obs=['planarity2', 'linearity']):
    #   dict = t.dict_of_ids()
    #    for node in t.list_of_ids():
    #        obs = []
    #        for att in list_obs:
    #            obs.append(dict[node]['attributes'][att])
    #        t.add_attribute_to_id('observations', obs, node)

    create_observation_list(t)

    #########################################################################

    initial_distribution = [1, 0]
    # initial_distribution = [1, 0, 0]
    transition_matrix = [[0.2, 0.8], [0, 1]]
    # transition_matrix = [[0.2, 0, 0.8], [0, 0.8, 0.2], [0, 0.8, 0.2]]
    # transition_matrix = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
    # insertion bruit
    # transition_matrix = [[0.2, 0.7, 0.1], [0, 0.9, 0.1], [0.3, 0.3, 0.4]]
    continuous_obs = True

    if continuous_obs:  # observations are Gaussian
        parameterstot = [[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]]


        # parameterstot = [[[0.3, 0.2], [0.8, 0.2], [0.0, 0.05]], [[0.8, 0.2], [0.3, 0.2], [0.1, 0.2]], [[0.30, 0.30], [0.8, 0.2], [0.20, 0.1]]]
        # parameterstot = [[[0.73, 0.1], [0.25, 0.1], [0.03, 0.05]], [[0.66, 0.04], [0.33, 0.05], [0.0001, 0.1]],
        #                 [[0.10, 0.20], [0.6, 0.35], [0.20, 0.1]]]
        # parameterstot = [[[0.8, 0.6], [0.2, 0.3], [0.8, 0.2]]]
        # insertion bruit
        # parameterstot = [[[0.2, 0.2], [0.8, 0.2], [1000, 10000]], [[0.6, 0.2], [0.3, 0.2], [1000, 10000]], [[0.5, 1], [0.5, 1], [110, 100]]]
        def gen_emission(k, parameters):  # Gaussian emission
            return random.gauss(parameters[k][0], parameters[k][1])


        def pdf_emission_dim1(x, moy, sd):  # Gaussian emission
            return 1.0 / (sd * math.sqrt(2 * math.pi)) * math.exp(
                -1.0 / (2 * sd ** 2) * (x - moy) ** 2)


        def pdf_emission_dimn(x, k, parameterstot):
            p = 1
            if len(parameterstot) == 1:
                p = pdf_emission_dim1(x, moy=parameterstot[0][k][0], sd=parameterstot[0][k][1])
                return p
            else:
                for i in range(len(parameterstot[0])):
                    p *= pdf_emission_dim1(x[i], moy=parameterstot[k][i][0], sd=parameterstot[k][i][1])
                return p

    viterbi(t, 'observations', initial_distribution, transition_matrix, pdf_emission_dimn, parameterstot)

    add_viterbi_results_to_quotient_graph(QG_toy, t, list_semantics=['leaf', 'stem', 'NSP'])
    display_and_export_quotient_graph_matplotlib(qg=QG_toy, node_sizes=20, name="quotient_graph_viterbi",
                                                 data_on_nodes='viterbi_class', data=True,
                                                 attributekmeans4clusters=False)

    export_quotient_graph_attribute_on_point_cloud(QG_toy, attribute='viterbi_class')

    # export_quotient_graph_attribute_on_point_cloud(QG, attribute='linearity')
