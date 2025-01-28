#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library allowing to optimize a score based on the topology ...

from collections import Counter

import networkx as nx
import numpy as np

from spectral_clustering.display_and_export import display_and_export_quotient_graph_matplotlib
from spectral_clustering.display_and_export import export_some_graph_attributes_on_point_cloud
from spectral_clustering.quotientgraph_operations import update_quotient_graph_attributes_when_node_change_cluster


def define_and_optimize_topological_energy(quotient_graph,
                                           point_cloud_graph,
                                           exports=True,
                                           formulae='improved',
                                           number_of_iteration=1000,
                                           choice_of_node_to_change='max_energy'):
    """Compute and optimize the topological scores of each node in the PointCloudGraph.

    This function first calculates the initial topological scores for each node in
    the `point_cloud_graph` using the `init_topo_scores` function. Following this, it
    optimizes the topological energy using the `optimization_topo_scores` function.

    Parameters
    ----------
    quotient_graph : spectral_clustering.graph.QuotientGraph
        The quotient graph derived from the `PointCloudGraph`.
    point_cloud_graph : spectral_clustering.graph.PointCloudGraph
        The associated distance-based PointCloudGraph.
    exports : bool, optional
        If True, exports the computed scores to a `.txt` file and saves a visualization
        of the scores on the quotient graph as a `.png` image. Default is True.
    formulae : str, optional
        Determines the formula used to compute the topological energy for a node.
        Options are:
        - `'improved'` (default): Uses the improved formula.
        - `'old'`: Uses the old formula.
    number_of_iteration : int, optional
        The number of iterations for topological energy optimization. Default is 1000.
    choice_of_node_to_change : str, optional
        The method used to select a node for changing its cluster. Options are:
        - `'max_energy'` (default): Selects the node with the maximum energy.
        - `'random_proba_energy'`: Selects a node based on a probability distribution
          of energy values.
        - `'max_energy_and_select'`: Selects the maximum energy node with additional
          selection criteria.

    Notes
    -----
    The function consists of two main steps:
    1. Calculation of initial topological scores with `init_topo_scores`.
    2. Optimization of scores using `optimization_topo_scores`.

    At the end of the process, a summary message is printed to indicate that the
    optimization has been completed.
    """
    init_topo_scores(quotient_graph=quotient_graph,
                     point_cloud_graph=point_cloud_graph,
                     exports=exports,
                     formulae=formulae)
    optimization_topo_scores(quotientgraph=quotient_graph,
                             pointcloudgraph=point_cloud_graph,
                             exports=exports,
                             number_of_iteration=number_of_iteration,
                             choice_of_node_to_change=choice_of_node_to_change,
                             formulae=formulae)

    print('Optimization of topological energy : Done')


def init_topo_scores(quotient_graph, point_cloud_graph, exports=True, formulae='improved'):
    """
    Calculate the topological scores for each node in the PointCloudGraph and associated
    energy metrics for the QuotientGraph.

    This function computes the number of adjacent clusters in the PointCloudGraph that are
    different from the cluster of the considered node. It also calculates a per-node energy
    score for the QuotientGraph and the global topological energy of the entire QuotientGraph,
    which is the sum of the energies of its nodes.

    Parameters
    ----------
    quotient_graph : spectral_clustering.graph.QuotientGraph
        The QuotientGraph object representing the coarse-grained view of the PointCloudGraph.
    point_cloud_graph : spectral_clustering.graph.PointCloudGraph
        The PointCloudGraph object, typically a distance-based graph, associated with
        the quotient graph.
    exports : bool, optional
        If True, exports the computed scores to a `.txt` file and a matplotlib
        visualization (`.png`) of the quotient graph. Default is True.
    formulae : {'improved', 'old'}, optional
        Specifies the formula used to compute the topological energy for each node:
        - 'improved': A refined computation method based on normalized counts of adjacent
          clusters.
        - 'old': A simpler, earlier computational approach. Default is 'improved'.

    Returns
    -------
    None
        This function modifies `quotient_graph` and `point_cloud_graph` in place by adding
        attributes such as:
        - `number_of_adj_labels` for the PointCloudGraph nodes.
        - `topological_energy` for the QuotientGraph nodes.
        Additionally, the global topological energy is stored in the `quotient_graph.graph`
        attribute: `global_topological_energy`.

    Notes
    -----
    - When `formulae='improved'`, the energy calculation considers the normalized
      contributions of connections to clusters different from the node’s own cluster.
    - This function works in-place, directly modifying the input graph objects.
    - If `exports` is True, the function will generate:
        - A `.txt` file containing the computed energy scores for the PointCloudGraph.
        - A `.png` visualization of the QuotientGraph with node energies annotated.

    See Also
    --------
    export_some_graph_attributes_on_point_cloud : Export node attributes for visualization or analysis.
    display_and_export_quotient_graph_matplotlib : Display and save a visualization of the graph structure.
    """
    QG = quotient_graph
    G = point_cloud_graph

    # Determinate a score for each vertex in a quotient node. Normalized by the number of neighbors
    # init
    maxNeighbSize = 0
    for u in G.nodes:
        G.nodes[u]['number_of_adj_labels'] = 0
    for u in QG.nodes:
        QG.nodes[u]['topological_energy'] = 0
    # global score for the entire graph
    QG.graph['global_topological_energy'] = 0
    # for to compute the score of each vertex
    if formulae == 'old':
        for v in G.nodes:
            number_of_neighb = len([n for n in G[v]])
            for n in G[v]:
                if G.nodes[v]['quotient_graph_node'] != G.nodes[n]['quotient_graph_node']:
                    G.nodes[v]['number_of_adj_labels'] += 1
            G.nodes[v]['number_of_adj_labels'] /= number_of_neighb
            u = G.nodes[v]['quotient_graph_node']
            QG.nodes[u]['topological_energy'] += G.nodes[v]['number_of_adj_labels']
            QG.graph['global_topological_energy'] += G.nodes[v]['number_of_adj_labels']
    elif formulae == 'improved':
        for v in G.nodes:
            list_neighb_clust = []
            for n in G[v]:
                list_neighb_clust.append(G.nodes[n]['quotient_graph_node'])
            number_of_clusters = len(Counter(list_neighb_clust).keys())
            if number_of_clusters == 1 and list_neighb_clust[0] == G.nodes[v]['quotient_graph_node']:
                G.nodes[v]['number_of_adj_labels'] = 0
            else:
                number_same = list_neighb_clust.count(G.nodes[v]['quotient_graph_node'])
                number_diff = len(list_neighb_clust) - number_same
                G.nodes[v]['number_of_adj_labels'] = number_diff / \
                                                     (number_diff + (number_of_clusters - 1) * number_same)
            u = G.nodes[v]['quotient_graph_node']
            QG.nodes[u]['topological_energy'] += G.nodes[v]['number_of_adj_labels']
            QG.graph['global_topological_energy'] += G.nodes[v]['number_of_adj_labels']

    if exports:
        export_some_graph_attributes_on_point_cloud(G, graph_attribute='number_of_adj_labels',
                                                    filename='graph_attribute_energy_init.txt')

        display_and_export_quotient_graph_matplotlib(QG, node_sizes=20, name="quotient_graph_matplotlib_energy_init",
                                                     data_on_nodes='topological_energy')


def optimization_topo_scores(quotientgraph, pointcloudgraph, exports=True, number_of_iteration=1000,
                             choice_of_node_to_change='max_energy', formulae='improved'):
    """Optimizes the topological scores by iteratively modifying clusters of nodes in
    a point cloud graph and updating the corresponding energies.

    The function starts by selecting a node based on its energy using the specified
    selection method. The selected node changes its cluster to one of its neighbor's
    clusters. The function then updates the global topological energy and other graph
    properties accordingly. The process is repeated for a specified number of iterations
    and can export the results, including graphs showing energy evolution.

    Parameters
    ----------
    quotientgraph : spectral_clustering.graph.QuotientGraph
        The quotient graph that contains meta-information about clusters and associated
        energy values. This graph gets updated with changes during optimization.
    pointcloudgraph : spectral_clustering.graph.PointCloudGraph
        The point cloud graph where each node represents a point and is associated with
        cluster data. This is the main graph on which optimization operations are performed.
    exports : bool, optional
        If True, the function will export the graph showing the evolution of the global
        energy, the quotient graph with energy values for each node, and point-cloud-related
        data. Default is True.
    number_of_iteration : int, optional
        The total number of iterations to perform for optimization. Default is 1000.
    choice_of_node_to_change : str, optional
        Method used to select the node for changing its cluster. Options include:
        - 'max_energy': Select nodes with the maximum energy.
        - 'random_proba_energy': Select nodes probabilistically based on normalized energy.
        - 'max_energy_and_select': Select nodes with maximum energy, applying additional
          constraints to avoid repeated selections.
        Default is 'max_energy'.
    formulae : str, optional
        The formula used for updating the energy based on cluster changes. Options:
        - 'old': Uses the original formula for energy updates.
        - 'improved': Uses an enhanced formula to adjust energy dependencies.
        This must match the formula used during the initialization of the system.
        Default is 'improved'.

    Returns
    -------
    None
        This function modifies the given input graphs (`quotientgraph` and
        `pointcloudgraph`) in place and optionally exports results as files.

    Notes
    -----
    - The function internally tracks the global energy values across iterations and can
      export these results visually as scatter plots.
    - Clusters for the nodes are determined by their neighbors' attributes and random
      weight probabilities derived from their local environment.
    """
    G = pointcloudgraph

    # nombre d'itérations
    iter = number_of_iteration
    # Liste contenant l'énergie globale du graph
    evol_energy = [quotientgraph.graph['global_topological_energy']]

    # list to detect repetition in 'max_energy_and_select'
    detect_rep = []
    ban_list = []

    # Start loops for the number of iteration specified
    for i in range(iter):

        # Choice of point to move from a cluster to another.

        if choice_of_node_to_change == 'max_energy':
            # Creation of a dictionary with the energy per node
            energy_per_node = nx.get_node_attributes(G, 'number_of_adj_labels')
            # Extraction of a random point to treat, use of "smart indexing"
            nodes = np.array(list(energy_per_node.keys()))
            mylist = list(energy_per_node.values())
            myRoundedList = [round(x, 2) for x in mylist]
            node_energies = np.array(myRoundedList)
            maximal_energy_nodes = nodes[node_energies == np.max(node_energies)]
            node_to_change = np.random.choice(maximal_energy_nodes)
        if choice_of_node_to_change == 'random_proba_energy':
            energy_per_node = nx.get_node_attributes(G, 'number_of_adj_labels')
            nodes = np.array(list(energy_per_node.keys()))
            total_energy = quotientgraph.graph['global_topological_energy']
            l = list(energy_per_node.values())
            node_energies = np.array([e / total_energy for e in l])
            node_to_change = np.random.choice(nodes, p=node_energies)
        if choice_of_node_to_change == 'max_energy_and_select':
            energy_per_node = nx.get_node_attributes(G, 'number_of_adj_labels')
            nodes = np.array(list(energy_per_node.keys()))
            node_energies = np.array(list(energy_per_node.values()))
            maximal_energy_nodes = nodes[node_energies == np.max(node_energies)]
            node_to_change = np.random.choice(maximal_energy_nodes)
            if ban_list.count(node_to_change) == 0:
                if detect_rep.count(node_to_change) == 0 and ban_list.count(node_to_change) == 0:
                    detect_rep.append(node_to_change)
                if detect_rep.count(node_to_change) != 0:
                    detect_rep.append(node_to_change)
            if ban_list.count(node_to_change) != 0:
                sort_energy_per_node = {k: v for k, v in
                                        sorted(energy_per_node.items(), key=lambda item: item[1], reverse=True)}
                for c in sort_energy_per_node:
                    if ban_list.count(c) == 0:
                        node_to_change = c
                        if detect_rep.count(node_to_change) == 0:
                            detect_rep.append(node_to_change)
                        else:
                            detect_rep.append(node_to_change)
                        break
            if detect_rep.count(node_to_change) >= G.nearest_neighbors * 2:
                ban_list.append(node_to_change)
                detect_rep = []

        # print()
        # print(i)
        # print(ban_list)
        # print(node_to_change)
        # print(G.nodes[node_to_change]['number_of_adj_labels'])
        # print(G.nodes[node_to_change]['quotient_graph_node'])

        # change the cluster of the node_to_change
        number_of_neighb = len([n for n in G[node_to_change]])
        # attribution for each label a probability depending on the number of points having this label
        # in the neighborhood of node_to_change
        # stocked in a dictionary
        old_cluster = G.nodes[node_to_change]['quotient_graph_node']
        proba_label = {}
        for n in G[node_to_change]:
            if G.nodes[n]['quotient_graph_node'] not in proba_label:
                proba_label[G.nodes[n]['quotient_graph_node']] = 0
            proba_label[G.nodes[n]['quotient_graph_node']] += 1.0 / number_of_neighb

        new_label_proba = np.random.random()
        new_energy = 0
        range_origin = 0
        for l in proba_label:
            if new_label_proba <= range_origin or new_label_proba > range_origin + proba_label[l]:
                new_energy += proba_label[l]
            else:
                G.nodes[node_to_change]['quotient_graph_node'] = l
            range_origin += proba_label[l]

        new_cluster = G.nodes[node_to_change]['quotient_graph_node']

        update_quotient_graph_attributes_when_node_change_cluster(quotientgraph, old_cluster, new_cluster,
                                                                  node_to_change)

        if formulae == 'old':
            # update of energy for the node changed
            previous_energy = G.nodes[node_to_change]['number_of_adj_labels']
            G.nodes[node_to_change]['number_of_adj_labels'] = new_energy
            quotientgraph.graph['global_topological_energy'] += (new_energy - previous_energy)
            u = G.nodes[node_to_change]['quotient_graph_node']
            quotientgraph.nodes[u]['topological_energy'] += new_energy
            quotientgraph.nodes[old_cluster]['topological_energy'] -= previous_energy
            # update of energy for the neighbors
            for n in G[node_to_change]:
                previous_energy = G.nodes[n]['number_of_adj_labels']
                G.nodes[n]['number_of_adj_labels'] = 0
                for v in G[n]:
                    number_of_neighb = len([n for n in G[v]])
                    if G.nodes[n]['quotient_graph_node'] != G.nodes[v]['quotient_graph_node']:
                        G.nodes[n]['number_of_adj_labels'] += 1 / number_of_neighb
                quotientgraph.graph['global_topological_energy'] += (
                        G.nodes[n]['number_of_adj_labels'] - previous_energy)
                u = G.nodes[n]['quotient_graph_node']
                quotientgraph.nodes[u]['topological_energy'] += (G.nodes[n]['number_of_adj_labels'] - previous_energy)

        elif formulae == 'improved':
            # update of energy for the node changed
            list_neighb_clust = []
            previous_energy = G.nodes[node_to_change]['number_of_adj_labels']
            for n in G[node_to_change]:
                list_neighb_clust.append(G.nodes[n]['quotient_graph_node'])
            number_of_clusters = len(Counter(list_neighb_clust).keys())
            if number_of_clusters == 1 and list_neighb_clust[0] == new_cluster:
                G.nodes[node_to_change]['number_of_adj_labels'] = 0
            else:
                number_same = list_neighb_clust.count(G.nodes[node_to_change]['quotient_graph_node'])
                number_diff = len(list_neighb_clust) - number_same
                G.nodes[node_to_change]['number_of_adj_labels'] = number_diff / (
                        number_diff + (number_of_clusters - 1) * number_same)

            new_energy = G.nodes[node_to_change]['number_of_adj_labels']
            quotientgraph.graph['global_topological_energy'] += (new_energy - previous_energy)
            quotientgraph.nodes[new_cluster]['topological_energy'] += new_energy
            quotientgraph.nodes[old_cluster]['topological_energy'] -= previous_energy

            # update energy of the neighbors
            for n in G[node_to_change]:
                list_neighb_clust = []
                previous_energy = G.nodes[n]['number_of_adj_labels']
                G.nodes[n]['number_of_adj_labels'] = 0
                for v in G[n]:
                    list_neighb_clust.append(G.nodes[v]['quotient_graph_node'])
                number_of_clusters = len(Counter(list_neighb_clust).keys())
                if number_of_clusters == 1 and list_neighb_clust[0] == G.nodes[n]['quotient_graph_node']:
                    G.nodes[n]['number_of_adj_labels'] = 0
                else:
                    number_same = list_neighb_clust.count(G.nodes[n]['quotient_graph_node'])
                    number_diff = len(list_neighb_clust) - number_same
                    G.nodes[n]['number_of_adj_labels'] = number_diff / (
                            number_diff + (number_of_clusters - 1) * number_same)
                new_energy = G.nodes[n]['number_of_adj_labels']
                quotientgraph.graph['global_topological_energy'] += (new_energy - previous_energy)
                u = G.nodes[n]['quotient_graph_node']
                quotientgraph.nodes[u]['topological_energy'] += (new_energy - previous_energy)

        # update list containing all the differents stages of energy obtained
        evol_energy.append(quotientgraph.graph['global_topological_energy'])

    quotientgraph.delete_empty_edges_and_nodes()
    quotientgraph.point_cloud_graph = G
    if exports:
        from matplotlib import pyplot as plt
        figure = plt.figure(1)
        figure.clf()
        figure.gca().set_title("Evolution_of_energy")
        plt.autoscale(enable=True, axis='both', tight=None)
        figure.gca().scatter(range(len(evol_energy)), evol_energy, color='blue')
        figure.set_size_inches(10, 10)
        figure.subplots_adjust(wspace=0, hspace=0)
        figure.tight_layout()
        figure.savefig('Evolution_global_energy')
        print("Export énergie globale")

        display_and_export_quotient_graph_matplotlib(quotientgraph, node_sizes=20,
                                                     name="quotient_graph_matplotlib_energy_final",
                                                     data_on_nodes='topological_energy')
        export_some_graph_attributes_on_point_cloud(G, graph_attribute='number_of_adj_labels',
                                                    filename='graph_attribute_energy_final.txt')

        export_some_graph_attributes_on_point_cloud(G, graph_attribute='quotient_graph_node',
                                                    filename='graph_attribute_quotient_graph_node_final.txt')
