########### Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from treex import *

from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as spsp
import scipy as sp
import sklearn.cluster as skc
import sklearn as sk
import spectral_clustering.similarity_graph as sgk
import open3d as open3d
import spectral_clustering.PointCloudGraph as kpcg
import time
import operator
from collections import Counter

########### Définition classe

class QuotientGraph(nx.Graph):

    def __init__(self):
        super().__init__()
        self.seed_colors = None
        self.graph_labels_dict = None
        self.label_count = None
        self.nodes_coordinates = None
        self.global_topological_energy = None
        self.point_cloud_graph = None

    def build_QuotientGraph_from_PointCloudGraph(self, G, labels_from_cluster, filename="pcd_attribute.txt"):
        """Construction of a quotient graph with a PointCloudGraph. Region growing approach using a first clustering
        of the points. Each region becomes a node of que quotient graph.

        Parameters
        ----------
        G : PointCloudGraph
        The associated distance-based graph
        labels_from_cluster : np.array
        Each point of que point cloud/ distance based graph is associated to a label.
        If the labels are an attribute to each node of the PointCloudGraph, use the following lines to convert :
        labels_qg = [k for k in dict(G.nodes(data = 'quotient_graph_node')).values()]
        labels_from_cluster = np.asarray(labels_qg)[:, np.newaxis]

        Returns
        -------
        Nothing
        Update :
        """

        kmeans_labels = labels_from_cluster
        connected_component_labels = np.zeros(kmeans_labels.shape, dtype=int)
        current_cc_size = 0
        connected_component_size = []
        seed_kmeans_labels = []
        # This line has been added to obtain particular colors when the initial clustering is done with four clusters
        # especially kmeans.
        lablist = kmeans_labels.tolist()
        my_count = pd.Series(lablist).value_counts()
        if len(my_count) == 4:
            cluster_colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
        else:
            jet = cm.get_cmap('jet', len(my_count))
            cluster_colors = jet(range(len(my_count)))
            list_labels = np.unique(lablist).tolist()
        seed_colors = []
        label_count = 0
        visited = np.zeros(kmeans_labels.shape, dtype=int)
        queue = []

        # Region growing algorithm to extract connected components made of the same k-means class.

        for i in range(kmeans_labels.shape[0]):

            # Find a seed to initialize the seed fill process.

            if visited[i] == 0:
                seed = i
                visited[i] = 1
                queue.append(i)
                seed_kmeans_labels.append(kmeans_labels[seed])
                if len(my_count) == 4:
                    seed_colors.append(cluster_colors[kmeans_labels[seed][0]])
                else:
                    seed_colors.append(cluster_colors[list_labels.index(kmeans_labels[seed][0])][:])
                current_cc_size = 0

                # Region growing from the specified seed.

                while len(queue) != 0:

                    current = queue.pop(0)
                    connected_component_labels[current] = label_count
                    current_cc_size += 1

                    for n in G[current]:
                        if kmeans_labels[n] == kmeans_labels[seed] and visited[n] == 0:
                            queue.append(n)
                            visited[n] = 1

                connected_component_size.append(current_cc_size)
                label_count += 1

        # Create quotient graph nodes.

        self.add_nodes_from(range(label_count))

        node_kmeans_labels_values = dict(
            zip(np.asarray(self.nodes()), np.transpose(np.asarray(seed_kmeans_labels))[0]))
        nx.set_node_attributes(self, node_kmeans_labels_values, 'kmeans_labels')

        # Create quotient graph edges.

        intra_edge_weight_sum = np.zeros(label_count, dtype=np.float64)
        intra_edge_count = np.zeros(label_count, dtype=np.int)

        for (u, v) in G.edges:
            a = connected_component_labels[u, 0]
            b = connected_component_labels[v, 0]

            # Inter-class edge case:

            if a != b:
                if not self.has_edge(a, b):
                    w = G.edges[u, v]['weight']
                    self.add_edge(a, b, inter_class_edge_weight=w, inter_class_edge_number=1)
                    #nx.set_edge_attributes(self, G.edges[u, v]['weight'], 'inter_class_edge_weight')
                    #self.edges[a, b]['inter_class_edge_weight'] = G.edges[u, v]['weight']
                    #nx.set_edge_attributes(self, 1, 'inter_class_edge_number')
                    #self.edges[a, b]['inter_class_edge_number'] = 1
                else:
                    self.edges[a, b]['inter_class_edge_weight'] += G.edges[u, v]['weight']
                    self.edges[a, b]['inter_class_edge_number'] += 1

            # Intra-class edge case:

            else:
                intra_edge_weight_sum[a] += G.edges[u, v]['weight']
                intra_edge_count[a] += 1

        # Assign to each point cloud graph node the corresponding quotient graph node.

        node_cc = dict(
            zip(np.asarray(G.nodes()), np.transpose(np.asarray(connected_component_labels))[0]))
        nx.set_node_attributes(G, node_cc, 'quotient_graph_node')

        nx.set_node_attributes(self, dict(
            zip(np.asarray(self.nodes()), np.transpose(np.asarray(connected_component_size)))), 'intra_class_node_number')

        nx.set_node_attributes(self, dict(
            zip(np.asarray(self.nodes()), np.transpose(np.asarray(intra_edge_weight_sum)))), 'intra_class_edge_weight')

        nx.set_node_attributes(self, dict(
            zip(np.asarray(self.nodes()), np.transpose(np.asarray(intra_edge_count)))), 'intra_class_edge_number')

        if len(my_count) == 4:
            nx.set_node_attributes(self, dict(
                zip(np.asarray(self.nodes()), np.transpose(np.asarray(seed_colors)))), 'seed_colors')
        else:
            nx.set_node_attributes(self, dict(
                zip(np.asarray(self.nodes()), seed_colors)), 'seed_colors')

        #self.seed_colors = seed_colors
        self.label_count = label_count
        self.point_cloud_graph = G
        #self.graph_labels_dict = dict(zip(np.asarray(self.nodes()), range(label_count)))
        kpcg.export_anything_on_point_cloud(G, attribute=connected_component_labels, filename=filename)


    def delete_small_clusters(self, min_number_of_element_in_a_quotient_node=50):
        nodes_to_remove = []
        for u in self.nodes:
            if self.nodes[u]['intra_class_node_number'] < min_number_of_element_in_a_quotient_node:
                adjacent_clusters = [n for n in self[u]]
                max_number_of_nodes_in_adjacent_clusters = 0
                for i in range(len(adjacent_clusters)):
                    if self.nodes[adjacent_clusters[i]][
                            'intra_class_node_number'] > max_number_of_nodes_in_adjacent_clusters:
                        max_number_of_nodes_in_adjacent_clusters = self.nodes[adjacent_clusters[i]][
                            'intra_class_node_number']
                        new_cluster = adjacent_clusters[i]
                # Opération de fusion du petit cluster avec son voisin le plus conséquent.
                # Mise à jour des attributs de la grande classe et suppression de la petite classe
                self.nodes[new_cluster]['intra_class_node_number'] += self.nodes[u]['intra_class_node_number']
                self.nodes[new_cluster]['intra_class_edge_weight'] += (self.nodes[u]['intra_class_edge_weight']
                                                                     + self.edges[new_cluster, u][
                                                                         'inter_class_edge_weight'])
                self.nodes[new_cluster]['intra_class_edge_number'] += (self.nodes[u]['intra_class_edge_number']
                                                                     + self.edges[new_cluster, u][
                                                                         'inter_class_edge_number'])

                # Mise à jour du lien avec le PointCloudGraph d'origine
                for v in G.nodes:
                    if G.nodes[v]['quotient_graph_node'] == u:
                        G.nodes[v]['quotient_graph_node'] = new_cluster

                # Mise à jour des edges
                for i in range(len(adjacent_clusters)):
                    if self.has_edge(new_cluster, adjacent_clusters[i]) is False:
                        self.add_edge(new_cluster, adjacent_clusters[i],
                                      inter_class_edge_weight=self.edges[u, adjacent_clusters[i]]['inter_class_edge_weight'],
                                      inter_class_edge_number=self.edges[u, adjacent_clusters[i]]['inter_class_edge_number'])
                    elif self.has_edge(new_cluster, adjacent_clusters[i]) and new_cluster != adjacent_clusters[i]:
                        self.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                            self.edges[u, adjacent_clusters[i]]['inter_class_edge_weight']
                        self.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                            self.edges[u, adjacent_clusters[i]]['inter_class_edge_number']
                nodes_to_remove.append(u)

        self.remove_nodes_from(nodes_to_remove)


    def compute_quotientgraph_nodes_coordinates(self, G):
        """Compute X Y Z coordinates for each node of the quotient graph using the underlying point of the
        PointCloudGraph (the distance_based_graph). The method just consist in computing the mean of each coordinates of
        the underlying points of a quotient graph node.

        Parameters
        ----------
        G : PointCloudGraph

        Returns
        -------

        """
        # Calcul de coordonnées moyennes pour chaque noeud du graphe quotient, dans le but d'afficher en 3D le graphe.
        new_classif = np.asarray(list((dict(G.nodes(data='quotient_graph_node')).values())))
        new_classif = new_classif[:, np.newaxis]
        pcd_attribute = np.concatenate([G.nodes_coords, new_classif], axis=1)
        sorted_pcd_attribute_by_quotient_graph_attribute = pcd_attribute[np.argsort(pcd_attribute[:, 3])]
        nodes_coords_moy = np.zeros((len(self), 3))
        j = 0
        for n in self.nodes:
            X = []
            Y = []
            Z = []
            for i in range(pcd_attribute.shape[0]):
                if sorted_pcd_attribute_by_quotient_graph_attribute[i, 3] == n:
                    X.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 0])
                    Y.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 1])
                    Z.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 2])
            nodes_coords_moy[j, 0] = np.mean(X)
            nodes_coords_moy[j, 1] = np.mean(Y)
            nodes_coords_moy[j, 2] = np.mean(Z)
            j += 1
        self.nodes_coordinates = nodes_coords_moy

    def init_topo_scores(self, G, exports=True, formulae='improved'):
        """Compute the topological scores of each node of the PointCloudGraph. It counts the number of adjacent clusters
        different from the cluster of the node considered.
        It computes a energy per node of the quotient graph
        It also computes the global score/energy of the quotient graph by computing the sum of the energy of each nodes.


        Parameters
        ----------
        G : PointCloudGraph
        The associated distance-based graph
        exports : Boolean
        Precise if the user want to export the scores values on the point cloud in a .txt file and the scores on a
        matplotlib picture .png of the quotient graph.
        formulae : 'improved' or 'old' is the way to compute topological energy for a node.

        Returns
        -------
        """
        # Determinate a score for each vertex in a quotient node. Normalized by the number of neighbors
        # init
        maxNeighbSize = 0
        for u in G.nodes:
            G.nodes[u]['number_of_adj_labels'] = 0
        for u in self.nodes:
            self.nodes[u]['topological_energy'] = 0
        # global score for the entire graph
        self.global_topological_energy = 0
        # for to compute the score of each vertex
        if formulae == 'old':
            for v in G.nodes:
                number_of_neighb = len([n for n in G[v]])
                for n in G[v]:
                    if G.nodes[v]['quotient_graph_node'] != G.nodes[n]['quotient_graph_node']:
                        G.nodes[v]['number_of_adj_labels'] += 1
                G.nodes[v]['number_of_adj_labels'] /= number_of_neighb
                u = G.nodes[v]['quotient_graph_node']
                self.nodes[u]['topological_energy'] += G.nodes[v]['number_of_adj_labels']
                self.global_topological_energy += G.nodes[v]['number_of_adj_labels']
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
                    G.nodes[v]['number_of_adj_labels'] = number_diff / (number_diff + (number_of_clusters-1)*number_same)
                u = G.nodes[v]['quotient_graph_node']
                self.nodes[u]['topological_energy'] += G.nodes[v]['number_of_adj_labels']
                self.global_topological_energy += G.nodes[v]['number_of_adj_labels']

        if exports:
            export_some_graph_attributes_on_point_cloud(G, graph_attribute='number_of_adj_labels',
                                                        filename='graph_attribute_energy_init.txt')

            display_and_export_quotient_graph_matplotlib(self, node_sizes=20,
                                                        filename="quotient_graph_matplotlib_energy_init",
                                                        data_on_nodes='topological_energy')


    def update_quotient_graph_attributes_when_node_change_cluster(self, old_cluster, new_cluster, node_to_change, G):
        """When the node considered change of cluster, this function updates all the attributes associated to the
        quotient graph.

        Parameters
        ----------
        node_to_change : int
        The node considered and changed in an other cluster
        old_cluster : int
        The original cluster of the node
        new_cluster : int
        The new cluster of the node
        G : PointCloudGraph

        Returns
        -------
        """
        # update Quotient Graph attributes
        # Intra_class_node_number
        #print(old_cluster)
        #print(new_cluster)
        self.nodes[old_cluster]['intra_class_node_number'] += -1
        self.nodes[new_cluster]['intra_class_node_number'] += 1

        # intra_class_edge_weight, inter_class_edge_number
        for ng in G[node_to_change]:
            if old_cluster != G.nodes[ng]['quotient_graph_node'] and new_cluster == G.nodes[ng]['quotient_graph_node']:
                self.nodes[new_cluster]['intra_class_edge_weight'] += G.edges[ng, node_to_change]['weight']
                self.nodes[new_cluster]['intra_class_edge_number'] += 1
                self.edges[old_cluster, new_cluster]['inter_class_edge_weight'] += - G.edges[ng, node_to_change][
                    'weight']
                self.edges[old_cluster, new_cluster]['inter_class_edge_number'] += - 1

            if old_cluster == G.nodes[ng]['quotient_graph_node'] and new_cluster != G.nodes[ng]['quotient_graph_node']:
                self.nodes[old_cluster]['intra_class_edge_weight'] += -G.edges[ng, node_to_change]['weight']
                self.nodes[old_cluster]['intra_class_edge_number'] += -1
                cluster_adj = G.nodes[ng]['quotient_graph_node']
                if self.has_edge(new_cluster, cluster_adj) is False:
                    self.add_edge(new_cluster, cluster_adj, inter_class_edge_weight=None, inter_class_edge_number=None)
                self.edges[new_cluster, cluster_adj]['inter_class_edge_weight'] += G.edges[ng, node_to_change]['weight']
                self.edges[new_cluster, cluster_adj]['inter_class_edge_number'] += 1

            if old_cluster != G.nodes[ng]['quotient_graph_node'] and new_cluster != G.nodes[ng]['quotient_graph_node']:
                cluster_adj = G.nodes[ng]['quotient_graph_node']
                if self.has_edge(new_cluster, cluster_adj) is False:
                    self.add_edge(new_cluster, cluster_adj,
                                  inter_class_edge_weight=G.edges[ng, node_to_change]['weight'],
                                  inter_class_edge_number=1)
                else:
                    self.edges[new_cluster, cluster_adj]['inter_class_edge_weight'] += G.edges[ng, node_to_change][
                        'weight']
                    self.edges[new_cluster, cluster_adj]['inter_class_edge_number'] += 1
                if new_cluster != old_cluster:
                    self.edges[old_cluster, cluster_adj]['inter_class_edge_weight'] += - \
                        G.edges[ng, node_to_change]['weight']
                    self.edges[old_cluster, cluster_adj]['inter_class_edge_number'] += - 1

    def check_connectivity_of_modified_cluster(self, old_cluster, new_cluster, node_to_change, G):
        """When the node considered change of cluster, this function checks if the old cluster of the node is still
        connected or forms two different parts.
        This function is to use after the change occurred in the distance-based graph

        Parameters
        ----------
        node_to_change : int
        The node considered and changed in an other cluster
        old_cluster : int
        The original cluster of the node
        new_cluster : int
        The new cluster of the node
        G : PointCloudGraph

        Returns
        -------
        Boolean
        True if the connectivity was conserved
        """
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

    #def rebuilt_topological_quotient_graph


    def delete_empty_edges_and_nodes(self):
        """Delete edges from the quotient graph that do not represent any edges in the distance-based graph anymore.
        Delete nodes from the quotient graph that do not represent any nodes of the distance-based graph.
        Update of the topological structure by removal only.

        Parameters
        ----------
        Returns
        -------
        """
        to_remove =[]
        list = [e for e in self.edges]
        for e in list:
            if self.edges[e[0], e[1]]['inter_class_edge_number'] == 0:
                to_remove.append(e)
        print(to_remove)
        self.remove_edges_from(to_remove)
        to_remove=[]
        for n in self.nodes:
            if self.nodes[n]['intra_class_node_number'] == 0:
                to_remove.append(n)
        self.remove_nodes_from(to_remove)
        print(to_remove)



    def optimization_topo_scores(self, G, exports=True, number_of_iteration=1000, choice_of_node_to_change='max_energy', formulae='improved'):
        """This function needs to init the topological_scores first. It works in two big parts : first, it chooses a node according to its energy.
        This node changes its cluster for one of its neighbors clusters. Then, the global energy is updated.

         Parameters
         ----------
        G : PointCloudGraph
        exports : Boolean, if True : the function exports the graph of the evolution of the global energy, the quotient
        graph with energy on each node, two point clouds, one with the energy of each point, the other with the clusters
        number_of_iteration : int
        choice_of_node_to_change : the method used to select a node which is going to change cluster options are
        'max_energy', 'random_proba_energy', 'max_energy_and_select'
        formulae : 'old' or 'improved' should be the same as the init function

         Returns
         -------
         Nothing

        """


        # nombre d'itérations
        iter = number_of_iteration
        # Liste contenant l'énergie globale du graph
        evol_energy = [self.global_topological_energy]

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
                total_energy = self.global_topological_energy
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
                        #detect_rep = []
                        detect_rep.append(node_to_change)
                    if detect_rep.count(node_to_change) != 0:
                        detect_rep.append(node_to_change)
                if ban_list.count(node_to_change) != 0:
                    sort_energy_per_node = {k: v for k, v in sorted(energy_per_node.items(), key=lambda item: item[1], reverse=True)}
                    for c in sort_energy_per_node:
                        #print(c)
                        if ban_list.count(c) == 0:
                            node_to_change = c
                            if detect_rep.count(node_to_change) == 0:
                                #detect_rep = []
                                detect_rep.append(node_to_change)
                            else:
                                detect_rep.append(node_to_change)
                            break
                if detect_rep.count(node_to_change) >= G.nearest_neighbors * 2:
                    ban_list.append(node_to_change)
                    detect_rep = []


            #print()
            #print(i)
            #print(ban_list)
            #print(node_to_change)
            #print(G.nodes[node_to_change]['number_of_adj_labels'])
            #print(G.nodes[node_to_change]['quotient_graph_node'])

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

            self.update_quotient_graph_attributes_when_node_change_cluster(old_cluster, new_cluster, node_to_change, G)

            if formulae == 'old':
                # update of energy for the node changed
                previous_energy = G.nodes[node_to_change]['number_of_adj_labels']
                G.nodes[node_to_change]['number_of_adj_labels'] = new_energy
                #print(new_energy)
                self.global_topological_energy += (new_energy - previous_energy)
                u = G.nodes[node_to_change]['quotient_graph_node']
                self.nodes[u]['topological_energy'] += new_energy
                self.nodes[old_cluster]['topological_energy'] -= previous_energy
                #print(self.nodes[u]['topological_energy'])
                # update of energy for the neighbors
                for n in G[node_to_change]:
                    previous_energy = G.nodes[n]['number_of_adj_labels']
                    G.nodes[n]['number_of_adj_labels'] = 0
                    for v in G[n]:
                        number_of_neighb = len([n for n in G[v]])
                        if G.nodes[n]['quotient_graph_node'] != G.nodes[v]['quotient_graph_node']:
                            G.nodes[n]['number_of_adj_labels'] += 1 / number_of_neighb
                    self.global_topological_energy += (G.nodes[n]['number_of_adj_labels'] - previous_energy)
                    u = G.nodes[n]['quotient_graph_node']
                    self.nodes[u]['topological_energy'] += (G.nodes[n]['number_of_adj_labels'] - previous_energy)

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
                    G.nodes[node_to_change]['number_of_adj_labels'] = number_diff / (number_diff + (number_of_clusters-1)*number_same)

                new_energy = G.nodes[node_to_change]['number_of_adj_labels']
                self.global_topological_energy += (new_energy - previous_energy)
                self.nodes[new_cluster]['topological_energy'] += new_energy
                self.nodes[old_cluster]['topological_energy'] -= previous_energy

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
                        G.nodes[n]['number_of_adj_labels'] = number_diff / (number_diff + (number_of_clusters - 1) * number_same)
                    new_energy = G.nodes[n]['number_of_adj_labels']
                    self.global_topological_energy += (new_energy - previous_energy)
                    u = G.nodes[n]['quotient_graph_node']
                    self.nodes[u]['topological_energy'] += (new_energy - previous_energy)


            # update list containing all the differents stages of energy obtained
            evol_energy.append(self.global_topological_energy)

        self.delete_empty_edges_and_nodes()
        self.point_cloud_graph = G
        if exports:
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



            display_and_export_quotient_graph_matplotlib(self, node_sizes=20, filename="quotient_graph_matplotlib_energy_final",
                                                         data_on_nodes='topological_energy')
            export_some_graph_attributes_on_point_cloud(G, graph_attribute='number_of_adj_labels',
                                                        filename='graph_attribute_energy_final.txt')

            export_some_graph_attributes_on_point_cloud(G, graph_attribute='quotient_graph_node',
                                                        filename='graph_attribute_quotient_graph_node_final.txt')


    def update_quotient_graph(self, G):
        labels_qg = [k for k in dict(G.nodes(data='quotient_graph_node')).values()]
        labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]
        self.build_QuotientGraph_from_PointCloudGraph(G, labels_qg_re)

    def rebuild_quotient_graph(self, G, filename="pcd_attribute.txt"):
        self.clear()
        labels_qg = [k for k in dict(G.nodes(data='quotient_graph_node')).values()]
        labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]
        self.build_QuotientGraph_from_PointCloudGraph(G, labels_qg_re, filename=filename)


    def segment_a_node_using_attribute(self, quotient_node_to_work, G):
        nw = quotient_node_to_work

        # make list of nodes inside the quotient graph node to work with
        list_of_nodes = [x for x,y in G.nodes(data=True) if y['quotient_graph_node'] == nw]

        # Use of a function to segment
        X = np.zeros((len(list_of_nodes), 3))

        for i in range(len(list_of_nodes)):
            X[i] = G.nodes[list_of_nodes[i]]['direction_gradient']

        #clustering = skc.KMeans(n_clusters=2, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(X)

        clustering = skc.OPTICS(min_samples=100).fit(X)

        # Instead of having 0 or 1, this gives a number not already used in the quotient graph to name the nodes
        #clustering_labels = kmeans.labels_[:, np.newaxis] + max(self.nodes) + 1

        clustering_labels = clustering.labels_[:, np.newaxis] + max(self.nodes) + 1

        # Integration of the new labels in the quotient graph and updates

        for i in range(len(list_of_nodes)):
            G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]

    def segment_several_nodes_using_attribute(self, list_quotient_node_to_work=[], algo='OPTICS',
                                              para_clustering_meth=100, attribute='direction_gradient'):
        lw = list_quotient_node_to_work
        G = self.point_cloud_graph
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
            clustering = skc.KMeans(n_clusters=para_clustering_meth, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(X)

        # Selection of max value for 'quotient_graph_node' in G
        q_g = nx.get_node_attributes(G, 'quotient_graph_node')
        q_g_values = list(q_g.values())

        # Instead of having 0 or 1, this gives a number not already used in the quotient graph to name the nodes
        clustering_labels = clustering.labels_[:, np.newaxis] + max(q_g_values) + 2

        # Integration of the new labels in the quotient graph and updates
        for i in range(len(list_of_nodes)):
            G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]

        self.point_cloud_graph = G

    def compute_local_descriptors(self, G, method='each_point', data='coords'):
        # compute the descriptor for all the point in a quotient graph node
        if method == 'all_qg_cluster' and data == 'coords':
            for qnode in self:
                list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
                mat = np.zeros((len(list_of_nodes_in_qnode), 3))
                i = 0
                for n in list_of_nodes_in_qnode:
                    mat[i] = G.nodes_coords[n]
                    i += 1
                covmat = np.cov(np.transpose(mat))

                eigenval, eigenvec = np.linalg.eigh(covmat)

                self.nodes[qnode]['planarity'] = (eigenval[1] - eigenval[0]) / eigenval[2]
                self.nodes[qnode]['linearity'] = (eigenval[2] - eigenval[1]) / eigenval[2]
                self.nodes[qnode]['scattering'] = eigenval[0] / eigenval[2]
                self.nodes[qnode]['curvature_eig'] = eigenval[0] / (eigenval[0] + eigenval[2] + eigenval[1])
                self.nodes[qnode]['lambda1'] = eigenval[2]
                self.nodes[qnode]['lambda2'] = eigenval[1]
                self.nodes[qnode]['lambda3'] = eigenval[0]



        if method == 'each_point' and data == 'coords':
            # compute each decriptor for each point of the point cloud and its neighborhood
            for p in G:
                mat = np.zeros((len(G[p]), 3))
                i = 0
                for n in G[p]:
                    mat[i] = G.nodes_coords[n]
                    i += 1
                covmat = np.cov(np.transpose(mat))

                eigenval, eigenvec = np.linalg.eigh(covmat)
                G.nodes[p]['planarity'] = (eigenval[1] - eigenval[0]) / eigenval[2]
                G.nodes[p]['linearity'] = (eigenval[2] - eigenval[1]) / eigenval[2]
                G.nodes[p]['scattering'] = eigenval[0] / eigenval[2]
                G.nodes[p]['curvature_eig'] = eigenval[0] / (eigenval[0] + eigenval[2] + eigenval[1])

            # compute mean value for the qnode
            for qnode in self:
                list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
                self.nodes[qnode]['planarity'] = 0
                self.nodes[qnode]['linearity'] = 0
                self.nodes[qnode]['scattering'] = 0
                self.nodes[qnode]['curvature_eig'] = 0

                for n in list_of_nodes_in_qnode:
                    self.nodes[qnode]['planarity'] += G.nodes[n]['planarity']
                    self.nodes[qnode]['linearity'] += G.nodes[n]['linearity']
                    self.nodes[qnode]['scattering'] += G.nodes[n]['scattering']
                    self.nodes[qnode]['curvature_eig'] += G.nodes[n]['curvature_eig']
                self.nodes[qnode]['planarity'] /= len(list_of_nodes_in_qnode)
                self.nodes[qnode]['linearity'] /= len(list_of_nodes_in_qnode)
                self.nodes[qnode]['scattering'] /= len(list_of_nodes_in_qnode)
                self.nodes[qnode]['curvature_eig'] /= len(list_of_nodes_in_qnode)

        if method == 'all_qg_cluster' and data == 'gradient_vector_fiedler':
            for qnode in self:
                list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
                mat = np.zeros((len(list_of_nodes_in_qnode), 3))
                i = 0
                for n in list_of_nodes_in_qnode:
                    mat[i] = G.nodes[n]['direction_gradient']
                    i += 1
                covmat = np.cov(np.transpose(mat))

                eigenval, eigenvec = np.linalg.eigh(covmat)

                self.nodes[qnode]['max_eigenval'] = eigenval[2]
                self.nodes[qnode]['planarity'] = (eigenval[1] - eigenval[0]) / eigenval[2]
                self.nodes[qnode]['linearity'] = (eigenval[2] - eigenval[1]) / eigenval[2]
                self.nodes[qnode]['scattering'] = eigenval[0] / eigenval[2]
                self.nodes[qnode]['curvature_eig'] = eigenval[0] / (eigenval[0] + eigenval[2] + eigenval[1])



    def compute_silhouette(self, method='topological', data='direction_gradient_vector_fiedler'):

        G = self.point_cloud_graph

        label = []
        for node in G:
            label.append(G.nodes[node]['quotient_graph_node'])


        if method == 'all_qg_cluster':
            # this method is based on the classical way to compute silhouette coefficients in the sci-kit learn module

            if data == 'direction_gradient_vector_fiedler':
                sil = sk.metrics.silhouette_samples(G.direction_gradient_on_Fiedler_scaled, label, metric='euclidean')
            elif data == 'norm_gradient_vector_fiedler':
                sil = sk.metrics.silhouette_samples(G.gradient_on_Fiedler, label, metric='euclidean')



        if method == 'topological':
            import scipy.ndimage as nd
            from sklearn.metrics import pairwise_distances_chunked
            # this method aims at comparing the cluster A of the point considered with the adjacent clusters of A.
            X = []

            if data == 'direction_gradient_vector_fiedler':
                X = G.direction_gradient_on_Fiedler_scaled
            elif data == 'norm_gradient_vector_fiedler':
                X = G.gradient_on_Fiedler

            labels = np.array(label)

            label_list = np.array(self.nodes)

            label_adjacency = nx.adjacency_matrix(self).todense().astype('float64')
            label_adjacency -= 1 * np.eye(len(label_adjacency)).astype('float64')

            label_index = np.array([np.where(label_list == l)[0][0] for l in labels])

            def mean_by_label(D_chunk, start):
                label_mean = []
                chunk_labels = labels[start:start + len(D_chunk)]
                for line, label in zip(D_chunk, chunk_labels):
                    label_sum = nd.sum(line, labels, index=label_list)
                    label_count = nd.sum(np.ones_like(labels), labels, index=label_list)
                    label_count -= label_list == label
                    label_mean += [label_sum / label_count]
                return np.array(label_mean)

            gen = pairwise_distances_chunked(X, reduce_func=mean_by_label)

            start = 0
            silhouette = []
            for label_mean_dissimilarity in gen:
                chunk_label_index = label_index[start:start + len(label_mean_dissimilarity)]

                adjacent_label_mean_dissimilarity = np.copy(label_mean_dissimilarity)
                adjacent_label_mean_dissimilarity[label_adjacency[chunk_label_index] != 1] = np.nan

                self_label_mean_dissimilarity = np.copy(label_mean_dissimilarity)
                self_label_mean_dissimilarity[label_adjacency[chunk_label_index] != -1] = np.nan

                intra = np.nanmin(self_label_mean_dissimilarity, axis=1)
                inter = np.nanmin(adjacent_label_mean_dissimilarity, axis=1)

                silhouette += list((inter - intra) / np.maximum(inter, intra))
                start += len(label_mean_dissimilarity)

            sil = np.array(silhouette)

        i = 0
        for node in G:
            G.nodes[node]['silhouette'] = sil[i]
            i += 1
        for qnode in self:
            list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
            self.nodes[qnode]['silhouette'] = 0
            for n in list_of_nodes_in_qnode:
                self.nodes[qnode]['silhouette'] += G.nodes[n]['silhouette']
            self.nodes[qnode]['silhouette'] /= len(list_of_nodes_in_qnode)


    def define_leaves_by_topo(self):
        G = self.point_cloud_graph
        # Put the "end" nodes in a list
        list_leaves = [x for x in self.nodes() if self.degree(x) == 1]


        # Pick the node with the most average norm gradient (most likely to be part of the stem)
        max_norm = 0
        max_norm_node = -1
        for l in list_leaves:
            self.nodes[l]['norm_gradient_mean'] = 0
            list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l]
            for n in list_of_nodes_in_qnode:
                self.nodes[l]['norm_gradient_mean'] += G.nodes[n]['norm_gradient']
            self.nodes[l]['norm_gradient_mean'] /= len(list_of_nodes_in_qnode)
            if self.nodes[l]['norm_gradient_mean'] > max_norm:
                max_norm = self.nodes[l]['norm_gradient_mean']
                max_norm_node = l

        # Finally I chose randomly the root among the leaves but I could choose the max_norm_node
        root = np.random.choice(list_leaves)
        # cluster again the branches by following a path
        #to_cluster = []
        #for n in self:
        #    if n not in list_leaves:
        #       to_cluster.append(n)

        #self.segment_several_nodes_using_attribute(G=self.point_cloud_graph, list_quotient_node_to_work=to_cluster,
        #                                          algo='OPTICS', para_clustering_meth=100, attribute='direction_gradient')
        #self.delete_small_clusters(min_number_of_element_in_a_quotient_node=100)

    def define_semantic_classes(self):

        # leaves : one adjacency
        list_leaves = [x for x in self.nodes() if self.degree(x) == 1]

        # the node with the most number of neighborhood is the stem
        degree_sorted = sorted(self.degree, key=lambda x: x[1], reverse=True)
        stem = [degree_sorted[0][0]]

        # the rest is defined as branches
        for n in self.nodes:
            if n in list_leaves:
                self.nodes[n]['semantic_label'] = 'leaf'
            if n in stem:
                self.nodes[n]['semantic_label'] = 'stem'
            elif n not in list_leaves and n not in stem:
                self.nodes[n]['semantic_label'] = 'petiole'

    def define_semantic_scores(self, method='similarity_dist'):
        self.compute_local_descriptors(self.point_cloud_graph, method='all_qg_cluster', data='coords')
        self.compute_silhouette()
        degree_sorted = sorted(self.degree, key=lambda x: x[1], reverse=True)
        stem = [degree_sorted[0][0]]

        if method == 'condition_list':
            for n in self.nodes:
                score_vector_leaf = [self.nodes[n]['planarity'] > 0.5, self.nodes[n]['linearity'] < 0.5, self.degree(n) == 1]
                score_vector_petiole = [self.nodes[n]['planarity'] < 0.5, self.nodes[n]['linearity'] > 0.5, self.degree(n) == 2, self.nodes[n]['silhouette'] > 0.2]
                score_vector_stem = [self.nodes[n]['planarity'] < 0.5, self.nodes[n]['linearity'] > 0.5, self.degree(n) == degree_sorted[0][1]]
                self.nodes[n]['score_leaf'] = score_vector_leaf.count(True)/len(score_vector_leaf)
                self.nodes[n]['score_petiole'] = score_vector_petiole.count(True)/len(score_vector_petiole)
                self.nodes[n]['score_stem'] = score_vector_stem.count(True)/len(score_vector_stem)

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
            for n in QG.nodes:
                # score_vector_n = sk.preprocessing.normalize(np.asarray([QG.nodes[n]['planarity'], QG.nodes[n]['linearity']]).reshape(1,len(score_vector_leaf_ref)), norm='l2')
                score_vector_n = [QG.nodes[n]['planarity'], QG.nodes[n]['linearity'], QG.nodes[n]['silhouette']]
                print(score_vector_n)
                d1_leaf = sp.spatial.distance.euclidean(score_vector_leaf_ref, score_vector_n)
                # print(d1_leaf)
                d2_linea = sp.spatial.distance.euclidean(score_vector_linea_ref, score_vector_n)
                # print(d2_leaf)
                f1 = 1 - d1_leaf
                f2 = 1 - d2_linea
                QG.nodes[n]['score_leaf'] = identity(f1)
                QG.nodes[n]['score_petiole'] = identity(f2)

    def segment_each_cluster_by_optics_using_directions(self, leaves_out=True):
        # Put the "end" nodes in a list
        list_leaves = [x for x in self.nodes() if self.degree(x) == 1]
        for n in self:
            if leaves_out:
                if n not in list_leaves and self.nodes[n]['intra_class_node_number'] > 100:
                    self.segment_several_nodes_using_attribute(list_quotient_node_to_work=[n])
            elif leaves_out is False and self.nodes[n]['intra_class_node_number'] > 100:
                self.segment_several_nodes_using_attribute(list_quotient_node_to_work=[n])

        self.rebuild_quotient_graph(self.point_cloud_graph, filename='optics_seg_directions.txt')

    def segment_each_cluster_by_optics_using_norm(self, leaves_out=True):
        # Put the "end" nodes in a list
        list_leaves = [x for x in self.nodes() if self.degree(x) == 1]
        for n in self:
            if leaves_out:
                if n not in list_leaves and self.nodes[n]['intra_class_node_number'] > 100:
                    self.segment_several_nodes_using_attribute(list_quotient_node_to_work=[n], attribute='norm_gradient')
            elif leaves_out is False and self.nodes[n]['intra_class_node_number'] > 100:
                self.segment_several_nodes_using_attribute(list_quotient_node_to_work=[n], attribute = 'norm_gradient')

        self.rebuild_quotient_graph(self.point_cloud_graph, filename='optics_seg_norms.txt')

    def compute_direction_info(self, list_leaves):
        G = self.point_cloud_graph
        for l in self:
            self.nodes[l]['dir_gradient_mean'] = 0
            list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l]
            for n in list_of_nodes_in_qnode:
                self.nodes[l]['dir_gradient_mean'] += G.nodes[n]['direction_gradient']
            self.nodes[l]['dir_gradient_mean'] /= len(list_of_nodes_in_qnode)

        # take all the edges and compute the dot product between the two cluster
        for e in self.edges():
            v1 = self.nodes[e[0]]['dir_gradient_mean']
            v2 = self.nodes[e[1]]['dir_gradient_mean']
            if e[0] in list_leaves or e[1] in list_leaves:
                self.edges[e]['energy_dot_product'] = 4
            else:
                dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
                energy_dot_product = 1-dot
                self.edges[e]['energy_dot_product'] = energy_dot_product

    def opti_energy_dot_product(self, energy_to_stop=0.13, leaves_out=False, list_graph_node_to_work=[], export_iter=True):
        # take the edge of higher energy and fuse the two nodes involved, then rebuilt the graph, export and redo
        list_leaves = [x for x in self.nodes() if self.degree(x) == 1]
        energy_min = 0
        i = 0

        while energy_min < energy_to_stop:
            G = self.point_cloud_graph
            if not list_graph_node_to_work:
                # select the edge
                self.compute_direction_info(list_leaves)
                energy_per_edges = nx.get_edge_attributes(self, 'energy_dot_product')
                edge_to_delete = min(energy_per_edges.items(), key=operator.itemgetter(1))[0]
                print(edge_to_delete)
                print(energy_per_edges[edge_to_delete])
                energy_min = energy_per_edges[edge_to_delete]
                # make list of nodes inside the different quotient graph nodes to work with
                list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == edge_to_delete[1]]
                for j in range(len(list_of_nodes)):
                    G.nodes[list_of_nodes[j]]['quotient_graph_node'] = edge_to_delete[0]

            else:
                # retrieve the quotient graphe nodes that encapsulate the nodes from pointcloudgraph (list_graph_node_to_work)
                list_quotient_node_to_work = []
                for no in range(len(list_graph_node_to_work)):
                    list_quotient_node_to_work.append(G.nodes[list_graph_node_to_work[no]]['quotient_graph_node'])
                list_of_new_quotient_graph_nodes = list(set(list_quotient_node_to_work))

                # create a subgraph of the quotient graph with only the interesting nodes and compute energy on nodes
                S = nx.subgraph(self, list_of_new_quotient_graph_nodes)
                self.compute_direction_info(list_leaves=[])
                # selection of the edge that bears the less energy
                energy_per_edges = nx.get_edge_attributes(S, 'energy_dot_product')
                edge_to_delete = min(energy_per_edges.items(), key=operator.itemgetter(1))[0]
                print(edge_to_delete)
                print(energy_per_edges[edge_to_delete])
                energy_min = energy_per_edges[edge_to_delete]
                # one cluster is merged into the other one, change of labels on the pointcloudgraph label
                list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == edge_to_delete[1]]
                for j in range(len(list_of_nodes)):
                    G.nodes[list_of_nodes[j]]['quotient_graph_node'] = edge_to_delete[0]


            if leaves_out:
                list_one_point_per_leaf = []
                for l in list_leaves:
                    list_one_point_per_leaf.append([x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l][0])
                list_leaves = []
                for p in list_one_point_per_leaf:
                    list_leaves.append(G.nodes[p]['quotient_graph_node'])

            self.point_cloud_graph = G
            #self.rebuild_quotient_graph(G=self.point_cloud_graph,filename='graph_attribute_quotient_graph_' + str(i) + '.txt')
            if export_iter:
                self.rebuild_quotient_graph(G=self.point_cloud_graph,
                                            filename='graph_attribute_quotient_graph_' + str(i) + '.txt')
                display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
                                                             filename="quotient_graph_matplotlib_QG_intra_class_number_" + str(i),
                                                             data_on_nodes='intra_class_node_number')
            i += 1


        """
        for l in list_leaves:
            if l != root:
                path = nx.bidirectional_shortest_path(self, root, l)
                path.pop()
                path.pop(0)
                le = len(path)
                if len(path) >= 2:                    for i in range(le-1):
                        self.segment_several_nodes_using_attribute(list_quotient_node_to_work=[path[i], path[i+1]])
                    G = self.point_cloud_graph
                    self.rebuild_quotient_graph(G)
                    
                    
        """
    def oversegment_part(self, list_quotient_node_to_work=[], average_size_cluster=20):
        lw = list_quotient_node_to_work
        G = self.point_cloud_graph
        # make list of nodes inside the different quotient graph nodes to work with
        list_of_nodes = []
        for qnode in lw:
            list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
            list_of_nodes += list_of_nodes_each

        # get all the x,y,z coordinates in an np.array
        Xcoord = np.zeros((len(list_of_nodes), 3))
        for i in range(len(list_of_nodes)):
            Xcoord[i] = G.nodes[list_of_nodes[i]]['pos']

        # determine a number of cluster we would like to have
        number_of_points = len(list_of_nodes)
        number_of_clusters = number_of_points/average_size_cluster

        # clustering via Kmeans to obtain new labels
        clustering = skc.KMeans(n_clusters=int(number_of_clusters), init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(Xcoord)
        clustering_labels = clustering.labels_[:, np.newaxis] + max(self.nodes) + 1

        # Integration of the new labels in the quotient graph and updates
        for i in range(len(list_of_nodes)):
            G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]
        self.rebuild_quotient_graph(G)

        # return of the list of nodes in pointcloudgraph to keep working with them
        list_of_nodes_pointcloudgraph = list_of_nodes
        return list_of_nodes_pointcloudgraph



def collect_quotient_graph_nodes_from_pointcloudpoints(quotient_graph, list_of_points=[]):
    G = quotient_graph.point_cloud_graph
    list = []
    for node in list_of_points:
        list.pop(G.nodes[node]['quotient_graph_node'])

    list_of_clusters = list(set(list))

    return list_of_clusters


def export_some_graph_attributes_on_point_cloud(G, graph_attribute='quotient_graph_node', filename='graph_attribute.txt'):
    new_classif = np.asarray(list((dict(G.nodes(data=graph_attribute)).values())))
    new_classif = new_classif[:, np.newaxis]
    kpcg.export_anything_on_point_cloud(G, attribute=new_classif, filename=filename)


def draw_quotientgraph_cellcomplex(pcd, QG, G, color_attribute='quotient_graph_node'):

    if type(pcd) is open3d.geometry.PointCloud:
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

    topomesh = edge_topomesh(np.array([e for e in QG.edges if e[0]!=e[1]]), dict(zip(np.asarray([n for n in QG.nodes]), pcdtab)))

    if color_attribute == 'quotient_graph_node':
        topomesh.update_wisp_property('quotient_graph_node', 0, dict(zip([n for n in QG.nodes], [n for n in QG.nodes])))
    else:
        topomesh.update_wisp_property(color_attribute, 0,
                                     dict(zip([n for n in QG.nodes], [QG.nodes[n][color_attribute] for n in QG.nodes])))

    compute_topomesh_property(topomesh, 'length', 1)

    edge_actor = VtkActorTopomesh()
    edge_actor.set_topomesh(topomesh, 1, property_name='length')
    edge_actor.line_glyph = 'tube'
    edge_actor.glyph_scale = 0.33
    edge_actor.update(colormap="gray")

    vertex_actor = VtkActorTopomesh(topomesh, 0, property_name=color_attribute)
    # vertex_actor.point_glyph = 'point'
    vertex_actor.point_glyph = 'sphere'
    vertex_actor.glyph_scale = 2
    vertex_actor.update(colormap="jet")

    graph_topomesh = vertex_topomesh(dict(zip([n for n in G.nodes], [G.nodes[n]['pos'] for n in G.nodes])))
    graph_topomesh.update_wisp_property(color_attribute, 0, dict(
        zip([n for n in G.nodes], [G.nodes[n][color_attribute] for n in G.nodes])))
    point_cloud_actor = VtkActorTopomesh(graph_topomesh, 0, property_name=color_attribute)
    point_cloud_actor.point_glyph = 'point'
    point_cloud_actor.update(colormap="jet")

    vtk_display_actors([vertex_actor.actor, edge_actor.actor, point_cloud_actor.actor], background=(0.9, 0.9, 0.9))


#3d
def draw_quotientgraph_matplotlib_3D(nodes_coords_moy, QG):
    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        for i in range(nodes_coords_moy.shape[0]):
            xi = nodes_coords_moy[i, 0]
            yi = nodes_coords_moy[i, 1]
            zi = nodes_coords_moy[i, 2]

            ax.scatter(xi, yi, zi, s=10**2, edgecolors='k', alpha=0.7)

        for i, j in enumerate(QG.edges()):
            corresp = dict(zip(QG.nodes, range(len(QG.nodes))))
            x = np.array((nodes_coords_moy[corresp[j[0]],0], nodes_coords_moy[corresp[j[1]],0]))
            y = np.array((nodes_coords_moy[corresp[j[0]],1], nodes_coords_moy[corresp[j[1]],1]))
            z = np.array((nodes_coords_moy[corresp[j[0]],2], nodes_coords_moy[corresp[j[1]],2]))
            ax.plot(x, y, z, c='black', alpha=0.5)
        ax.view_init(30, angle)
        ax.set_axis_off()
        plt.show()


def display_and_export_quotient_graph_matplotlib(quotient_graph, node_sizes=20, filename="quotient_graph_matplotlib", data_on_nodes='intra_class_node_number', data=True, attributekmeans4clusters = False):

    figure = plt.figure(0)
    figure.clf()
    graph_layout = nx.kamada_kawai_layout(quotient_graph)
    colormap = 'jet'

    if attributekmeans4clusters and data:
        labels_from_attributes = dict(quotient_graph.nodes(data=data_on_nodes))
        # Rounding the data to allow an easy display
        for dict_value in labels_from_attributes:
            labels_from_attributes[dict_value] = round(labels_from_attributes[dict_value], 2)
        node_color_from_attribute = dict(quotient_graph.nodes(data='seed_colors')).values()
        node_color = [quotient_graph.nodes[i]['kmeans_labels'] / 4 for i in quotient_graph.nodes()]
        nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=True,
                                          node_size=node_sizes,
                                          node_color=node_color_from_attribute,
                                          labels=labels_from_attributes,
                                          cmap=plt.get_cmap(colormap))

    elif attributekmeans4clusters is False and data is False:
        nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=False,
                                          node_size=node_sizes,
                                          node_color="r",
                                          cmap=plt.get_cmap(colormap))

    elif attributekmeans4clusters is False and data:
        labels_from_attributes = dict(quotient_graph.nodes(data=data_on_nodes))
        # Rounding the data to allow an easy display
        for dict_value in labels_from_attributes:
            if data_on_nodes != 'semantic_label':
                labels_from_attributes[dict_value] = round(labels_from_attributes[dict_value], 2)
        nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=True,
                                          node_size=node_sizes,
                                          node_color="r",
                                          labels=labels_from_attributes,
                                          cmap=plt.get_cmap(colormap))


    #nx.drawing.nx_pylab.draw_networkx_edge_labels(quotient_graph, pos=graph_layout, font_size=20, font_family="sans-sherif")

    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du graphe quotient matplotlib")

######### Main

if __name__ == '__main__':
    pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/Older_Cheno.ply", format='ply')
    r = 18
    G = kpcg.PointCloudGraph()
    G.PointCloudGraph_init_with_pcd(point_cloud=pcd, method='knn', nearest_neighbors=r)
    print(nx.is_connected(G))
    if nx.is_connected(G) is False:
        largest_cc = max(nx.connected_components(G), key=len)
        # creating the new pcd point clouds
        coords = np.zeros((len(largest_cc), 3))
        i = 0
        for node in largest_cc:
            coords[i, :] = G.nodes[node]['pos']
            i += 1
        np.savetxt('New_pcd_connected.txt', coords, delimiter=' ', fmt='%f')
        pcd2 = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Src/spectral_clustering/New_pcd_connected.txt", format='xyz')
        r = 18
        G = kpcg.PointCloudGraph()
        G.PointCloudGraph_init_with_pcd(point_cloud=pcd2, method='knn', nearest_neighbors=r)


    G.compute_graph_eigenvectors()
    G.compute_gradient_of_Fiedler_vector(method='by_Fiedler_weight')
    G.clustering_by_kmeans_in_four_clusters_using_gradient_norm(export_in_labeled_point_cloud=True)
    #G.clustering_by_fiedler_and_optics(criteria=np.multiply(G.direction_gradient_on_Fiedler_scaled, G.gradient_on_Fiedler))

    QG = QuotientGraph()
    QG.build_QuotientGraph_from_PointCloudGraph(G, G.kmeans_labels_gradient)

    QG.init_topo_scores(QG.point_cloud_graph, exports=True, formulae='improved')

    QG.optimization_topo_scores(G=QG.point_cloud_graph, exports=True, number_of_iteration=10000,
                                choice_of_node_to_change='max_energy', formulae='improved')

    QG.rebuild_quotient_graph(QG.point_cloud_graph)

    QG.segment_each_cluster_by_optics_using_directions()

    QG.init_topo_scores(G=QG.point_cloud_graph, exports=True, formulae='improved')
    QG.optimization_topo_scores(G=QG.point_cloud_graph, exports=True, number_of_iteration=1000,
                                choice_of_node_to_change='max_energy', formulae='improved')
    QG.rebuild_quotient_graph(QG.point_cloud_graph)

    list_of_nodes_to_work = QG.oversegment_part(list_quotient_node_to_work=[10], average_size_cluster=50)
    QG.compute_direction_info(list_leaves=[])
    QG.opti_energy_dot_product(energy_to_stop=0.29, leaves_out=False, list_graph_node_to_work=list_of_nodes_to_work)



    """
    
    QG.opti_energy_dot_product(iter=8)
    QG.rebuild_quotient_graph(QG.point_cloud_graph)

    QG.compute_local_descriptors(QG.point_cloud_graph, method='all_qg_cluster', data='coords')


    

    labels_qg = [k for k in dict(G.nodes(data = 'quotient_graph_node')).values()]
    labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]


    display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
                                                 filename="quotient_graph_matplotlib_QG_intra_class_number2",
                                                 data_on_nodes='intra_class_node_number')
    export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph, graph_attribute='quotient_graph_node',
                                                filename='graph_attribute_quotient_graph_node_very_end.txt')

    QG.define_semantic_classes()
    for (u,v) in QG.edges:
        if QG.nodes[u]['semantic_label'] == QG.nodes[v]['semantic_label']:
            QG.edges[u, v]['semantic_weight'] = 50
        else:
            QG.edges[u, v]['semantic_weight'] = 1

    ac = nx.minimum_spanning_tree(QG, algorithm='kruskal', weight='semantic_weight')
    display_and_export_quotient_graph_matplotlib(quotient_graph=ac, node_sizes=20,
                                                 filename="ac", data_on_nodes='semantic_label', data=True, attributekmeans4clusters=False)


    for (u, v) in QG.edges:
        QG.edges[u, v]['inverse_inter_class_edge_weight'] = 1.0 / QG.edges[u, v]['inter_class_edge_weight']
    mst = nx.minimum_spanning_tree(QG, algorithm='kruskal', weight='inverse_inter_class_edge_weight')
    display_and_export_quotient_graph_matplotlib(quotient_graph=mst, node_sizes=20,
                                                 filename="mst", data=False, attributekmeans4clusters=False)

    QG.compute_local_descriptors(QG.point_cloud_graph, method='all_qg_cluster', data='coords')
    tab = np.zeros((len(QG), 3))
    i = 0
    for n in QG.nodes:
        tab[i, 0] = QG.nodes[n]['planarity']
        tab[i, 1] = QG.nodes[n]['linearity']
        tab[i, 2] = QG.nodes[n]['intra_class_node_number']/len(pcd.points)
        i += 1

    figure = plt.figure(1)
    figure.clf()
    figure.gca().set_title("Evolution_of_energy")
    plt.autoscale(enable=True, axis='both', tight=None)
    figure.gca().scatter(tab[:, 0], tab[:, 1], color='blue', s=(tab[:, 2]*10000))
    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig('linearity_planarity')
    print("linearity_planarity")

    labels_from_qg = np.zeros((len(QG.point_cloud_graph), 4))
    i = 0
    G = QG.point_cloud_graph
    for n in G.nodes:
        labels_from_qg[i, 0:3] = G.nodes[n]['pos']
        labels_from_qg[i, 3] = QG.nodes[G.nodes[n]['quotient_graph_node']]['silhouette']
        i += 1

    np.savetxt('pcd_silhouette.txt', labels_from_qg, delimiter=",")

    #display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
    #                                             filename="QG_leaf", data_on_nodes='score_leaf', data=True,
    #                                             attributekmeans4clusters=False)

    sil = []
    

    nw = 12
    G = QG.point_cloud_graph

    # make list of nodes inside the quotient graph node to work with
    list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == nw]


    # Use of a function to segment
    Xcoord = np.zeros((len(list_of_nodes), 3))

    for i in range(len(list_of_nodes)):
        Xcoord[i] = G.nodes[list_of_nodes[i]]['pos']

    H = nx.subgraph(G, list_of_nodes)
    H.compute_graph_eigenvectors(is_sparse=True, k=50, smallest_first=True, laplacian_type='classical')

    figure = plt.figure(1)
    figure.clf()
    figure.gca().set_title("eigenvalues")
    plt.autoscale(enable=True, axis='both', tight=None)
    figure.gca().scatter(range(len(H.keigenval)), H.keigenval, color='blue')
    figure.savefig('valeurs_propres')



    clustering = skc.SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=18).fit(Xcoord)


    #clustering = skc.KMeans(n_clusters=2, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(X2)
    clustering_labels = clustering.labels_[:, np.newaxis] + max(QG.nodes) + 1

    

    # Integration of the new labels in the quotient graph and updates
    for i in range(len(list_of_nodes)):
        G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]
    QG.point_cloud_graph = G

    QG.rebuild_quotient_graph(G=QG.point_cloud_graph)
    QG.compute_silhouette()
    labels_from_qg = np.zeros((len(QG.point_cloud_graph), 4))
    i = 0
    G = QG.point_cloud_graph
    for n in G.nodes:
        labels_from_qg[i, 0:3] = G.nodes[n]['pos']
        labels_from_qg[i, 3] = QG.nodes[G.nodes[n]['quotient_graph_node']]['silhouette']
        i += 1
    np.savetxt('pcd_silhouette.txt', labels_from_qg, delimiter=",")


    labels_qg = [k for k in dict(G.nodes(data='quotient_graph_node')).values()]
    labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]

    export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph, graph_attribute='quotient_graph_node',
                                                filename='graph_attribute_quotient_graph_node_very_end.txt')

    sil.append(sk.metrics.silhouette_score(X1, clustering_labels))

    #sil = []
    #for rep in range(8):
    #    clustering = skc.KMeans(n_clusters=rep+2, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(X)
    #affinity = 1 - sp.spatial.distance_matrix(X, X)

    #clustering = skc.affinity_propagation(S=affinity)
    #clustering = skc.AffinityPropagation(damping=0.95).fit(X)

    # Instead of having 0 or 1, this gives a number not already used in the quotient graph to name the nodes
    # clustering_labels = kmeans.labels_[:, np.newaxis] + max(self.nodes) + 1

    #    clustering_labels = clustering.labels_[:, np.newaxis] + max(QG.nodes) + 1

    #    sil.append(sk.metrics.silhouette_score(X, clustering_labels))

    figure = plt.figure(1)
    figure.clf()
    figure.gca().set_title("eigenvalues")
    plt.autoscale(enable=True, axis='both', tight=None)
    figure.gca().scatter(range(len(H.keigenval)), H.keigenval, color='blue')
    figure.savefig('valeurs_propres')

    QG.compute_quotientgraph_nodes_coordinates(G)

    draw_quotientgraph_cellcomplex(pcd=QG.nodes_coordinates, QG=QG, G=QG.point_cloud_graph)
    """
    """
    # Integration of the new labels in the quotient graph and updates
    clustering = skc.KMeans(n_clusters=2, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(X)
    clustering_labels = clustering.labels_[:, np.newaxis] + max(QG.nodes) + 1
    for i in range(len(list_of_nodes)):
        G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]
    QG.point_cloud_graph = G

    labels_qg = [k for k in dict(G.nodes(data='quotient_graph_node')).values()]
    labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]

    export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph, graph_attribute='quotient_graph_node',
                                                filename='graph_attribute_quotient_graph_node_very_end.txt')
    """


"""
    #display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
    #                                             filename="quotient_graph_matplotlib_brut")
    #QG.delete_small_clusters()
    #display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
    #                                             filename="quotient_graph_matplotlib_without_small_clusters")
    export_some_graph_attributes_on_point_cloud(G)

    QG.compute_quotientgraph_nodes_coordinates(G)

    #draw_quotientgraph_cellcomplex(pcd=QG.nodes_coordinates, QG=QG, G=G, color_attribute='kmeans_labels')
    export_some_graph_attributes_on_point_cloud(G, graph_attribute='quotient_graph_node',
                                                filename='graph_attribute_quotient_graph_node_init.txt')

    QG.init_topo_scores(G)
    display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_init_number",
                                                 data_on_nodes='intra_class_node_number', attributekmeans4clusters=True)
    start = time.time()
    QG.optimization_topo_scores(G=G, exports=True, number_of_iteration=10000, choice_of_node_to_change='max_energy_and_select')
    end = time.time()
    timescore = end - start
    display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
                                                 filename="quotient_graph_matplotlib_end_number",
                                                 data_on_nodes='intra_class_node_number', attributekmeans4clusters=True)

    QG2 = QuotientGraph()
    labels_qg = [k for k in dict(G.nodes(data = 'quotient_graph_node')).values()]
    labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]
    start2 = time.time()
    QG2.build_QuotientGraph_from_PointCloudGraph(G, labels_qg_re)
    end2 = time.time()
    timebuild = end2 - start2
    QG.delete_empty_edges_and_nodes()

    #draw_quotientgraph_cellcomplex(pcd=QG.nodes_coordinates, QG=QG2, G=G, color_attribute='kmeans_labels')

    export_some_graph_attributes_on_point_cloud(G, graph_attribute='quotient_graph_node',
                                                filename='graph_attribute_quotient_graph_node_end.txt')



    display_and_export_quotient_graph_matplotlib(quotient_graph=QG2, node_sizes=20,
                                                 filename="quotient_graph_matplotlib_QG2_intra_class_number",
                                                 data_on_nodes='intra_class_node_number')
    display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
                                                 filename="quotient_graph_matplotlib_QG_intra_class_number",
                                                 data_on_nodes='intra_class_node_number')

"""



