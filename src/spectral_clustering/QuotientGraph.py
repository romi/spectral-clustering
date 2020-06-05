########### Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as spsp
import scipy as sp
import sklearn.cluster as skc
import sklearn as sk
import spectral_clustering.similarity_graph as sgk
import open3d as open3d
import spectral_clustering.PointCloudGraph as kpcg
import time

########### Définition classe

class QuotientGraph(nx.Graph):

    def __init__(self):
        super().__init__()
        self.seed_colors = None
        self.graph_labels_dict = None
        self.label_count = None
        self.nodes_coordinates = None
        self.global_topological_energy = None

    def build_QuotientGraph_from_PointCloudGraph(self, G, labels_from_cluster):

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
        #self.graph_labels_dict = dict(zip(np.asarray(self.nodes()), range(label_count)))
        kpcg.export_anything_on_point_cloud(G, attribute=connected_component_labels)


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

    def init_topo_scores(self, G, exports=True):
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
        for v in G.nodes:
            number_of_neighb = len([n for n in G[v]])
            for n in G[v]:
                if G.nodes[v]['quotient_graph_node'] != G.nodes[n]['quotient_graph_node']:
                    G.nodes[v]['number_of_adj_labels'] += 1
            G.nodes[v]['number_of_adj_labels'] /= number_of_neighb
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
        # update Quotient Graph attributes
        # Intra_class_node_number
        self.nodes[old_cluster]['intra_class_node_number'] += -1
        self.nodes[new_cluster]['intra_class_node_number'] += 1

        # intra_class_edge_weight, inter_class_edge_number
        for ng in G[node_to_change]:
            if old_cluster != G.nodes[ng]['quotient_graph_node'] and new_cluster == G.nodes[ng]['quotient_graph_node']:
                self.node[new_cluster]['intra_class_edge_weight'] += G.edges[ng, node_to_change]['weight']
                self.node[new_cluster]['intra_class_edge_number'] += 1
                self.edges[old_cluster, new_cluster]['inter_class_edge_weight'] += - G.edges[ng, node_to_change][
                    'weight']
                self.edges[old_cluster, new_cluster]['inter_class_edge_number'] += - 1

            if old_cluster == G.nodes[ng]['quotient_graph_node'] and new_cluster != G.nodes[ng]['quotient_graph_node']:
                self.node[old_cluster]['intra_class_edge_weight'] += -G.edges[ng, node_to_change]['weight']
                self.node[old_cluster]['intra_class_edge_number'] += -1
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

    def delete_empty_edges_and_nodes(self):
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



    def optimization_topo_scores(self, G, exports=True, number_of_iteration=1000, choice_of_node_to_change='max_energy'):
        # nombre d'itérations
        n = number_of_iteration
        # Liste contenant l'énergie globale du graph
        evol_energy = [self.global_topological_energy]

        # list to detect repetition in 'max_energy_and_select'
        detect_rep = []
        ban_list = []

        # Start loops for the number of iteration specified
        for i in range(n):

            # Choice of point to move from a cluster to another.

            if choice_of_node_to_change == 'max_energy':
                # Creation of a dictionary with the energy per node
                energy_per_node = nx.get_node_attributes(G, 'number_of_adj_labels')
                # Extraction of a random point to treat, use of "smart indexing"
                nodes = np.array(list(energy_per_node.keys()))
                node_energies = np.array(list(energy_per_node.values()))
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
                        print(c)
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


            print()
            print(i)
            print(ban_list)
            print(node_to_change)
            print(G.nodes[node_to_change]['number_of_adj_labels'])
            print(G.nodes[node_to_change]['quotient_graph_node'])

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


            # update of energy for the node changed
            previous_energy = G.nodes[node_to_change]['number_of_adj_labels']
            G.nodes[node_to_change]['number_of_adj_labels'] = new_energy
            print(new_energy)
            self.global_topological_energy += (new_energy - previous_energy)
            u = G.nodes[node_to_change]['quotient_graph_node']
            self.nodes[u]['topological_energy'] += new_energy
            self.nodes[old_cluster]['topological_energy'] -= previous_energy
            print(self.nodes[u]['topological_energy'])
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

            # update list containing all the differents stages of energy obtained
            evol_energy.append(self.global_topological_energy)

        self.delete_empty_edges_and_nodes()
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

            display_and_export_quotient_graph_matplotlib(QG, node_sizes=20, filename="quotient_graph_matplotlib_energy_final",
                                                         data_on_nodes='topological_energy')
            export_some_graph_attributes_on_point_cloud(G, graph_attribute='number_of_adj_labels',
                                                        filename='graph_attribute_energy_final.txt')

            export_some_graph_attributes_on_point_cloud(G, graph_attribute='quotient_graph_node',
                                                        filename='graph_attribute_quotient_graph_node_final.txt')


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


def display_and_export_quotient_graph_matplotlib(quotient_graph, node_sizes=20, filename="quotient_graph_matplotlib", data_on_nodes='intra_class_node_number', attributekmeans4clusters = False):

    figure = plt.figure(0)
    figure.clf()
    graph_layout = nx.kamada_kawai_layout(quotient_graph)
    colormap = 'jet'
    node_color_from_attribute = dict(quotient_graph.nodes(data='seed_colors')).values()
    node_color = [quotient_graph.nodes[i]['kmeans_labels'] / 4 for i in quotient_graph.nodes()]


    labels_from_attributes = dict(quotient_graph.nodes(data=data_on_nodes))
    # Rounding the data to allow an easy display
    for dict_value in labels_from_attributes:
            labels_from_attributes[dict_value] = round(labels_from_attributes[dict_value], 2)

    if attributekmeans4clusters:
        nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=True,
                                          node_size=node_sizes,
                                          node_color=node_color_from_attribute,
                                          labels=labels_from_attributes,
                                          cmap=plt.get_cmap(colormap))


    else:
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

    pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenopode_propre.ply", format='ply')
    r = 18
    G = kpcg.PointCloudGraph(point_cloud=pcd, method='knn', nearest_neighbors=r)
    G.compute_graph_eigenvectors()
    G.compute_gradient_of_Fiedler_vector(method='simple')
    G.clustering_by_kmeans_in_four_clusters_using_gradient_norm(export_in_labeled_point_cloud=True)
    QG = QuotientGraph()
    QG.build_QuotientGraph_from_PointCloudGraph(G, G.kmeans_labels_gradient)

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

    QG.optimization_topo_scores(G=G, exports=True, number_of_iteration=3000, choice_of_node_to_change='max_energy_and_select')
    display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
                                                 filename="quotient_graph_matplotlib_end_number",
                                                 data_on_nodes='intra_class_node_number', attributekmeans4clusters=True)

    QG2 = QuotientGraph()
    labels_qg = [k for k in dict(G.nodes(data = 'quotient_graph_node')).values()]
    labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]
    QG2.build_QuotientGraph_from_PointCloudGraph(G, labels_qg_re)
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





