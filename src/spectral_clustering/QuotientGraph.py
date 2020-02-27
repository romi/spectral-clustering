########### Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsp
import scipy as sp
import sklearn.cluster as skc
import sklearn as sk
import spectral_clustering.similarity_graph as sgk
import open3d as open3d
import spectral_clustering.PointCloudGraph as kpcg

########### Définition classe

class QuotientGraph(nx.Graph):

    def __init__(self):
        super().__init__()
        self.seed_colors = None
        self.graph_labels_dict = None
        self.label_count = None

    def build_QuotientGraph_from_PointCloudGraph(self, G):

        kmeans_labels = G.kmeans_labels_gradient
        connected_component_labels = np.zeros(kmeans_labels.shape, dtype=int)
        current_cc_size = 0
        connected_component_size = []
        seed_kmeans_labels = []
        cluster_colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
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
                seed_colors.append(cluster_colors[kmeans_labels[seed][0]])
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

        nx.set_node_attributes(self, dict(
            zip(np.asarray(self.nodes()), np.transpose(np.asarray(seed_colors)))), 'seed_colors')

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

def export_some_graph_attributes_on_point_cloud(graph, graph_attribute='quotient_graph_node', filename='graph_attribute.txt'):
    new_classif = np.asarray(list((dict(G.nodes(data=graph_attribute)).values())))
    new_classif = new_classif[:, np.newaxis]
    kpcg.export_anything_on_point_cloud(G, attribute=new_classif, filename=filename)


def display_and_export_quotient_graph_matplotlib(quotient_graph_manual, node_sizes=20, filename="quotient_graph_matplotlib"):

    figure = plt.figure(0)
    figure.clf()
    graph_layout = nx.kamada_kawai_layout(quotient_graph_manual)
    colormap = 'jet'
    node_color_from_attribute = dict(QG.nodes(data='seed_colors')).values()
    node_color = [quotient_graph_manual.nodes[i]['kmeans_labels'] / 4 for i in quotient_graph_manual.nodes()]
    nx.drawing.nx_pylab.draw_networkx(quotient_graph_manual,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=True,
                                          node_size=node_sizes,
                                          node_color=node_color_from_attribute,
                                          labels=dict(QG.nodes(data='intra_class_node_number')),
                                          cmap=plt.get_cmap(colormap))

    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du graphe quotient matplotlib")

######### Main

if __name__ == '__main__':

    pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenopode_propre.ply")
    r = 18
    G = kpcg.PointCloudGraph(point_cloud=pcd, method='knn', nearest_neighbors=r)
    G.compute_graph_eigenvectors()
    G.compute_gradient_of_Fiedler_vector(method='simple')
    #G.clustering_by_fiedler_and_agglomerative(number_of_clusters=45, criteria=X)
    G.clustering_by_kmeans_in_four_clusters_using_gradient_norm(export_in_labeled_point_cloud=True)
    QG = QuotientGraph()
    QG.build_QuotientGraph_from_PointCloudGraph(G)

    display_and_export_quotient_graph_matplotlib(QG, node_sizes=20,
                                                 filename="quotient_graph_matplotlib_brut")
    QG.delete_small_clusters()
    display_and_export_quotient_graph_matplotlib(QG, node_sizes=20,
                                                 filename="quotient_graph_matplotlib_without_small_clusters")
    export_some_graph_attributes_on_point_cloud(G)











