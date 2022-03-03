########### Imports
import time
from collections import Counter

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse as spsp
import sklearn as sk
import pandas as pd

import spectral_clustering.point_cloud_graph as kpcg
from spectral_clustering.similarity_graph import *


########### Définition classe

class QuotientGraph(nx.Graph):

    def __init__(self):
        super().__init__()
        # self.seed_colors = None
        # self.graph_labels_dict = None
        # self.label_count = None
        self.point_cloud_graph = None
        self.nodes_coordinates = None

    def build_from_pointcloudgraph(self, G, labels_from_cluster, region_growing=True):
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
        filename : str
            File to export

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
        lablist = list(kmeans_labels[:])
        my_count = pd.Series(lablist).value_counts()
        # if len(my_count) == 4:
        #     #cluster_colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
        #     cluster_colors = ['glasbey_'+str(i) for i in range(len(my_count))]
        # else:
        #     jet = cm.get_cmap('jet', len(my_count))
        #     cluster_colors = jet(range(len(my_count)))
        #     list_labels = np.unique(lablist).tolist()
        seed_colors = []
        label_count = 0
        visited = np.zeros(kmeans_labels.shape, dtype=int)
        queue = []

        # Region growing algorithm to extract connected components made of the same k-means class.
        if region_growing:
            for i in range(kmeans_labels.shape[0]):

                # Find a seed to initialize the seed fill process.

                if visited[i] == 0:
                    seed = i
                    visited[i] = 1
                    queue.append(i)
                    seed_kmeans_labels.append(kmeans_labels[seed])
                    seed_colors.append('glasbey_'+ str(kmeans_labels[seed][0]%256))
                    # if len(my_count) == 4:
                    #     seed_colors.append(cluster_colors[kmeans_labels[seed][0]])
                    # else:
                    #     seed_colors.append(cluster_colors[list_labels.index(kmeans_labels[seed][0])][:])
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
        else:
            # To correct/finish, it doesn't work for now
            list_kmeans_labels = [kmeans_labels[i][0] for i in range(len(kmeans_labels))]
            label_count = len(list(set(list_kmeans_labels)))
            for i in range(kmeans_labels.shape[0]):
                seed_colors.append('glasbey_'+ str(kmeans_labels[i][0]%256))
                seed_kmeans_labels.append(kmeans_labels[i])

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

        # if len(my_count) == 4:
        #     nx.set_node_attributes(self, dict(
        #         zip(np.asarray(self.nodes()), np.transpose(np.asarray(seed_colors)))), 'seed_colors')
        # else:
        nx.set_node_attributes(self, dict(
            zip(np.asarray(self.nodes()), seed_colors)), 'seed_colors')

        #self.seed_colors = seed_colors
        # self.label_count = label_count
        self.point_cloud_graph = G
        #self.graph_labels_dict = dict(zip(np.asarray(self.nodes()), range(label_count)))
        #kpcg.export_anything_on_point_cloud(G, attribute=connected_component_labels, filename=filename)


    def compute_nodes_coordinates(self):
        """Compute X Y Z coordinates for each node.

        Compute X Y Z coordinates for each node of the quotient graph using the underlying point of the
        PointCloudGraph (the distance_based_graph). The method just consist in computing the mean of each coordinates of
        the underlying points of a quotient graph node.


        Returns
        -------

        """

        G = self.point_cloud_graph

        # Calcul de coordonnées moyennes pour chaque noeud du graphe quotient, dans le but d'afficher en 3D le graphe.
        new_classif = np.asarray(list((dict(G.nodes(data='quotient_graph_node')).values())))
        nodes_coords_moy = np.array([np.mean(G.nodes_coords[new_classif == n], axis=0) for n in self.nodes])

        # import scipy.ndimage as nd
        # nodes_coords_moy = np.transpose([nd.mean(G.nodes_coords[:,k],
        #                                          new_classif,
        #                                          index=list(self.nodes)) for k in range(3)])
        #
        # nodes_coords_moy = np.zeros((len(self), 3))
        # for j, n in enumerate(self.nodes):
        #     nodes_coords_moy[j] = np.mean(G.nodes_coords[new_classif == n], axis=0)
        #
        #     X = []
        #     Y = []
        #     Z = []
        #     for i in range(pcd_attribute.shape[0]):
        #         if sorted_pcd_attribute_by_quotient_graph_attribute[i, 3] == n:
        #             X.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 0])
        #             Y.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 1])
        #             Z.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 2])
        #     nodes_coords_moy[j, 0] = np.mean(X)
        #     nodes_coords_moy[j, 1] = np.mean(Y)
        #     nodes_coords_moy[j, 2] = np.mean(Z)


        # new_classif = new_classif[:, np.newaxis]
        # pcd_attribute = np.concatenate([G.nodes_coords, new_classif], axis=1)
        # sorted_pcd_attribute_by_quotient_graph_attribute = pcd_attribute[np.argsort(pcd_attribute[:, 3])]
        # nodes_coords_moy = np.zeros((len(self), 3))
        # j = 0
        # for n in self.nodes:
        #     X = []
        #     Y = []
        #     Z = []
        #     for i in range(pcd_attribute.shape[0]):
        #         if sorted_pcd_attribute_by_quotient_graph_attribute[i, 3] == n:
        #             X.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 0])
        #             Y.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 1])
        #             Z.append(sorted_pcd_attribute_by_quotient_graph_attribute[i, 2])
        #     nodes_coords_moy[j, 0] = np.mean(X)
        #     nodes_coords_moy[j, 1] = np.mean(Y)
        #     nodes_coords_moy[j, 2] = np.mean(Z)
        #     j += 1
        self.nodes_coordinates = nodes_coords_moy

        for nodes in self.nodes:
            self.nodes[nodes]['centroide_coordinates'] = np.nan
        i = 0
        for nodes in self.nodes:
            self.nodes[nodes]['centroide_coordinates'] = nodes_coords_moy[i, :]
            i += 1

    # def update_quotient_graph(self, G):
    #     labels_qg = [k for k in dict(G.nodes(data='quotient_graph_node')).values()]
    #     labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]
    #     self.build_from_pointcloudgraph(G, labels_qg_re)

    def ponderate_with_coordinates_of_points(self):
        self.compute_nodes_coordinates()
        # Calcul des poids
        for e in self.edges:
            n1 = e[0]
            n2 = e[1]
            c1 = self.nodes[n1]['centroide_coordinates']
            c2 = self.nodes[n2]['centroide_coordinates']
            dist = np.linalg.norm(c1 - c2)
            self.edges[e]['distance_centroides'] = dist

    def rebuild(self, G, clear=True):
        """

        Parameters
        ----------
        G
        clear
        filename

        Returns
        -------

        """
        if clear:
            self.clear()
        labels_qg = [k for k in dict(G.nodes(data='quotient_graph_node')).values()]
        labels_qg_re = np.asarray(labels_qg)[:, np.newaxis]
        self.build_from_pointcloudgraph(G, labels_qg_re)

    def compute_quotientgraph_metadata_on_a_node_interclass(self, node):
        G = self.point_cloud_graph
        count = {}
        for c in self[node]:
            count[c] = 0

        list_of_nodes_G = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == node]
        for ng in list_of_nodes_G:
            for neing in G[ng]:
                if G.nodes[ng]['quotient_graph_node'] != G.nodes[neing]['quotient_graph_node']:
                    count[G.nodes[neing]['quotient_graph_node']] += 1

        return count



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

    def compute_local_descriptors(self, method='all_qg_cluster', data='coords', neighborhood='radius', scale = 10):
        G = self.point_cloud_graph
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
                self.nodes[qnode]['planarity2'] = (eigenval[1] - eigenval[0]) / eigenval[1]
                self.nodes[qnode]['linearity'] = (eigenval[2] - eigenval[1]) / eigenval[2]
                self.nodes[qnode]['scattering'] = eigenval[0] / eigenval[2]
                self.nodes[qnode]['curvature_eig'] = eigenval[0] / (eigenval[0] + eigenval[2] + eigenval[1])
                self.nodes[qnode]['lambda1'] = eigenval[2]
                self.nodes[qnode]['lambda2'] = eigenval[1]
                self.nodes[qnode]['lambda3'] = eigenval[0]

        if method == 'each_point' and data == 'coords' and neighborhood == 'radius':
            # compute a new pointcloudgraph with the neighborhood wanted
            NewG = create_riemannian_graph(G.pcd, method=neighborhood, nearest_neighbors= neighborhood, radius=scale)

        if method == 'each_point' and data == 'coords' and neighborhood == 'pointcloudgraph':
            # compute each descriptor for each point of the point cloud and its neighborhood
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
                sil = sk.metrics.silhouette_samples(G.direction_gradient_on_fiedler_scaled, label, metric='euclidean')
            elif data == 'norm_gradient_vector_fiedler':
                sil = sk.metrics.silhouette_samples(G.gradient_on_fiedler, label, metric='euclidean')

        if method == 'topological':
            import scipy.ndimage as nd
            from sklearn.metrics import pairwise_distances_chunked
            # this method aims at comparing the cluster A of the point considered with the adjacent clusters of A.
            X = []

            if data == 'direction_gradient_vector_fiedler':
                X = G.direction_gradient_on_fiedler_scaled
            elif data == 'norm_gradient_vector_fiedler':
                X = G.gradient_on_fiedler

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

    def count_local_extremum_of_Fiedler(self):
        G = self.point_cloud_graph
        G.find_local_extremum_of_Fiedler()
        for c in self.nodes():
            self.nodes[c]['number_of_local_Fiedler_extremum'] = 0
            list_of_nodes_r = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == c]
            for pt in list_of_nodes_r:
                self.nodes[c]['number_of_local_Fiedler_extremum'] += G.nodes[pt]['extrem_local_Fiedler']





######### Main

if __name__ == '__main__':
    import open3d as open3d

    import spectral_clustering.similarity_graph as sgk

    pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/Older_Cheno.ply", format='ply')
    r = 18

    G = sgk.create_riemannian_graph(pcd, method='knn', nearest_neighbors=r)
    G = kpcg.PointCloudGraph(G)

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

        G = sgk.create_riemannian_graph(pcd2, method='knn', nearest_neighbors=r)
        G = kpcg.PointCloudGraph(G)


    G.compute_graph_eigenvectors()
    G.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
    G.clustering_by_kmeans_in_four_clusters_using_gradient_norm(export_in_labeled_point_cloud=True)
    #G.clustering_by_fiedler_and_optics(criteria=np.multiply(G.direction_gradient_on_fiedler_scaled, G.gradient_on_fiedler))

    QG = QuotientGraph()
    QG.build_from_pointcloudgraph(G, G.kmeans_labels_gradient)

    QG.init_topo_scores(QG.point_cloud_graph, exports=True, formulae='improved')

    QG.optimization_topo_scores(G=QG.point_cloud_graph, exports=True, number_of_iteration=10000,
                                choice_of_node_to_change='max_energy', formulae='improved')

    QG.rebuild(QG.point_cloud_graph)

    QG.segment_each_cluster_by_optics_using_directions()

    QG.init_topo_scores(G=QG.point_cloud_graph, exports=True, formulae='improved')
    QG.optimization_topo_scores(G=QG.point_cloud_graph, exports=True, number_of_iteration=1000,
                                choice_of_node_to_change='max_energy', formulae='improved')
    QG.rebuild(QG.point_cloud_graph)

    list_of_nodes_to_work = QG.oversegment_part(list_quotient_node_to_work=[10], average_size_cluster=50)
    QG.compute_direction_info(list_leaves=[])
    QG.opti_energy_dot_product(energy_to_stop=0.29, leaves_out=False, list_graph_node_to_work=list_of_nodes_to_work)




    """
    
    QG.opti_energy_dot_product(iter=8)
    QG.rebuild(QG.point_cloud_graph)

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

    QG.rebuild(G=QG.point_cloud_graph)
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
    QG2.build_from_pointcloudgraph(G, labels_qg_re)
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



