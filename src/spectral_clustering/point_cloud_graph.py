# Structure which allows to work on a class which stock all the informations.

########### Imports
import statistics as st

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsp
import scipy as sp
import sklearn.cluster as skc
import sklearn as sk

import spectral_clustering.similarity_graph as sgk
import spectral_clustering.utils.angle as utilsangle
import spectral_clustering.quotient_graph as kQG
from spectral_clustering.display_and_export import *

########### Définition classe

class PointCloudGraph(nx.Graph):
    """
    A graph structure that keep in memory all the computations made and the informations about the point cloud from which
    the graph was constructed.
    """
    #def __init__(self, method='knn', nearest_neighbors=1, radius=1.):
    def __init__(self, G=None):
        super().__init__(G)
        # self.method = method
        # self.nearest_neighbors = nearest_neighbors
        # self.radius = radius

        # self.normals = None
        if G is not None:
            self.nodes_coords = np.array(list(dict(self.nodes(data='pos')).values()))
        else:
            self.nodes_coords = None

        self.laplacian = None
        self.keigenvec = None
        self.keigenval = None
        self.gradient_on_fiedler = None
        self.direction_gradient_on_fiedler_scaled = None
        self.clustering_labels = None
        self.clusters_leaves = None
        self.kmeans_labels_gradient = None
        self.minimum_local = None
        self.pcd = None
        self.extrem_local_fiedler = None

    # def build_from_pointcloud(self, point_cloud):
    #     """Initialize the graph.
    #
    #     The graph is created from the point cloud indicated.
    #
    #     Parameters
    #     ----------
    #
    #     The point cloud in .ply format.
    #     """
    #
    #     G = sgk.create_riemannian_graph(point_cloud, method=self.method, nearest_neighbors=self.nearest_neighbors, radius=self.radius)
    #     super().__init__(G)
    #     open3d.geometry.estimate_normals(point_cloud)
    #
    #     self.normals = np.asarray(point_cloud.normals)
    #     for i in range(len(point_cloud.points)):
    #         G.add_node(i, normal=np.asarray(point_cloud.normals)[i, :])
    #
    #     self.nodes_coords = np.asarray(point_cloud.points)
    #     self.laplacian = None
    #     self.keigenvec = None
    #     self.keigenval = None
    #     self.gradient_on_fiedler = None
    #     self.direction_gradient_on_fiedler_scaled = None
    #     self.clustering_labels = None
    #     self.clusters_leaves = None
    #     self.kmeans_labels_gradient = None
    #     self.minimum_local = None

    def add_coordinates_as_attribute_for_each_node(self):
        nodes_coordinates_X = dict(zip(self.nodes(), self.nodes_coords[:, 0]))
        nodes_coordinates_Y = dict(zip(self.nodes(), self.nodes_coords[:, 1]))
        nodes_coordinates_Z = dict(zip(self.nodes(), self.nodes_coords[:, 2]))

        nx.set_node_attributes(self, nodes_coordinates_X, 'X_coordinate')
        nx.set_node_attributes(self, nodes_coordinates_Y, 'Y_coordinate')
        nx.set_node_attributes(self, nodes_coordinates_Z, 'Z_coordinate')

    def compute_graph_laplacian(self, laplacian_type='classical'):
        L = graph_laplacian(nx.Graph(self), laplacian_type)
        self.laplacian = L

    def compute_graph_eigenvectors(self, is_sparse=True, k=50, smallest_first=True, laplacian_type='classical'):
        if self.laplacian is None:
            self.compute_graph_laplacian(laplacian_type=laplacian_type)
        keigenval, keigenvec = graph_spectrum(nx.Graph(self), sparse=is_sparse, k=k, smallest_first=smallest_first,
                                              laplacian_type=laplacian_type, laplacian=self.laplacian)
        self.keigenvec = keigenvec
        self.keigenval = keigenval

        self.add_eigenvector_value_as_attribute()

    def add_eigenvector_value_as_attribute(self, k=2):
        if self.keigenvec is None:
            print("Compute the graph spectrum first !")

        node_eigenvector_values = dict(zip(self.nodes(), np.transpose(self.keigenvec[:,k-1])))
        nx.set_node_attributes(self, node_eigenvector_values, 'eigenvector_'+str(k))

    def add_anything_as_attribute(self, anything, name_of_the_new_attribute):
        node_anything_values = dict(zip(self.nodes(), anything))
        nx.set_node_attributes(self, node_anything_values, name_of_the_new_attribute)

    def compute_gradient_of_fiedler_vector(self, method='simple'):

        if method != 'by_fiedler_weight':
            A = nx.adjacency_matrix(self).astype(float)
            vp2 = np.asarray(self.keigenvec[:, 1])

            vp2_matrix = A.copy()
            vp2_matrix[A.nonzero()] = vp2[A.nonzero()[1]]

            if method == 'simple':
                node_neighbor_max_vp2_node = np.array(
                    [vp2_matrix[node].indices[np.argmax(vp2_matrix[node].data)] for node in range(A.shape[0])])
                node_neighbor_min_vp2_node = np.array(
                    [vp2_matrix[node].indices[np.argmin(vp2_matrix[node].data)] for node in range(A.shape[0])])

                # Second method fo gradient, just obtain the max and min value in the neighborhood.
                node_neighbor_max_vp2 = np.array([vp2_matrix[node].data.max() for node in range(A.shape[0])])
                node_neighbor_min_vp2 = np.array([vp2_matrix[node].data.min() for node in range(A.shape[0])])
                vp2grad = node_neighbor_max_vp2 - node_neighbor_min_vp2

                # First method not adapted to big clouds
                # node_neighbor_max_vp2 = np.array([vp2[A[node].nonzero()[1]].max() for node in range(A.shape[0])])
                # node_neighbor_min_vp2 = np.array([vp2[A[node].nonzero()[1]].min() for node in range(A.shape[0])])

            if method == 'simple_divided_by_distance':
                node_neighbor_max_vp2_node = np.array(
                    [vp2_matrix[node].indices[np.argmax(vp2_matrix[node].data)] for node in range(A.shape[0])])
                node_neighbor_max_vp2 = np.array([vp2_matrix[node, neighbor_max_node] for node, neighbor_max_node in
                                                zip(range(A.shape[0]), node_neighbor_max_vp2_node)])

                node_neighbor_min_vp2_node = np.array(
                    [vp2_matrix[node].indices[np.argmin(vp2_matrix[node].data)] for node in range(A.shape[0])])
                node_neighbor_min_vp2 = np.array([vp2_matrix[node, neighbor_min_node] for node, neighbor_min_node in
                                                zip(range(A.shape[0]), node_neighbor_min_vp2_node)])

                pcdtab = self.nodes_coords
                vp2_max_min = (node_neighbor_max_vp2 - node_neighbor_min_vp2)

                grad_weight = np.zeros(node_neighbor_min_vp2_node.shape[0])
                for i in range(node_neighbor_min_vp2_node.shape[0]):
                    grad_weight[i] = sp.spatial.distance.euclidean(pcdtab[node_neighbor_max_vp2_node[i]].reshape(1, 3), pcdtab[node_neighbor_min_vp2_node[i]].reshape(1, 3))

                vp2grad = np.divide(vp2_max_min, grad_weight)

            if method == 'simple_divided_by_distance_along_edges':
                node_neighbor_max_vp2_node = np.array(
                    [vp2_matrix[node].indices[np.argmax(vp2_matrix[node].data)] for node in range(A.shape[0])])
                node_neighbor_max_vp2 = np.array([vp2_matrix[node, neighbor_max_node] for node, neighbor_max_node in
                                                zip(range(A.shape[0]), node_neighbor_max_vp2_node)])

                node_neighbor_min_vp2_node = np.array(
                    [vp2_matrix[node].indices[np.argmin(vp2_matrix[node].data)] for node in range(A.shape[0])])
                node_neighbor_min_vp2 = np.array([vp2_matrix[node, neighbor_min_node] for node, neighbor_min_node in
                                                zip(range(A.shape[0]), node_neighbor_min_vp2_node)])

                pcdtab = self.nodes_coords
                vp2_max_min = (node_neighbor_max_vp2 - node_neighbor_min_vp2)

                grad_weight = np.zeros(node_neighbor_min_vp2_node.shape[0])
                for i in range(node_neighbor_min_vp2_node.shape[0]):
                    grad_weight[i] = sp.spatial.distance.euclidean(pcdtab[node_neighbor_max_vp2_node[i]].reshape(1, 3), pcdtab[i].reshape(1, 3)) + \
                                    sp.spatial.distance.euclidean(pcdtab[i].reshape(1, 3), pcdtab[node_neighbor_min_vp2_node[i]].reshape(1, 3))

                vp2grad = np.divide(vp2_max_min, grad_weight)

            if method == 'by_laplacian_matrix_on_fiedler_signal':
                vp2grad = self.laplacian.dot(vp2)

            if method == 'by_weight':
                node_neighbor_max_vp2_node = np.array(
                    [vp2_matrix[node].indices[np.argmax(vp2_matrix[node].data)] for node in range(A.shape[0])])
                node_neighbor_max_vp2 = np.array([vp2_matrix[node, neighbor_max_node] for node, neighbor_max_node in
                                                zip(range(A.shape[0]), node_neighbor_max_vp2_node)])

                node_neighbor_min_vp2_node = np.array(
                    [vp2_matrix[node].indices[np.argmin(vp2_matrix[node].data)] for node in range(A.shape[0])])
                node_neighbor_min_vp2 = np.array([vp2_matrix[node, neighbor_min_node] for node, neighbor_min_node in
                                                zip(range(A.shape[0]), node_neighbor_min_vp2_node)])

                wmax = np.zeros(node_neighbor_min_vp2.shape)
                wmin = np.zeros(node_neighbor_max_vp2.shape)

                for i in range(node_neighbor_min_vp2_node.shape[0]):
                    print(i)
                    wmax[i] = G.edges[node_neighbor_max_vp2_node[i], i]['weight']
                    print(wmax[i])
                    wmin[i] = G.edges[i, node_neighbor_min_vp2_node[i]]['weight']
                    print(wmin[i])

                vp2grad = np.divide((wmin*node_neighbor_max_vp2[i] - wmax*node_neighbor_min_vp2[i]), wmax+wmin)

            self.gradient_on_fiedler = vp2grad[:, np.newaxis]
            self.direction_gradient_on_fiedler_scaled = create_normalized_vector_field(node_neighbor_max_vp2_node, node_neighbor_min_vp2_node, self.nodes_coords)
            self.add_anything_as_attribute(self.direction_gradient_on_fiedler_scaled, 'direction_gradient')

        else:
            if method == 'by_fiedler_weight':
                vp2grad = np.zeros((len(self), 1))
                vp2dir = np.zeros((len(self), 3))
                line = 0
                for n in self.nodes:
                    grad = 0
                    for v in self[n]:
                        grad += (self.nodes[n]['eigenvector_2'] - self.nodes[v]['eigenvector_2']) * \
                                (self.nodes[n]['pos'] - self.nodes[v]['pos']) / np.linalg.norm(self.nodes[n]['pos'] - self.nodes[v]['pos'])
                    self.nodes[n]['norm_gradient'] = np.linalg.norm(grad)
                    self.nodes[n]['direction_gradient'] = grad / np.linalg.norm(grad)
                    self.nodes[n]['vector_gradient'] = grad
                    self.nodes[n]['vector_gradient4'] = np.append(self.nodes[n]['direction_gradient'], self.nodes[n]['norm_gradient'])
                    vp2grad[line] = self.nodes[n]['norm_gradient']
                    vp2dir[line, :] = self.nodes[n]['direction_gradient']
                    line += 1
                self.gradient_on_fiedler = vp2grad
                self.direction_gradient_on_fiedler_scaled = vp2dir

    def compute_angles_from_gradient_directions(self, angle_computed='angle_max'):
        for u in self.nodes:
            self.nodes[u][angle_computed] = 0

        for i in self.nodes:
            angles = []
            for v in self[i]:
                angle = utilsangle.angle(self.nodes[i]['direction_gradient'], self.nodes[v]['direction_gradient'], degree=True)
                angles.append(angle)
            if angle_computed == 'angle_max':
                print(max(angles))
                self.nodes[i][angle_computed] = max(angles)
            elif angle_computed == 'angle_mean':
                self.nodes[i][angle_computed] = sum(angles)/len(angles)
            elif angle_computed == 'angle_variance':
                #mean = sum(angles) / len(angles)
                #self.nodes[i][angle_computed] = sum ((a - mean) ** 2 for a in angles) / len(angles)
                self.nodes[i][angle_computed] = np.var(angles)
            elif angle_computed == 'angle_standard_deviation':
                self.nodes[i][angle_computed] = st.stdev(angles)
                #mean = sum(angles) / len(angles)
                #self.nodes[i][angle_computed] = np.sqrt(sum((a - mean) ** 2 for a in angles) / len(angles))
            elif angle_computed == 'angle_median':
                self.nodes[i][angle_computed] = st.median(angles)

    def add_gradient_of_fiedler_vector_as_attribute(self):
        node_gradient_fiedler_values = dict(zip(self.nodes(), np.transpose(self.gradient_on_fiedler)))
        nx.set_node_attributes(self, node_gradient_fiedler_values, 'gradient_of_fiedler_vector')

    def clustering_by_fiedler_and_agglomerative(self, number_of_clusters=2, criteria=[]):
        A = nx.adjacency_matrix(self)
        X = criteria
        clustering = skc.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward',
                                                     n_clusters=number_of_clusters).fit(X)

        node_clustering_label = dict(zip(self.nodes(), np.transpose(clustering.labels_)))
        nx.set_node_attributes(self, node_clustering_label, 'clustering_labels')
        self.clustering_labels = clustering.labels_[:, np.newaxis]

    def clustering_by_fiedler_and_optics(self, criteria=[]):
        X = criteria
        clustering = skc.OPTICS(min_samples=100).fit(X)
        clustering_labels = clustering.labels_[:, np.newaxis]
        for i in range(len(self.nodes)):
            self.nodes[i]['optics_label'] = clustering_labels[i]

    def clustering_by_fiedler(self, method='agglomerative', number_of_clusters=2, criteria=[], name_attribute='clustering_label'):
        X = criteria

        if method=='agglomerative':
            A = nx.adjacency_matrix(self)
            clustering = skc.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward',
                                                     n_clusters=number_of_clusters).fit(X)
        if method=='optics':
            clustering = skc.OPTICS(min_samples=100).fit(X)

        clustering_labels = clustering.labels_[:, np.newaxis]
        for i in range(len(self.nodes)):
            self.nodes[i][name_attribute] = clustering_labels[i]
        self.clustering_labels

    def clustering_by_kmeans_using_gradient_norm(self, export_in_labeled_point_cloud=False, number_of_clusters=4):
        kmeans = skc.KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(
            self.gradient_on_fiedler)

        if export_in_labeled_point_cloud is True:
            export_anything_on_point_cloud(self, attribute=kmeans.labels_[:, np.newaxis], filename='kmeans_clusters.txt')

        kmeans_labels = kmeans.labels_[:, np.newaxis]
        self.kmeans_labels_gradient = kmeans_labels

        kmeans_labels_gradient_dict = dict(
            zip(np.asarray(self.nodes()), np.transpose(np.asarray(self.kmeans_labels_gradient))[0]))
        nx.set_node_attributes(self, kmeans_labels_gradient_dict, 'kmeans_labels')

        return kmeans.cluster_centers_

    def find_local_minimum_of_gradient_norm(self):
        min_local = []
        for i in self:
            A = [n for n in self[i]]
            min_neighborhood = min(self.gradient_on_fiedler[A])
            gradient_on_point = self.gradient_on_fiedler[i]
            if gradient_on_point < min_neighborhood:
                min_local.append(i)

        self.min_local = min_local

    def find_local_extremum_of_Fiedler(self):
        extrem_local = []
        for i in self.nodes:
            self.nodes[i]['extrem_local_Fiedler']=0
            val = self.nodes[i]['eigenvector_2']
            max = True
            min = True
            for vois in self[i]:
                if self.nodes[vois]['eigenvector_2'] > val:
                    max = False
                elif self.nodes[vois]['eigenvector_2'] < val:
                    min = False
            if max or min:
                extrem_local.append(i)
                self.nodes[i]['extrem_local_Fiedler'] = 1
        self.extrem_local_fiedler = extrem_local







def graph_laplacian(G, laplacian_type='classical'):
    if laplacian_type == 'classical':
        L = nx.laplacian_matrix(G, weight='weight')
    if laplacian_type == 'simple_normalized':
        L_normalized_numpyformat = nx.normalized_laplacian_matrix(G, weight='weight')
        L = spsp.csr_matrix(L_normalized_numpyformat)

    return L

def graph_spectrum(G, sparse=True, k=50, smallest_first=True, laplacian_type='classical', laplacian=None):
    """

    Parameters
    ----------
    G : nx.Graph

    sparse
    k

    Returns
    -------

    """

    if laplacian is None:
        # fonction condensée plus efficace en quantité de points :
        L = graph_laplacian(G, laplacian_type=laplacian_type)
    else:
        L = laplacian


    if sparse:
        Lcsr = spsp.csr_matrix.asfptype(L)

        # k = 50
        # On précise que l'on souhaite les k premières valeurs propres directement dans la fonction
        # Les valeurs propres sont bien classées par ordre croissant

        # Calcul des k premiers vecteurs et valeurs propres
        if smallest_first:
            keigenval, keigenvec = spsp.linalg.eigsh(Lcsr, k=k, sigma=0, which='LM')
        else:
            #TODO check if ordering is ok
            keigenval, keigenvec = spsp.linalg.eigsh(Lcsr, k=k, which='LM')

    else:
        keigenval, keigenvec = np.linalg.eigh(L)
        if not smallest_first:
            keigenvec = keigenvec[np.argsort(-np.abs(keigenval))]
            keigenval = keigenval[np.argsort(-np.abs(keigenval))]

    return keigenval, keigenvec


def create_normalized_vector_field(list_of_max_nodes, list_of_min_nodes, pcd_coordinates):
    vectors = pcd_coordinates[list_of_max_nodes] - pcd_coordinates[list_of_min_nodes]
    # Normalization of the directions
    vectors_scaled = sk.preprocessing.normalize(vectors, norm='l2')

    return vectors_scaled


def simple_graph_to_test_methods():
    G = nx.Graph()
    G.add_nodes_from([0,1,2,3,4])
    G.add_weighted_edges_from([(0,1,10),(0,2,30),(0,3,40),(0,4,50),(1,2,20),(1,3,40),(1,4,60)])

    L = nx.laplacian_matrix(G, weight='weight')
    keigenval, keigenvec = sgk.graph_spectrum(G, k=2)

    A = nx.adjacency_matrix(G).astype(float)
    vp2 = np.asarray(keigenvec[:, 1])
    vp2_matrix = A.copy()
    vp2_matrix[A.nonzero()] = vp2[A.nonzero()[1]]

    node_neighbor_max_vp2_node = np.array([vp2_matrix[node].indices[np.argmax(vp2_matrix[node].data)] for node in range(A.shape[0])])
    node_neighbor_max_vp2 = np.array([vp2_matrix[node,neighbor_max_node] for node, neighbor_max_node in
                                      zip(range(A.shape[0]), node_neighbor_max_vp2_node)])

    node_neighbor_min_vp2_node = np.array([vp2_matrix[node].indices[np.argmin(vp2_matrix[node].data)] for node in range(A.shape[0])])
    node_neighbor_min_vp2 = np.array([vp2_matrix[node,neighbor_min_node] for node, neighbor_min_node in
                                      zip(range(A.shape[0]), node_neighbor_min_vp2_node)])

    wmax = np.zeros(node_neighbor_min_vp2.shape)
    wmin = np.zeros(node_neighbor_max_vp2.shape)

    for i in range(node_neighbor_min_vp2_node.shape[0]):
        print(i)
        wmax[i] = G.edges[node_neighbor_max_vp2_node[i], i]['weight']
        print(wmax[i])
        wmin[i] = G.edges[i, node_neighbor_min_vp2_node[i]]['weight']
        print(wmin[i])

######### Main

if __name__ == '__main__':
    import open3d as open3d

    pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenopode_propre.ply")
    r = 18
    G = sgk.create_connected_riemannian_graph(pcd, method='knn', nearest_neighbors=r)
    G = PointCloudGraph(G)
    G.compute_graph_eigenvectors()
    G.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
    #G.compute_angles_from_gradient_directions(angle_computed='angle_variance')
    #kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_variance', filename='angle_variance.txt')
    G.compute_angles_from_gradient_directions(angle_computed='angle_median')
    kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_median', filename='angle_median.txt')
    G.compute_angles_from_gradient_directions(angle_computed='angle_standard_deviation')
    kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_standard_deviation', filename='angle_standard_deviation.txt')
    G.compute_angles_from_gradient_directions(angle_computed='angle_max')
    kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_max', filename='angle_max.txt')
    G.compute_angles_from_gradient_directions(angle_computed='angle_mean')
    kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_mean', filename='angle_mean.txt')


    #X = G.gradient_on_fiedler * G.direction_gradient_on_fiedler_scaled
    #G.clustering_by_fiedler_and_agglomerative(number_of_clusters=45, criteria=X)
    #export_clustering_labels_on_point_cloud(G)
    #export_gradient_of_fiedler_vector_on_pointcloud(G)
    #display_and_export_graph_of_gradient_of_fiedler_vector(G=G, sorted_by_gradient=True)
    #display_and_export_graph_of_fiedler_vector(G)
    display_gradient_vector_field(G, normalized=False, scale= 10000.)
    #kmeans = skc.KMeans(n_clusters=15, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(G.direction_gradient_on_fiedler_scaled)
    #export_anything_on_point_cloud(G, attribute=kmeans.labels_[:, np.newaxis], filename='kmeans_clusters.txt')

    #
