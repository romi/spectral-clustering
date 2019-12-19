# Structure which allows to work on a class which stock all the informations.

########### Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsp
import sklearn.cluster as sk
import similarity_graph as sgk
import open3d as open3d

########### Définition classe

class PointCloudGraph(nx.Graph):
    """
    A graph structure that keep in memory all the computations made and the informations about the point cloud from which
    the graph was constructed.
    """

    def __init__(self, point_cloud, method='knn', nearest_neighbors=1, radius=1.):
        """Initialize the graph.

        The graph is created from the point cloud indicated.

        Parameters
        ----------

        The point cloud in .ply format.
        """
        self.method = method
        self.nearest_neighbors = nearest_neighbors
        self.radius = radius

        G = sgk.gen_graph(point_cloud, method=self.method, nearest_neighbors=self.nearest_neighbors, radius=self.radius)
        super().__init__(G)

        self.nodes_coords = np.asarray(point_cloud.points)
        self.Laplacian = None
        self.keigenvec = None
        self.keigenval = None
        self.gradient_on_Fiedler = None

    def add_coordinates_as_attribute_for_each_node(self):
        nodes_coordinates_X = dict(zip(self.nodes(), self.nodes_coords[:, 0]))
        nodes_coordinates_Y = dict(zip(self.nodes(), self.nodes_coords[:, 1]))
        nodes_coordinates_Z = dict(zip(self.nodes(), self.nodes_coords[:, 2]))

        nx.set_node_attributes(self, nodes_coordinates_X, 'X_coordinate')
        nx.set_node_attributes(self, nodes_coordinates_Y, 'Y_coordinate')
        nx.set_node_attributes(self, nodes_coordinates_Z, 'Z_coordinate')

    def compute_graph_Laplacian(self, laplacian_type='classical'):
        if laplacian_type=='classical':
            L = nx.laplacian_matrix(self, weight='weight')
        if laplacian_type=='simple_normalized':
            L_normalized_numpyformat = nx.normalized_laplacian_matrix(self, weight='weight')
            L = spsp.csr_matrix(L_normalized_numpyformat)

        self.Laplacian = L

    def compute_graph_eigenvectors(self, is_sparse=True, k=50, smallest_first=True, laplacian_type='classical'):
        keigenval, keigenvec = sgk.graph_spectrum(self, sparse=is_sparse, k=k, smallest_first=smallest_first,
                                                  laplacian_type=laplacian_type)
        self.keigenvec = keigenvec
        self.keigenval = keigenval

    def add_eigenvector_value_as_attribute(self, k=2):
        if self.keigenvec is None:
            print("Compute the graph spectrum first !")

        node_eigenvector_values = dict(zip(self.nodes(), np.transpose(self.keigenvec[:,k-1])))
        nx.set_node_attributes(self, node_eigenvector_values, 'eigenvector_'+str(k))

    def compute_gradient_of_Fiedler_vector(self, method = 'simple'):

        A = nx.adjacency_matrix(G)
        vp2 = np.asarray(self.keigenvec[:, 1])

        vp2_matrix = A.copy()
        vp2_matrix[A.nonzero()] = vp2[A.nonzero()[1]]

        if method == 'simple':
            # Second method fo gradient, just obtain the max and min value in the neighborhood.
            node_neighbor_max_vp2 = np.array([vp2_matrix[node].data.max() for node in range(A.shape[0])])
            node_neighbor_min_vp2 = np.array([vp2_matrix[node].data.min() for node in range(A.shape[0])])
            vp2grad = node_neighbor_max_vp2 - node_neighbor_min_vp2

            # First method not adapted to big clouds
            # node_neighbor_max_vp2 = np.array([vp2[A[node].nonzero()[1]].max() for node in range(A.shape[0])])
            # node_neighbor_min_vp2 = np.array([vp2[A[node].nonzero()[1]].min() for node in range(A.shape[0])])

        if method == 'simple_divided_by_distance':
            node_neighbor_max_vp2_node = np.array([np.argmax(vp2_matrix[node].data) for node in range(A.shape[0])])
            node_neighbor_max_vp2 = np.array([vp2_matrix[node].data[neighbor_max_node] for node,neighbor_max_node in zip(range(A.shape[0]),node_neighbor_max_vp2_node)])

            node_neighbor_min_vp2_node = np.array([np.argmin(vp2_matrix[node].data) for node in range(A.shape[0])])
            node_neighbor_min_vp2 = np.array([vp2_matrix[node].data[neighbor_min_node] for node,neighbor_min_node in zip(range(A.shape[0]),node_neighbor_min_vp2_node)])

            pcdtab = self.nodes_coords
            vp2_max_min = (node_neighbor_max_vp2 - node_neighbor_min_vp2)

            grad_weight = np.zeros(node_neighbor_min_vp2_node.shape[0])
            for i in range(node_neighbor_min_vp2_node.shape[0]):
                grad_weight[i] = sp.spatial.distance.euclidean(pcdtab[node_neighbor_max_vp2_node[i]].reshape(1, 3), pcdtab[node_neighbor_min_vp2_node[i]].reshape(1, 3))

            vp2grad = np.divide(vp2_max_min, grad_weight)

        if method == 'simple_divided_by_distance_along_edges':
            node_neighbor_max_vp2_node = np.array([np.argmax(vp2_matrix[node].data) for node in range(A.shape[0])])
            node_neighbor_max_vp2 = np.array([vp2_matrix[node].data[neighbor_max_node] for node,neighbor_max_node in zip(range(A.shape[0]),node_neighbor_max_vp2_node)])

            node_neighbor_min_vp2_node = np.array([np.argmin(vp2_matrix[node].data) for node in range(A.shape[0])])
            node_neighbor_min_vp2 = np.array([vp2_matrix[node].data[neighbor_min_node] for node,neighbor_min_node in zip(range(A.shape[0]),node_neighbor_min_vp2_node)])

            pcdtab = self.nodes_coords
            vp2_max_min = (node_neighbor_max_vp2 - node_neighbor_min_vp2)

            grad_weight = np.zeros(node_neighbor_min_vp2_node.shape[0])
            for i in range(node_neighbor_min_vp2_node.shape[0]):
                grad_weight[i] = sp.spatial.distance.euclidean(pcdtab[node_neighbor_max_vp2_node[i]].reshape(1, 3), pcdtab[node_neighbor_min_vp2_node[i]].reshape(1, 3))

            vp2grad = np.divide(vp2_max_min, grad_weight)


        #if method == 'simple_divided_by_weight_along_edges':


        #if method == '':

        self.gradient_on_Fiedler = vp2grad

    def add_gradient_of_Fiedler_vector_as_attribute(self):
        node_gradient_Fiedler_values = dict(zip(self.nodes(), np.transpose(self.gradient_on_Fiedler)))
        nx.set_node_attributes(self, node_gradient_Fiedler_values, 'gradient_of_Fiedler_vector')


######### Main

if __name__ == '__main__':

    pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/arabi_densep_clean_segm.ply")
    r = 8
    G = PointCloudGraph(point_cloud=pcd, method='knn', nearest_neighbors=r)
    G.compute_graph_eigenvectors(k=2)
    G.compute_gradient_of_Fiedler_vector()