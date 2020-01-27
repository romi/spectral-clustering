# Structure which allows to work on a class which stock all the informations.

########### Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsp
import scipy as sp
import sklearn.cluster as sk
import spectral_clustering.similarity_graph as sgk
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
        if laplacian_type == 'classical':
            L = nx.laplacian_matrix(self, weight='weight')
        if laplacian_type == 'simple_normalized':
            L_normalized_numpyformat = nx.normalized_laplacian_matrix(self, weight='weight')
            L = spsp.csr_matrix(L_normalized_numpyformat)

        self.Laplacian = L

    def compute_graph_eigenvectors(self, is_sparse=True, k=50, smallest_first=True, laplacian_type='classical'):
        keigenval, keigenvec = sgk.graph_spectrum(self, sparse=is_sparse, k=k, smallest_first=smallest_first,
                                                  laplacian_type=laplacian_type)
        self.compute_graph_Laplacian(laplacian_type=laplacian_type)
        self.keigenvec = keigenvec
        self.keigenval = keigenval

    def add_eigenvector_value_as_attribute(self, k=2):
        if self.keigenvec is None:
            print("Compute the graph spectrum first !")

        node_eigenvector_values = dict(zip(self.nodes(), np.transpose(self.keigenvec[:,k-1])))
        nx.set_node_attributes(self, node_eigenvector_values, 'eigenvector_'+str(k))


    def compute_gradient_of_Fiedler_vector(self, method = 'simple'):

        A = nx.adjacency_matrix(G).astype(float)
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
                grad_weight[i] = sp.spatial.distance.euclidean(pcdtab[node_neighbor_max_vp2_node[i]].reshape(1, 3), pcdtab[node_neighbor_min_vp2_node[i]].reshape(1, 3))

            vp2grad = np.divide(vp2_max_min, grad_weight)


        if method == 'by_incidence_matrix_on_Fiedler_signal':
            vp2grad = self.Laplacian.dot(vp2)


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


        self.gradient_on_Fiedler = vp2grad

    def add_gradient_of_Fiedler_vector_as_attribute(self):
        node_gradient_Fiedler_values = dict(zip(self.nodes(), np.transpose(self.gradient_on_Fiedler)))
        nx.set_node_attributes(self, node_gradient_Fiedler_values, 'gradient_of_Fiedler_vector')

def export_gradient_of_Fiedler_vector_on_pointcloud(G, filename="pcd_vp2_grad.txt"):
    pcd_vp2_grad = np.concatenate([G.nodes_coords, G.gradient_on_Fiedler[:, np.newaxis]], axis=1)
    np.savetxt(filename, pcd_vp2_grad, delimiter=",")
    print("Export du nuage avec gradient du vecteur propre 2")

def export_figure_graph_of_Fiedler_vector(G, filename="Fiedler_vector", sorted_by_fiedler_vector=True):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_sort_by_vp2 = pcd_vp2[pcd_vp2[:, 3].argsort()]
    figure = plt.figure(0)
    figure.clf()
    figure.gca().set_title("Fiedler vector")
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    if sorted_by_fiedler_vector:
        figure.gca().scatter(range(len(pcd_vp2_sort_by_vp2)), pcd_vp2_sort_by_vp2[:, 3], color='blue')
    if sorted_by_fiedler_vector is False:
        figure.gca().scatter(range(len(pcd_vp2)), pcd_vp2[:, 3], color='blue')
    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du vecteur propre 2")

def export_figure_graph_of_gradient_of_Fiedler_vector(G, filename="Gradient_of_Fiedler_vector", sorted_by_fiedler_vector=True):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_grad_vp2 = np.concatenate([pcd_vp2, G.gradient_on_Fiedler[:, np.newaxis]], axis=1)
    pcd_vp2_grad_vp2_sort_by_vp2 = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 3].argsort()]

    figure = plt.figure(1)
    figure.clf()
    figure.gca().set_title("Gradient of Fiedler vector")
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    if sorted_by_fiedler_vector:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_vp2[:, 4], color='blue')
    if sorted_by_fiedler_vector is False:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2)), pcd_vp2_grad_vp2[:, 4], color='blue')

    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig("Gradient_of_Fiedler_vector")
    print("Export du gradient")

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

    pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/cylindres/cylindre_temoin_unebranche60.ply")
    r = 8
    G = PointCloudGraph(point_cloud=pcd, method='knn', nearest_neighbors=r)
    G.compute_graph_eigenvectors(k=2)
    G.compute_gradient_of_Fiedler_vector(method='simple_divided_by_distance')
    export_gradient_of_Fiedler_vector_on_pointcloud(G)
    export_figure_graph_of_gradient_of_Fiedler_vector(G)
    export_figure_graph_of_Fiedler_vector(G)