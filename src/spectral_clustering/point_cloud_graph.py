#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Structure which allows to work on a class which stock all the information.

import statistics as st

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse as spsp
import sklearn as sk
import sklearn.cluster as skc
import spectral_clustering.quotient_graph as kQG
import spectral_clustering.similarity_graph as sgk
import spectral_clustering.utils.angle as utilsangle
from spectral_clustering.display_and_export import export_anything_on_point_cloud


class PointCloudGraph(nx.Graph):
    """A graph class representing point cloud data and providing various functionalities for graph processing.

    This class is designed to represent and manipulate a graph derived from point cloud data. It supports
    storing and processing graph-related attributes such as adjacency matrix, eigenvectors, and gradient
    information. The class enables operations including spectral graph analysis and gradient computation
    on specific signal vectors.

    Attributes
    ----------
    nodes_coords : numpy.ndarray or None
        Coordinates of the nodes in the graph, derived from point cloud data. None if not initialized.
    laplacian : numpy.ndarray or None
        The Laplacian matrix of the graph. None if not computed.
    keigenvec : numpy.ndarray or None
        Array of eigenvectors of the graph Laplacian. None if not computed.
    keigenval : numpy.ndarray or None
        Array of eigenvalues of the graph Laplacian. None if not computed.
    gradient_on_fiedler : numpy.ndarray or None
        Gradient computed on the Fiedler vector of the graph. None if not computed.
    direction_gradient_on_fiedler_scaled : numpy.ndarray or None
        Scaled direction gradient computed on the Fiedler vector of the graph. None if not computed.
    clustering_labels : numpy.ndarray or None
        Labels for node clustering based on graph properties. None if not computed.
    clusters_leaves : list or None
        Leaf nodes in each cluster. None if not computed.
    kmeans_labels_gradient : numpy.ndarray or None
        Labels for k-means clustering applied to gradient data. None if not computed.
    minimum_local : list or None
        Local minima within a processed graph signal. None if not initialized.
    pcd : Open3D.geometry.PointCloud or None
        Point cloud data associated with the graph. None if not provided or initialized.
    extrem_local_fiedler : list or None
        Local extrema computed on the Fiedler vector. None if not initialized.
    """

    def __init__(self, G=None):
        super().__init__(G)
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
    #     o3d.geometry.estimate_normals(point_cloud)
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
        """Adds coordinates as attributes to each node in the graph.

        This method assigns the X, Y, and Z coordinates of each node in the graph
        as attributes named 'X_coordinate', 'Y_coordinate', and 'Z_coordinate', respectively.
        The node attributes are created by mapping the node IDs to their respective
        coordinate values in the provided coordinate data.

        Each node's X, Y, and Z coordinate values are extracted from `nodes_coords`
        and then applied as attributes to the nodes using NetworkX's
        `set_node_attributes` method.

        Notes
        -----
        This method assumes that the `nodes_coords` attribute is a NumPy array and that
        it contains node coordinate data in the format where:
            - The first column represents the X coordinates.
            - The second column represents the Y coordinates.
            - The third column represents the Z coordinates.
        The number of nodes in the graph must match the number of rows in the
        `nodes_coords` array.
        """
        nodes_coordinates_X = dict(zip(self.nodes(), self.nodes_coords[:, 0]))
        nodes_coordinates_Y = dict(zip(self.nodes(), self.nodes_coords[:, 1]))
        nodes_coordinates_Z = dict(zip(self.nodes(), self.nodes_coords[:, 2]))

        nx.set_node_attributes(self, nodes_coordinates_X, 'X_coordinate')
        nx.set_node_attributes(self, nodes_coordinates_Y, 'Y_coordinate')
        nx.set_node_attributes(self, nodes_coordinates_Z, 'Z_coordinate')

    def compute_graph_laplacian(self, laplacian_type='classical'):
        """Compute the graph Laplacian for the graph object.

        This method computes the Laplacian matrix of a graph represented by the
        current object using the specified type of Laplacian. The computed Laplacian
        differentiate various aspects of graph structure depending upon the `laplacian_type`
        parameter value provided.

        The resulting Laplacian is stored in the `laplacian` attribute of the object
        for subsequent graph-based analysis.

        Parameters
        ----------
        laplacian_type: str, optional
            The type of Laplacian matrix to compute. Default is 'classical'. Other
            options include normalized or non-standard Laplacians depending on the
            library capabilities or analysis requirements.
        """
        L = graph_laplacian(nx.Graph(self), laplacian_type)
        self.laplacian = L

    def compute_graph_eigenvectors(self, is_sparse=True, k=50, smallest_first=True, laplacian_type='classical'):
        """Computes the eigenvectors of the graph's Laplacian matrix.

        This method calculates the eigenvalues and eigenvectors of the
        graph's Laplacian matrix. It uses either a sparse or dense matrix representation
        based on the input parameters and retains the desired number of smallest or largest
        eigenvectors as specified. The computed eigenvectors and eigenvalues are stored
        as attributes of the graph, and the eigenvector values are added as attributes
        to graph nodes.

        Before computation, the graph Laplacian is generated if it does not already
        exist. The `graph_spectrum` function is used to perform the spectral
        decomposition. Finally, node attributes are updated using the computed
        eigenvector values.

        Parameters
        ----------
        is_sparse : bool, optional
            Determines if the sparse representation of the graph's Laplacian is used
            during the computation. If set to True, a sparse implementation is applied.
            Default is True.
        k : int, optional
            The number of eigenvalues and eigenvectors to compute. For example, if
            set to 50, the 50 smallest (or largest) eigenvalues are selected based on
            the `smallest_first` parameter. Default is 50.
        smallest_first : bool, optional
            Indicates whether the k smallest or largest eigenvalues are computed. If
            True, the k smallest eigenvalues (and corresponding eigenvectors) are
            computed. Default is True.
        laplacian_type : str, optional
            The type of graph Laplacian to compute. Options could include "classical"
            or other variations depending on the implementation of `compute_graph_laplacian`
            and `graph_spectrum`. Default is "classical".
        """
        if self.laplacian is None:
            self.compute_graph_laplacian(laplacian_type=laplacian_type)
        keigenval, keigenvec = graph_spectrum(nx.Graph(self), sparse=is_sparse, k=k, smallest_first=smallest_first,
                                              laplacian_type=laplacian_type, laplacian=self.laplacian)
        self.keigenvec = keigenvec
        self.keigenval = keigenval

        self.add_eigenvector_value_as_attribute()

    def add_eigenvector_value_as_attribute(self, k=2):
        """Adds the k-th eigenvector's values as a node attribute to the graph.

        This method assigns the values of the specified eigenvector from the
        precomputed eigenvector matrix `keigenvec` to the nodes of the graph. The
        attribute is stored under the name `'eigenvector_k'`, where `k` corresponds
        to the specified eigenvector index. It is required that the graph spectrum
        has been computed prior to using this method; otherwise, it will prompt
        the user to compute the spectrum.

        Parameters
        ----------
        k : int, optional
            Index of the eigenvector to be added as a node attribute, starting
            from 1. Defaults to 2. Must be in the range of available eigenvectors
            in the precomputed eigenvector matrix `keigenvec`.

        Notes
        -----
        This method modifies the graph in place by assigning the eigenvector values
        as node attributes. The spectrum of the graph must be computed before using
        this method, and the computed eigenvector matrix should be stored in the
        `keigenvec` attribute of the instance.
        """
        if self.keigenvec is None:
            print("Compute the graph spectrum first !")

        node_eigenvector_values = dict(zip(self.nodes(), np.transpose(self.keigenvec[:, k - 1])))
        nx.set_node_attributes(self, node_eigenvector_values, 'eigenvector_' + str(k))

    def add_anything_as_attribute(self, anything, name_of_the_new_attribute):
        """Assigns specified values as attributes to the nodes of a graph.

        This function adds attributes to the nodes of a graph by mapping the given
        values to the nodes based on their order, and then assigning these mapped
        values to a new attribute name provided by the user.

        Parameters
        ----------
        anything : list
            A list of values corresponding to nodes of the graph. The values should
            be in the same order as the nodes for proper mapping.
        name_of_the_new_attribute : str
            The name of the new attribute that will be added to each node, with
            corresponding values from the input list.
        """
        node_anything_values = dict(zip(self.nodes(), anything))
        nx.set_node_attributes(self, node_anything_values, name_of_the_new_attribute)

    def compute_gradient_of_fiedler_vector(self, method='simple'):
        """Computes the gradient of the Fiedler vector for a graph using various methods.

        The Fiedler vector is used for a variety of graph analysis tasks, including spectral clustering
        and structure discovery.
        The gradient is computed based on adjacency relationships and associated weights depending on the
        chosen method.

        Parameters
        ----------
        method : str, optional
            The method to compute the gradient of the Fiedler vector. It supports the following options:
            - 'simple': Computes the gradient as the difference between the maximum and minimum values of
              the Fiedler vector in the neighborhood of each node.
            - 'simple_divided_by_distance': Computes the same as 'simple', but normalizes the gradient
              values by the Euclidean distance between nodes corresponding to the maximum and minimum
              values in the neighborhood.
            - 'simple_divided_by_distance_along_edges': Normalizes the gradient by summing Euclidean
              distances along edges linking the nodes with the maximum, minimum, and central (current node)
              Fiedler values.
            - 'by_laplacian_matrix_on_fiedler_signal': Uses the graph Laplacian matrix to compute the
              gradient directly on the Fiedler vector.
            - 'by_weight': Computes the gradient using edge weights and Fiedler values.
            - 'by_fiedler_weight': Computes the gradient as the weighted difference of positional
              displacements and Fiedler values between neighboring nodes.

        Raises
        ------
        ValueError
            If an unsupported method is provided.
        AttributeError
            If any required attributes such as `keigenvec`, `nodes_coords`, or `laplacian` is missing on
            the object or improperly initialized.

        Notes
        -----
        This function modifies several instance attributes, including:
        - `gradient_on_fiedler`: Stores the computed gradient tensor for the Fiedler vector.
        - `direction_gradient_on_fiedler_scaled`: Stores the normalized direction vectors of the computed
          gradient for each node.
        - Direction gradient as an added attribute on the object.

        This function requires the graph to be represented in a NetworkX-compatible adjacency matrix or with
        node and edge attributes such as positional coordinates or weight information. Different methods
        may involve additional computational costs; for instance, distance-based normalizations loop over
        nodes and their neighborhoods, which could be computationally expensive for dense graphs.

        """
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
                    grad_weight[i] = sp.spatial.distance.euclidean(pcdtab[node_neighbor_max_vp2_node[i]].reshape(1, 3),
                                                                   pcdtab[node_neighbor_min_vp2_node[i]].reshape(1, 3))

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
                    grad_weight[i] = sp.spatial.distance.euclidean(pcdtab[node_neighbor_max_vp2_node[i]].reshape(1, 3),
                                                                   pcdtab[i].reshape(1, 3)) + \
                                     sp.spatial.distance.euclidean(pcdtab[i].reshape(1, 3),
                                                                   pcdtab[node_neighbor_min_vp2_node[i]].reshape(1, 3))

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

                vp2grad = np.divide((wmin * node_neighbor_max_vp2[i] - wmax * node_neighbor_min_vp2[i]), wmax + wmin)

            self.gradient_on_fiedler = vp2grad[:, np.newaxis]
            self.direction_gradient_on_fiedler_scaled = create_normalized_vector_field(node_neighbor_max_vp2_node,
                                                                                       node_neighbor_min_vp2_node,
                                                                                       self.nodes_coords)
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
                                (self.nodes[n]['pos'] - self.nodes[v]['pos']) / np.linalg.norm(
                            self.nodes[n]['pos'] - self.nodes[v]['pos'])
                    self.nodes[n]['norm_gradient'] = np.linalg.norm(grad)
                    self.nodes[n]['direction_gradient'] = grad / np.linalg.norm(grad)
                    self.nodes[n]['vector_gradient'] = grad
                    self.nodes[n]['vector_gradient4'] = np.append(self.nodes[n]['direction_gradient'],
                                                                  self.nodes[n]['norm_gradient'])
                    vp2grad[line] = self.nodes[n]['norm_gradient']
                    vp2dir[line, :] = self.nodes[n]['direction_gradient']
                    line += 1
                self.gradient_on_fiedler = vp2grad
                self.direction_gradient_on_fiedler_scaled = vp2dir

    def compute_angles_from_gradient_directions(self, angle_computed='angle_max'):
        """
        Computes and assigns a specified statistical value derived from gradient direction angles
        between a node and its neighbors to each node in the graph.

        The computation is performed based on the gradient direction of the current node
        and its neighbors. The user can specify the statistical value to be computed
        from the angles, such as maximum, mean, variance, standard deviation, or median.

        Parameters
        ----------
        angle_computed : str, optional
            The type of statistical value to compute from the gradient direction angles.
            Acceptable values are:
            - 'angle_max': Maximum angle among all neighbors.
            - 'angle_mean': Mean of the angles.
            - 'angle_variance': Variance of the angles.
            - 'angle_standard_deviation': Standard deviation of the angles.
            - 'angle_median': Median of the angles.
            Default is 'angle_max'.
        """
        for u in self.nodes:
            self.nodes[u][angle_computed] = 0

        for i in self.nodes:
            angles = []
            for v in self[i]:
                angle = utilsangle.angle(self.nodes[i]['direction_gradient'], self.nodes[v]['direction_gradient'],
                                         degree=True)
                angles.append(angle)
            if angle_computed == 'angle_max':
                print(max(angles))
                self.nodes[i][angle_computed] = max(angles)
            elif angle_computed == 'angle_mean':
                self.nodes[i][angle_computed] = sum(angles) / len(angles)
            elif angle_computed == 'angle_variance':
                # mean = sum(angles) / len(angles)
                # self.nodes[i][angle_computed] = sum ((a - mean) ** 2 for a in angles) / len(angles)
                self.nodes[i][angle_computed] = np.var(angles)
            elif angle_computed == 'angle_standard_deviation':
                self.nodes[i][angle_computed] = st.stdev(angles)
                # mean = sum(angles) / len(angles)
                # self.nodes[i][angle_computed] = np.sqrt(sum((a - mean) ** 2 for a in angles) / len(angles))
            elif angle_computed == 'angle_median':
                self.nodes[i][angle_computed] = st.median(angles)

    def add_gradient_of_fiedler_vector_as_attribute(self):
        """Adds the gradient of the Fiedler vector for each node as a node attribute.

        This method computes the gradient of the Fiedler vector and assigns it to a
        new node attribute called 'gradient_of_fiedler_vector' for all nodes in the graph.
        The gradient is given as a dictionary where the keys are node identifiers and
        the values are the gradient values corresponding to those nodes. This information
        can be useful for analyzing the structural properties of the graph.

        Returns
        -------
        None
            This method does not return any value. It modifies the graph in place by
            adding a new node attribute.
        """
        node_gradient_fiedler_values = dict(zip(self.nodes(), np.transpose(self.gradient_on_fiedler)))
        nx.set_node_attributes(self, node_gradient_fiedler_values, 'gradient_of_fiedler_vector')

    def clustering_by_fiedler_and_agglomerative(self, number_of_clusters=2, criteria=[]):
        """Clusters the nodes of the graph using Fiedler embeddings and an agglomerative clustering algorithm.

        This method applies the adjacency matrix of the graph to create clusters of
        nodes based on their similarity or proximity as determined by the provided `criteria`.

        Parameters
        ----------
        number_of_clusters : int, optional
            The number of clusters to form. Defaults to 2.
        criteria : array-like
            The attributes or features of the nodes, which are used to compute the clustering.
        """
        A = nx.adjacency_matrix(self)
        X = criteria
        clustering = skc.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward',
                                                 n_clusters=number_of_clusters).fit(X)

        node_clustering_label = dict(zip(self.nodes(), np.transpose(clustering.labels_)))
        nx.set_node_attributes(self, node_clustering_label, 'clustering_labels')
        self.clustering_labels = clustering.labels_[:, np.newaxis]

    def clustering_by_fiedler_and_optics(self, criteria=[]):
        """Clusters data points using Fiedler vector and OPTICS algorithm.

        This method applies the OPTICS clustering algorithm with a specified minimum number of samples
        (min_samples=100) on the given input criteria. The computed clustering labels are then
        used to assign an 'optics_label' to each node in the object's nodes list.

        Parameters
        ----------
        criteria : list, optional
            A list of criteria representing data points, typically required by the OPTICS clustering
            algorithm for analysis and computation. Each element in the list corresponds to a feature
            vector representing a node.
        """
        X = criteria
        clustering = skc.OPTICS(min_samples=100).fit(X)
        clustering_labels = clustering.labels_[:, np.newaxis]
        for i in range(len(self.nodes)):
            self.nodes[i]['optics_label'] = clustering_labels[i]

    def clustering_by_fiedler(self, method='agglomerative', number_of_clusters=2, criteria=[],
                              name_attribute='clustering_label'):
        """Performs clustering on a graph using the specified method and criteria.

        This method applies clustering techniques to a graph using a given method
        (either 'agglomerative' or 'optics') based on the provided criteria. For
        the 'agglomerative' method, the graph's adjacency matrix is used as the
        connectivity constraint. The clustering labels generated are stored as
        node attributes within the graph.

        Parameters
        ----------
        method : str, optional
            The clustering method to be used. Accepted values are 'agglomerative'
            and 'optics'. Default is 'agglomerative'.
        number_of_clusters : int, optional
            The number of clusters to be formed. Only applicable when the method
            is 'agglomerative'. Default is 2.
        criteria : list
            The criteria or feature matrix used as input for the clustering algorithm.
        name_attribute : str, optional
            The attribute name to store the clustering labels in the graph nodes.
            Default is 'clustering_label'.

        """
        X = criteria

        if method == 'agglomerative':
            A = nx.adjacency_matrix(self)
            clustering = skc.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward',
                                                     n_clusters=number_of_clusters).fit(X)
        if method == 'optics':
            clustering = skc.OPTICS(min_samples=100).fit(X)

        clustering_labels = clustering.labels_[:, np.newaxis]
        for i in range(len(self.nodes)):
            self.nodes[i][name_attribute] = clustering_labels[i]
        self.clustering_labels

    def clustering_by_kmeans_using_gradient_norm(self, export_in_labeled_point_cloud=False, number_of_clusters=4):
        """Performs K-means clustering on the gradient norm of the graph's Fiedler vector.

        This method applies the K-means clustering algorithm to nodes of a graph,
        based on their gradient norm values calculated on the graph's Fiedler
        vector. The number of clusters and the option to export the clustering
        results in a labeled point cloud file are configurable. The method also
        stores the clustering labels as node attributes in the graph.

        Parameters
        ----------
        export_in_labeled_point_cloud : bool, optional
            If True, exports the clustering labels into a labeled point cloud
            file named 'kmeans_clusters.txt'. Defaults to False.
        number_of_clusters : int, optional
            The number of clusters to create using K-means. Defaults to 4.

        Returns
        -------
        numpy.ndarray
            The coordinates of the cluster centers as determined by K-means.
        """
        kmeans = skc.KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(
            self.gradient_on_fiedler)

        if export_in_labeled_point_cloud is True:
            export_anything_on_point_cloud(self, attribute=kmeans.labels_[:, np.newaxis],
                                           filename='kmeans_clusters.txt')

        kmeans_labels = kmeans.labels_[:, np.newaxis]
        self.kmeans_labels_gradient = kmeans_labels

        kmeans_labels_gradient_dict = dict(
            zip(np.asarray(self.nodes()), np.transpose(np.asarray(self.kmeans_labels_gradient))[0]))
        nx.set_node_attributes(self, kmeans_labels_gradient_dict, 'kmeans_labels')

        return kmeans.cluster_centers_

    def find_local_minimum_of_gradient_norm(self):
        """Find and store local minima of the gradient norm in the provided field.

        This method identifies the elements where the calculated gradient norm
        is smaller than the minimum gradient norm of their direct neighbors
        in the field. It then stores these local minima for further
        processing or usage within the object.

        Notes
        -----
        The method assumes the object contains a `gradient_on_fiedler` attribute
        associated with the relevant gradient values for analysis and an iterable
        structure to define neighborhood relationships. It updates the `min_local`
        attribute upon execution, which stores the identified local minima indices.
        """
        min_local = []
        for i in self:
            A = [n for n in self[i]]
            min_neighborhood = min(self.gradient_on_fiedler[A])
            gradient_on_point = self.gradient_on_fiedler[i]
            if gradient_on_point < min_neighborhood:
                min_local.append(i)

        self.min_local = min_local

    def find_local_extremum_of_Fiedler(self):
        """Identifies the nodes in the graph that are local extrema based on the second-smallest eigenvector
        (Fiedler vector) of the graph's Laplacian matrix.

        Nodes are compared to their immediate neighbors to determine whether they are local maxima or minima.

        Attributes
        ----------
        extrem_local_fiedler : list
            Contains the IDs of nodes identified as local extrema based on the
            Fiedler vector values. Updated after the function is executed.

        Notes
        -----
        Each node in the graph is evaluated by comparing its eigenvector_2 value
        to those of its neighbors. Nodes are marked as local extrema if their
        eigenvector_2 value is higher or lower than all of their neighbors. The
        property `extrem_local_Fiedler` is updated for each node to indicate its
        status (1 if it is a local extremum, 0 otherwise).
        """
        extrem_local = []
        for i in self.nodes:
            self.nodes[i]['extrem_local_Fiedler'] = 0
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
    """
    Compute the graph Laplacian matrix of a given graph.

    This function computes the graph Laplacian matrix of the input graph
    `G` based on the specified type. The graph Laplacian is widely used in
    various graph-based algorithms, such as spectral clustering and network
    analysis. Depending on the value of `laplacian_type`, the function
    returns either the classical graph Laplacian matrix or the simple
    normalized Laplacian matrix. The computed Laplacian is returned as a
    sparse matrix.

    Parameters
    ----------
    G : networkx.Graph
        The input graph for which the Laplacian matrix is computed. It can
        be any NetworkX-compatible graph object.
    laplacian_type : str, optional, default='classical'
        A string specifying the type of graph Laplacian to compute.
        Accepts either 'classical' or 'simple_normalized'.
        If 'classical' is passed, the classical unnormalized Laplacian matrix is computed.
        If 'simple_normalized' is passed, the simple normalized Laplacian matrix is returned.

    Returns
    -------
    scipy.sparse.csr_matrix
        The computed Laplacian matrix in Compressed Sparse Row (CSR) format.

    """
    if laplacian_type == 'classical':
        L = nx.laplacian_matrix(G, weight='weight')
    elif laplacian_type == 'simple_normalized':
        L_normalized_numpyformat = nx.normalized_laplacian_matrix(G, weight='weight')
        L = spsp.csr_matrix(L_normalized_numpyformat)
    else:
        raise Exception(f"Unknown specified Laplacian type '{laplacian_type}'!")

    return L


def graph_spectrum(G, sparse=True, k=50, smallest_first=True, laplacian_type='classical', laplacian=None):
    """Compute the spectrum of a graph using its Laplacian matrix.

    The function computes eigenvalues and eigenvectors of the graph's Laplacian
    matrix. It supports both dense and sparse computation methods, and allows
    control over whether the smallest or largest eigenvalues are computed. The type
    of Laplacian used can also be specified.

    Parameters
    ----------
    G : networkx.Graph
        The input graph for which the spectrum is computed.
    sparse : bool, optional
        If True, uses sparse matrix computation for efficiency. Defaults to True.
    k : int, optional
        The number of eigenvalues and eigenvectors to compute. Defaults to 50.
    smallest_first : bool, optional
        If True, computes the smallest eigenvalues first. Defaults to True.
    laplacian_type : str, optional
        The type of Laplacian matrix to use. Possible values include 'classical'.
        Defaults to 'classical'.
    laplacian : numpy.ndarray, optional
        Precomputed Laplacian matrix to use instead of computing it internally from
        the graph `G`. If None, the Laplacian will be computed based on `G`.

    Returns
    -------
    numpy.ndarray
        Eigenvalues of the Laplacian matrix, as a 1D array.
    numpy.ndarray
        Eigenvectors of the Laplacian matrix, as a 2D array where each column is an eigenvector.
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
            # TODO check if ordering is ok
            keigenval, keigenvec = spsp.linalg.eigsh(Lcsr, k=k, which='LM')

    else:
        keigenval, keigenvec = np.linalg.eigh(L)
        if not smallest_first:
            keigenvec = keigenvec[np.argsort(-np.abs(keigenval))]
            keigenval = keigenval[np.argsort(-np.abs(keigenval))]

    return keigenval, keigenvec


def create_normalized_vector_field(list_of_max_nodes, list_of_min_nodes, pcd_coordinates):
    """Creates a normalized vector field based on input node indices and point cloud coordinates.

    This function computes vectors between corresponding nodes specified in the list of maximum
    and minimum nodes using their respective coordinates in the point cloud. The computed
    vectors are then normalized to unit length using L2 norm.

    Parameters
    ----------
    list_of_max_nodes : list of int
        A list containing the indices of the maximum nodes in the point cloud.
    list_of_min_nodes : list of int
        A list containing the indices of the minimum nodes in the point cloud.
    pcd_coordinates : numpy.ndarray
        A two-dimensional array representing the coordinates of the point cloud.
        Each row corresponds to a point in the point cloud, and its dimensions
        are determined by the spatial representation of the data (e.g., 3 for
        3D Cartesian coordinates).

    Returns
    -------
    numpy.ndarray
        A two-dimensional array containing the normalized vectors. Each row corresponds
        to the normalized vector computed for the respective node pairs from the input
        lists and the point cloud coordinates. The shape is the same as the number of
        node pairs, with the dimensionality matching the input point cloud coordinates.
    """
    vectors = pcd_coordinates[list_of_max_nodes] - pcd_coordinates[list_of_min_nodes]
    # Normalization of the directions
    vectors_scaled = sk.preprocessing.normalize(vectors, norm='l2')

    return vectors_scaled


def simple_graph_to_test_methods():
    """Generate a test graph and perform spectral graph analysis

    This function creates a simple undirected weighted graph, computes its Laplacian
    matrix, eigenvalues, and eigenvectors. It also analyzes node-level relationships
    based on eigenvector components, identifying nodes associated with the maximum
    and minimum values of specific eigenvector components among their neighbors.
    Additionally, it retrieves the corresponding weights of these connections.
    """
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_weighted_edges_from([(0, 1, 10), (0, 2, 30), (0, 3, 40), (0, 4, 50), (1, 2, 20), (1, 3, 40), (1, 4, 60)])

    L = nx.laplacian_matrix(G, weight='weight')
    keigenval, keigenvec = sgk.graph_spectrum(G, k=2)

    A = nx.adjacency_matrix(G).astype(float)
    vp2 = np.asarray(keigenvec[:, 1])
    vp2_matrix = A.copy()
    vp2_matrix[A.nonzero()] = vp2[A.nonzero()[1]]

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


######### Main


if __name__ == '__main__':
    import open3d as o3d

    pcd = o3d.io.read_point_cloud("Data/chenopode_propre.ply")
    r = 18
    G = sgk.create_connected_riemannian_graph(pcd, method='knn', nearest_neighbors=r)
    G = PointCloudGraph(G)
    G.compute_graph_eigenvectors()
    G.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
    # G.compute_angles_from_gradient_directions(angle_computed='angle_variance')
    # kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_variance', filename='angle_variance.txt')
    G.compute_angles_from_gradient_directions(angle_computed='angle_median')
    kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_median', filename='angle_median.txt')
    G.compute_angles_from_gradient_directions(angle_computed='angle_standard_deviation')
    kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_standard_deviation',
                                                    filename='angle_standard_deviation.txt')
    G.compute_angles_from_gradient_directions(angle_computed='angle_max')
    kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_max', filename='angle_max.txt')
    G.compute_angles_from_gradient_directions(angle_computed='angle_mean')
    kQG.export_some_graph_attributes_on_point_cloud(G, graph_attribute='angle_mean', filename='angle_mean.txt')

    # X = G.gradient_on_fiedler * G.direction_gradient_on_fiedler_scaled
    # G.clustering_by_fiedler_and_agglomerative(number_of_clusters=45, criteria=X)
    # export_clustering_labels_on_point_cloud(G)
    # export_gradient_of_fiedler_vector_on_pointcloud(G)
    # display_and_export_graph_of_gradient_of_fiedler_vector(G=G, sorted_by_gradient=True)
    # display_and_export_graph_of_fiedler_vector(G)
    display_gradient_vector_field(G, normalized=False, scale=10000.)
    # kmeans = skc.KMeans(n_clusters=15, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(G.direction_gradient_on_fiedler_scaled)
    # export_anything_on_point_cloud(G, attribute=kmeans.labels_[:, np.newaxis], filename='kmeans_clusters.txt')

    #
