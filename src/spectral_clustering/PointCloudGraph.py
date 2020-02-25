# Structure which allows to work on a class which stock all the informations.

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
        self.direction_gradient_on_Fiedler_scaled = None
        self.clustering_labels = None
        self.clusters_leaves = None
        self.kmeans_labels_gradient = None
        self.minimum_local = None

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

    def add_anything_as_attribute(self, anything, name_of_the_new_attribute):
        node_anything_values = dict(zip(self.nodes(), np.transpose(anything)))
        nx.set_node_attributes(self, node_anything_values, name_of_the_new_attribute)


    def compute_gradient_of_Fiedler_vector(self, method = 'simple'):

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


        if method == 'by_Laplacian_matrix_on_Fiedler_signal':
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


        self.gradient_on_Fiedler = vp2grad[:, np.newaxis]
        self.direction_gradient_on_Fiedler_scaled = create_normalized_vector_field(node_neighbor_max_vp2_node, node_neighbor_min_vp2_node, self.nodes_coords)


    def add_gradient_of_Fiedler_vector_as_attribute(self):
        node_gradient_Fiedler_values = dict(zip(self.nodes(), np.transpose(self.gradient_on_Fiedler)))
        nx.set_node_attributes(self, node_gradient_Fiedler_values, 'gradient_of_Fiedler_vector')


    def clustering_by_fiedler_and_agglomerative(self, number_of_clusters=2, criteria=[]):
        A = nx.adjacency_matrix(self)
        X = criteria
        clustering = skc.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward',
                                                     n_clusters=number_of_clusters).fit(X)

        node_clustering_label = dict(zip(self.nodes(), np.transpose(clustering.labels_)))
        nx.set_node_attributes(self, node_clustering_label, 'clustering_labels')
        self.clustering_labels = clustering.labels_[:, np.newaxis]


    def clustering_by_kmeans_in_four_clusters_using_gradient_norm(self, export_in_labeled_point_cloud=False):
        kmeans = skc.KMeans(n_clusters=4, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(
            self.gradient_on_Fiedler)

        if export_in_labeled_point_cloud is True:
            export_anything_on_point_cloud(self, attribute=kmeans.labels_[:, np.newaxis], filename='kmeans_clusters.txt')

        kmeans_labels = kmeans.labels_[:, np.newaxis]
        self.kmeans_labels_gradient = kmeans_labels


    def find_local_minimum_of_gradient_norm(self):
        min_local = []
        for i in self:
            A = [n for n in self[i]]
            min_neighborhood = min(self.gradient_on_Fiedler[A])
            gradient_on_point = self.gradient_on_Fiedler[i]
            if gradient_on_point < min_neighborhood:
                min_local.append(i)

        self.min_local = min_local





def create_normalized_vector_field(list_of_max_nodes, list_of_min_nodes, pcd_coordinates):
    vectors = pcd_coordinates[list_of_max_nodes] - pcd_coordinates[list_of_min_nodes]
    # Normalization of the directions
    vectors_scaled = sk.preprocessing.normalize(vectors, norm='l2')

    return vectors_scaled

def display_gradient_vector_field(G, normalized=True, scale= 1.):
    from cellcomplex.property_topomesh.creation import vertex_topomesh
    from cellcomplex.property_topomesh.visualization.vtk_actor_topomesh import VtkActorTopomesh
    from cellcomplex.property_topomesh.visualization.vtk_tools import vtk_display_actors

    n_points = G.nodes_coords.shape[0]

    topomesh = vertex_topomesh(dict(zip(range(n_points), G.nodes_coords)))

    if normalized:
        vectors = G.direction_gradient_on_Fiedler_scaled
    if normalized is False:
        vectors = G.gradient_on_Fiedler * G.direction_gradient_on_Fiedler_scaled

    topomesh.update_wisp_property('vector', 0, dict(zip(range(n_points), vectors)))

    actors = []

    vector_actor = VtkActorTopomesh(topomesh, degree=0, property_name='vector')
    vector_actor.vector_glyph = 'arrow'
    vector_actor.glyph_scale = scale
    vector_actor.update(colormap='Reds', value_range=(0,0))
    actors += [vector_actor.actor]

    # Change of background
    vtk_display_actors(actors, background=(0.9, 0.9, 0.9))

    vtk_display_actors(actors)

def export_clustering_labels_on_point_cloud(G, filename="pcd_clustered.txt"):
    pcd_clusters = np.concatenate([G.nodes_coords, G.clustering_labels], axis=1)
    np.savetxt(filename, pcd_clusters, delimiter=",")
    print("Export du nuage avec les labels de cluster")

def export_anything_on_point_cloud(G, filename="pcd_attribute.txt", attribute = "attribute"):
    pcd_attribute = np.concatenate([G.nodes_coords, attribute], axis=1)
    np.savetxt(filename, pcd_attribute, delimiter=",")
    print("Export du nuage avec les attributs demandés")

def export_gradient_of_Fiedler_vector_on_pointcloud(G, filename="pcd_vp2_grad.txt"):
    pcd_vp2_grad = np.concatenate([G.nodes_coords, G.gradient_on_Fiedler], axis=1)
    np.savetxt(filename, pcd_vp2_grad, delimiter=",")
    print("Export du nuage avec gradient du vecteur propre 2")

def export_Fiedler_vector_on_pointcloud(G, filename="pcd_vp2.txt"):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    np.savetxt(filename, pcd_vp2, delimiter=",")
    print("Export du nuage avec le vecteur propre 2")

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

def export_figure_graph_of_gradient_of_Fiedler_vector(G, filename="Gradient_of_Fiedler_vector", sorted_by_fiedler_vector=True, sorted_by_gradient=False):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_grad_vp2 = np.concatenate([pcd_vp2, G.gradient_on_Fiedler], axis=1)
    pcd_vp2_grad_vp2_sort_by_vp2 = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 3].argsort()]
    pcd_vp2_grad_vp2_sort_by_grad = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 4].argsort()]

    figure = plt.figure(1)
    figure.clf()
    figure.gca().set_title("Gradient of Fiedler vector")
    plt.autoscale(enable=True, axis='both', tight=None)
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    if sorted_by_fiedler_vector and sorted_by_gradient is False:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_vp2[:, 4], color='blue')
    if sorted_by_fiedler_vector is False and sorted_by_gradient is False:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2)), pcd_vp2_grad_vp2[:, 4], color='blue')

    if sorted_by_gradient:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_grad[:, 4], color='blue')


    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
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

    pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenopode_propre.ply")
    r = 18
    G = PointCloudGraph(point_cloud=pcd, method='knn', nearest_neighbors=r)
    G.compute_graph_eigenvectors()
    G.compute_gradient_of_Fiedler_vector(method='simple')

    X = G.gradient_on_Fiedler * G.direction_gradient_on_Fiedler_scaled
    G.clustering_by_fiedler_and_agglomerative(number_of_clusters=45, criteria=X)
    export_clustering_labels_on_point_cloud(G)
    export_gradient_of_Fiedler_vector_on_pointcloud(G)
    export_figure_graph_of_gradient_of_Fiedler_vector(G=G, sorted_by_gradient=True)
    #export_figure_graph_of_Fiedler_vector(G)
    #display_gradient_vector_field(G, normalized=False, scale= 10000.)
    kmeans = skc.KMeans(n_clusters=4, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(G.gradient_on_Fiedler)
    export_anything_on_point_cloud(G, attribute=kmeans.labels_[:, np.newaxis], filename='kmeans_clusters.txt')

    #

    kmeans_labels = kmeans.labels_[:, np.newaxis]

    connected_component_labels = np.zeros(kmeans_labels.shape, dtype=int)
    seed_kmeans_labels = []
    cluster_colors = ['#0000FF','#00FF00','#FFFF00','#FF0000']
    seed_colors = []
    next_label = 0

    visited = np.zeros(kmeans_labels.shape, dtype=int)
    queue = []

    for i in range(kmeans_labels.shape[0]):

        # Find a seed to initialize the seed fill process.

        if visited[i] == 0:
            seed = i
            visited[i] = 1
            queue.append(i)
            seed_kmeans_labels.append(kmeans_labels[seed])
            seed_colors.append(cluster_colors[kmeans_labels[seed][0]])

            # Region growing from the specified seed.

            while len(queue) != 0:

                current = queue.pop(0)
                connected_component_labels[current] = next_label

                for n in G[current]:
                    if kmeans_labels[n] == kmeans_labels[seed] and visited[n] == 0:
                        queue.append(n)
                        visited[n] = 1

            next_label += 1

    export_anything_on_point_cloud(G, attribute=connected_component_labels)

    quotient_graph_manual = nx.Graph()
    quotient_graph_manual.add_nodes_from(range(next_label))
    node_quotient_graph_labels = dict(zip(np.asarray(quotient_graph_manual.nodes()), range(next_label)))
    node_kmeans_labels_values = dict(zip(np.asarray(quotient_graph_manual.nodes()), np.transpose(np.asarray(seed_kmeans_labels))[0]))
    nx.set_node_attributes(quotient_graph_manual, node_kmeans_labels_values, 'kmeans_labels')

    for (u, v) in G.edges:
        a = connected_component_labels[u, 0]
        b = connected_component_labels[v, 0]
        if a != b:
            if not quotient_graph_manual.has_edge(a, b):
                quotient_graph_manual.add_edge(a, b)

    figure = plt.figure(0)
    figure.clf()
    graph_layout = nx.kamada_kawai_layout(quotient_graph_manual)
    colormap = 'jet'
    node_sizes = 20
    node_color = [quotient_graph_manual.nodes[i]['kmeans_labels']/4 for i in quotient_graph_manual.nodes()]
    nx.drawing.nx_pylab.draw_networkx(quotient_graph_manual,
                                      ax=figure.gca(),
                                      pos=graph_layout,
                                      with_labels=True,
                                      node_size=node_sizes,
                                      node_color=seed_colors,
                                      labels=node_quotient_graph_labels,
                                      cmap=plt.get_cmap(colormap))

    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()



    """
    Gcopy = nx.MultiGraph()
    Gcopy.add_nodes_from(G)
    Gcopy.add_edges_from(G.edges)

    node_clusters_values = dict(zip(G.nodes(), connected_component_labels))
    same_attribute = lambda u, v: node_clusters_values[u] == node_clusters_values[v]

    quotient_graph = nx.quotient_graph(Gcopy, partition=same_attribute, edge_relation=None,
                                       create_using=nx.MultiGraph)
    print("quotient_graph_ok")

    figure = plt.figure(0)
    figure.clf()
    graph_layout = nx.kamada_kawai_layout(G)
    colormap = 'jet'
    node_size = 10
    nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                      ax=figure.gca(),
                                      pos=graph_layout,
                                      with_labels=False,
                                      node_size=node_sizes,
                                      cmap=plt.get_cmap(colormap))

    figure.set_size_inches(10 * len(attribute_names), 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()



    G.find_local_minimum_of_gradient_norm()
    np.savetxt('min_loc.txt', G.nodes_coords[G.min_local], delimiter=",")
    
    """

    """
    # Obtain the list of points associated with leaves
    clusters = kmeans.labels_.tolist()
    node_cluster0 = [i for i, x in enumerate(clusters) if x == 0]
    clusters_array = kmeans.labels_[:,np.newaxis]
    G.clusters_leaves = clusters_array
    G.add_anything_as_attribute(anything=clusters_array, name_of_the_new_attribute='clusters_leaves')

    # Creating Quotient Graph
    clusters_leaves = nx.get_node_attributes(G, 'clusters_leaves')
    node_clusters_values = dict(zip(G.nodes(), clusters_array))
    same_attribute = lambda u, v: node_clusters_values[u] == node_clusters_values[v]

    Gcopy = nx.MultiGraph()
    Gcopy.add_nodes_from(G)
    Gcopy.add_edges_from(G.edges)

    quotient_graph = nx.quotient_graph(Gcopy, partition=same_attribute, edge_relation=None,
                                       create_using=nx.MultiGraph)
    print("quotient_graph_ok")

    figure = plt.figure(0)
    figure.clf()
    graph_layout = nx.kamada_kawai_layout(G)
    colormap = 'jet'
    node_size = 10
    nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                      ax=figure.gca(),
                                      pos=graph_layout,
                                      with_labels=False,
                                      node_size=node_sizes,
                                      cmap=plt.get_cmap(colormap))

    figure.set_size_inches(10 * len(attribute_names), 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    """