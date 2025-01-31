#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Tests sur graphes chaînes

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import sklearn as sk
import sklearn.cluster as skc


class BranchingGraph(nx.Graph):
    """A graph structure that allows adding branches to a main stem.

    The resulting graph is formed by chains of nodes linked together at branching points.
    This structure enables the modeling of hierarchical systems or trees, where each branch
    is classified by an order relative to the main stem.

    Attributes
    ----------
    node_coords : numpy.ndarray
        A matrix of coordinates for each node in the graph. Used for visualization or spatial representation.
    branch_nodes : dict
        A dictionary where keys are branch IDs and values are lists of node IDs in that branch.
    branch_linking_node : dict
        A dictionary where keys are branch IDs and values are the node IDs to which the branch is connected (linking point).
    branch_order : dict
        A dictionary where keys are branch IDs and values are the order of each branch (relative to the main stem).
    keigenvec : numpy.ndarray or None
        Eigenvector matrix computed from the graph Laplacian. Used for spectral graph analysis.
    keigenval : numpy.ndarray or None
        Eigenvalues corresponding to the graph Laplacian. Used for spectral graph analysis.

    Examples
    --------
    >>> from spectral_clustering.branching_graph import BranchingGraph
    >>> # Create a main stem with 10 nodes
    >>> graph = BranchingGraph(stem_size=100)
    >>> linking_nodes = [32, 63, 40, 47, 50, 55, 62, 63, 70]
    >>> # Add a branch of size 5 to node 4
    >>> [graph.add_branch(branch_size=5, linking_node=ln) for ln in linking_nodes]
    >>> # Compute eigenvectors for the graph Laplacian
    >>> graph.compute_graph_eigenvectors()
    >>> print(graph.keigenvec)
    >>> # Export graph eigenvectors as a point cloud
    >>> graph.export_eigenvectors_on_pointcloud(path="./eigenvectors", k=3)

    Notes
    -----
    - This class subclasses `networkx.Graph` and extends its functionality for hierarchical branching systems.
    - Visualization libraries like matplotlib can be used to plot the graph or the point clouds saved.
    """

    def __init__(self, stem_size=100):
        """Initialize the graph with a main stem.

        The graph is created with a single stem of length `stem_size`.

        Parameters
        ----------
        stem_size : int
            The number of nodes of the main stem.
        """
        super().__init__(self)

        self.node_coords = None
        self.branch_nodes = {}
        self.branch_linking_node = {}
        self.branch_order = {}

        self.keigenvec = None
        self.keigenval = None

        self.init_main_stem(stem_size)

    def init_main_stem(self, stem_size=100):
        """
        Initialize the main stem of the graph, including nodes, edges, and attributes.
        This function sets up a linear chain (main stem) in the graph with the specified number of nodes.

        Parameters
        ----------
        stem_size : int
            The number of nodes in the main stem.
        """
        # Create a list of node IDs for the main stem
        list_nodes = [i for i in range(stem_size)]
        # Add the nodes of the stem to the graph
        self.add_nodes_from(list_nodes)

        # Determine the branch ID for the main stem
        # If branch_nodes is non-empty, the new branch ID is the max key + 1, otherwise use 0
        branch_id = np.max(list(self.branch_nodes.keys())) + 1 if len(self.branch_nodes) > 0 else 0
        # Assign all nodes of the main stem to the corresponding branch ID
        self.branch_nodes[branch_id] = list_nodes
        # Set linking node for the main stem to the first node (node 0)
        self.branch_linking_node[branch_id] = 0
        # Set branch order for the main stem to 0 (main stem is the primary chain)
        self.branch_order[branch_id] = 0

        # Assign the branch ID as a node attribute for all nodes in the main stem
        nx.set_node_attributes(
            self, dict(zip(list_nodes, [branch_id for _ in list_nodes])), 'branch_id'
        )
        # Assign the branch order as a node attribute for all nodes in the main stem
        nx.set_node_attributes(
            self, dict(zip(list_nodes, [self.branch_order[branch_id] for _ in list_nodes])), 'branch_order'
        )

        # Create a list of edges to form a linear chain by connecting each node to the next
        list_edge = [(i, i + 1) for i in range(stem_size - 1)]
        # Add the edges to the graph
        self.add_edges_from(list_edge)
        # Create 3D coordinates for the nodes, spaced 1 unit apart along the z-axis
        # The x and y coordinates are set to 0
        self.node_coords = np.asarray([[0, 0, i] for i in range(stem_size)])

    # Ajout d'une branche à la tige principale
    # Paramètres en entrée : Tb taille de la branche
    # Eb nom du node sur lequel la branche est ajoutée
    # G graphe de travail.
    # Matrice de coordonnées C tige origine
    # Facteur pour nommer les noeuds F : à modifier si l'on fait deux branches sur le même noeud
    # Choisir F supérieur à 1000 et par *10.
    # Prendre en compte la taille de la tige principale.
    # y_orientation pour voir les deux branches dans deux directions différentes
    # Jouer avec x_offset si on a plus de deux branches au même endroit.
    def add_branch(self, branch_size, linking_node, y_orientation=1, x_offset=0):
        """Create a branch and add it to the graph.

        Parameters
        ----------
        branch_size : int
            The number of nodes in the branch.
        linking_node : int
            The node on which to attach the branch.
        y_orientation : int, optional
            Whether to go left or right on the Y axis (-1 or 1).
        x_offset : float, optional
            The offset on the X axis.
        """
        # Création de la branche et ajout dans le graphe

        starting_node = np.max(list(self.nodes)) + 1
        list_nodes = [starting_node + i for i in range(branch_size)]
        self.add_nodes_from(list_nodes)

        branch_id = np.max(list(self.branch_nodes.keys())) + 1 if len(self.branch_nodes) > 0 else 0
        self.branch_nodes[branch_id] = list_nodes

        linking_branch_id = self.nodes[linking_node]['branch_id']
        linking_branch_order = self.branch_order[linking_branch_id]
        self.branch_linking_node[branch_id] = linking_node
        self.branch_order[branch_id] = linking_branch_order + 1

        nx.set_node_attributes(self, dict(zip(list_nodes, [branch_id for _ in list_nodes])), 'branch_id')
        nx.set_node_attributes(self, dict(zip(list_nodes, [self.branch_order[branch_id] for _ in list_nodes])),
                               'branch_order')

        list_edge = [(starting_node + i, starting_node + i + 1) for i in range(branch_size - 1)]
        self.add_edges_from(list_edge)

        # On relie la branche au point d'insertion
        self.add_edge(linking_node, starting_node)
        # Ajout de coordonnées pour représentation graphique
        branch_node_coords = np.array(
            [[x_offset, y_orientation * (i + 1), self.node_coords[linking_node, 2]] for i in range(branch_size)])
        # branch_node_coords = branch_node_coords.reshape(branch_size, 3)
        # print(np.shape(Cbranche))
        self.node_coords = np.concatenate((self.node_coords, branch_node_coords), axis=0)
        # C = Ctot
        # print(np.shape(C))
        # return G, C

    def compute_graph_eigenvectors(self, is_sparse=False, k=50):
        """Computes the eigenvalues and eigenvectors of the Laplacian matrix of the graph.

        Parameters
        ----------
        is_sparse : bool, optional
            Indicates whether to perform the computation in sparse mode. If ``False`` (default),
            all eigenvalues and eigenvectors are calculated in dense mode.
        k : int, optional
            The number of eigenvalues and eigenvectors to compute in sparse mode, if `is_sparse`
            is True. Ignored in dense mode. Default is ``50``.

        Notes
        -----
        This function calculates the eigenvalues and eigenvectors of the graph's Laplacian
        matrix. When `is_sparse` is set to `False`, a dense computation is performed using
        `numpy.linalg.eigh`, which computes all eigenvalues and eigenvectors. Otherwise,
        if `is_sparse` is set to `True`, the computation utilizes `scipy.sparse.linalg.eigsh`,
        which is suitable for large sparse matrices.
        """
        # Appli Laplacien
        L = nx.laplacian_matrix(self, weight='weight')
        L = L.toarray()
        print(L.shape)

        # if isinstance(L, np.ndarray):
        if not is_sparse:
            # Calcul vecteurs propres
            # Utilisation de eigsh impossible lorsque le graphe est petit.
            # eigh calcul tous les vecteurs propres.
            self.keigenval, self.keigenvec = np.linalg.eigh(L)
        else:
            self.keigenval, self.keigenvec = spsp.linalg.eigsh(L, k=k, sigma=0, which='LM')
        #
        # return keigenvec, keigenval

    # Cette fonction génère des fichiers contenant les nuages de points pour chaque vecteur propre.
    # k le nombre de vecteurs propres voulus
    # C la matrice des coordonnées
    # keigenvec la matrice des vecteurs propres
    def add_eigenvector_value_as_attribute(self, k=2, compute_branch_relative=True):
        """Adds eigenvector values as node attributes in the graph and optionally computes
        branch-relative values for the specified eigenvector.

        This method first calculates eigenvector values if not already computed and then assigns these values to graph
        node attributes. Optionally, values can be normalized within individual branches defined by nodes' branch IDs.

        Parameters
        ----------
        k : int, optional
            Index of the eigenvector to use (1-based index). Defaults to ``2``.
        compute_branch_relative : bool, optional
            If ``True`` (default), branch-relative normalized eigenvector values are computed and added as attributes.
        """
        if self.keigenvec is None:
            self.compute_graph_eigenvectors()

        # print(type(keigenvec))
        # vp2 = dict(enumerate(np.transpose(keigenvec[:,1])))
        node_eigenvector_values = dict(zip(self.nodes(), np.transpose(self.keigenvec[:, k - 1])))
        # print(vp2)
        nx.set_node_attributes(self, node_eigenvector_values, 'eigenvector_' + str(k))
        # print(G.nodes[1]['valp2'])
        # nx.write_graphml(G, "graphetestattributs")

        if compute_branch_relative:
            branch_relative_values = {}
            for i in self.nodes:
                branch_id = self.nodes[i]['branch_id']
                branch_nodes = self.branch_nodes[branch_id]

                branch_min_value = np.min([self.nodes[j]['eigenvector_' + str(k)] for j in branch_nodes])
                branch_max_value = np.max([self.nodes[j]['eigenvector_' + str(k)] for j in branch_nodes])
                branch_relative_values[i] = (self.nodes[i]['eigenvector_' + str(k)] - branch_min_value) / (
                            branch_max_value - branch_min_value)
            nx.set_node_attributes(self, branch_relative_values, 'branch_relative_eigenvector_' + str(k))

    def export_eigenvectors_on_pointcloud(self, path=".", k=50):
        """Export the eigenvectors of a graph to individual point cloud files.

        This method selects the first `k` eigenvectors computed on the graph and
        exports them as point clouds. The point cloud includes the node coordinates
        from the graph concatenated with the values of each individual eigenvector.
        Each eigenvector is saved as a separate TXT file in the specified output directory.

        Parameters
        ----------
        path : str, optional
            The directory where the output files will be saved. Default is the
            current working directory ("./").
        k : int, optional
            The number of eigenvectors to export. Default is 50.

        """
        if self.keigenvec is None:
            self.compute_graph_eigenvectors()

        keigenvec = np.asarray(self.keigenvec[:, :k])
        # print(np.shape(keigenvec))
        # print(np.shape(C))
        # Concaténation avec les labels (vecteurs propres)
        # Faire boucle pour sortir tous les nuages de points associés à chaque vecteur propre.
        for i in range(k):
            # keigenvecK = keigenvec[:, i].reshape(keigenvec.shape[0], 1)
            # print(np.shape(keigenvecK))
            pcdtabclassif = np.concatenate((self.node_coords, keigenvec[:, i][:, np.newaxis]), axis=1)
            # Sortie de tous les nuages
            filename = 'testchain' + str(i)
            print(filename)
            np.savetxt(path + "/" + filename + '.txt', pcdtabclassif, delimiter=",")

    def clustering_by_fiedler_and_agglomerative(self, number_of_clusters=2, with_coordinates=False):
        """Performs graph clustering using Fiedler vector and agglomerative clustering.

        This method leverages the Fiedler vector (second smallest eigenvector of the
        Laplacian matrix) to compute gradients across graph nodes, and then applies
        agglomerative clustering on the gradient values. Optionally incorporates
        node coordinates into the clustering process.

        Parameters
        ----------
        number_of_clusters : int, optional
            The number of clusters to form. Default is 2.
        with_coordinates : bool, optional
            If True, the clustering process includes node coordinates as an
            additional feature. Default is False.

        Notes
        -----
        This method assigns two attributes to the graph's nodes:
        1. `gradient_vp2` - The gradient of the Fiedler vector for the respective node.
        2. `clustering_label` - The cluster label assigned to the node after clustering.

        `AgglomerativeClustering` from scikit-learn is used with the 'ward' linkage
        and 'euclidean' distance metric. If coordinates are included, the features are
        scaled using `MinMaxScaler` before clustering. Connectivity of the graph
        is incorporated into the clustering process.
        """
        self.compute_graph_eigenvectors()
        self.add_eigenvector_value_as_attribute(k=2)

        A = nx.adjacency_matrix(self)
        vp2 = np.asarray(self.keigenvec[:, 1])

        vp2_matrix = np.tile(vp2, (len(self), 1))
        vp2_matrix[A.todense() == 0] = np.nan
        vp2grad = (np.nanmax(vp2_matrix, axis=1) - np.nanmin(vp2_matrix, axis=1)) / 2.
        node_vp2grad_values = dict(zip(self.nodes(), vp2grad))
        nx.set_node_attributes(self, node_vp2grad_values, 'gradient_vp2')

        if not with_coordinates:
            X = vp2grad[:, np.newaxis]
            clustering = skc.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward',
                                                     n_clusters=number_of_clusters).fit(X)

        if with_coordinates:
            X = np.concatenate((self.node_coords, vp2grad[:, np.newaxis]), axis=1)
            Xscaled = sk.preprocessing.MinMaxScaler().fit_transform(X)
            clustering = skc.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward',
                                                     n_clusters=number_of_clusters).fit(Xscaled)

        node_clustering_label = dict(zip(self.nodes(), np.transpose(clustering.labels_)))
        nx.set_node_attributes(self, node_clustering_label, 'clustering_label')


def save_eigenval_plot_G(branching_graph, filename="ValeursPropres.png"):
    """Save the eigenvalue plot of a graph to a specified file.

    This function saves a plot of the eigenvalues of the graph associated with
    `branching_graph`. If the eigenvalues are not yet computed, it calculates them
    first. The eigenvalue plot is created using matplotlib and stored in a file
    with the specified or default filename.

    Parameters
    ----------
    branching_graph : Graph
        The graph object whose eigenvalues are to be plotted. It is expected to
        have an attribute `keigenval` that contains its eigenvalues and a method
        `compute_graph_eigenvectors()` for computing those eigenvalues if necessary.
    filename : str, optional
        The name of the file where the eigenvalue plot will be saved. Defaults to
        "ValeursPropres.png".
    """
    if branching_graph.keigenval is None:
        branching_graph.compute_graph_eigenvectors()
    save_eigenval_plot(branching_graph.keigenval, filename)


def save_eigenval_plot(eigenval, filename="ValeursPropres.png"):
    """Saves a plot of eigenvalues to a file.

    This function generates a plot of the provided eigenvalues. The eigenvalues
    will be plotted as blue circles, and the x-axis corresponds to the range of
    their transposed indices. The size of the plot and layout adjustments are
    preset before saving the figure to the specified filename.

    Parameters
    ----------
    eigenval : array-like
        The eigenvalues to be plotted. Each column in `eigenval` represents a
        set of eigenvalues whose transposed indices are used as the x-axis.
    filename : str, optional
        The filename of the image file where the plot is saved. Default is "ValeursPropres.png".
    """
    figureval = plt.figure(0)
    figureval.clf()
    figureval.gca().plot(range(len(np.transpose(eigenval))), np.transpose(eigenval), 'bo')
    figureval.set_size_inches(20, 10)
    figureval.subplots_adjust(wspace=0, hspace=0)
    figureval.tight_layout()
    figureval.savefig(filename)


def save_eigenvec_plot(branching_graph, sort_values=True, filename="eigenvectors.png"):
    """Generates and saves a plot of eigenvectors from a branching graph.

    This function processes the eigenvectors of the given branching graph,
    sorts them based on specified criteria, and creates a subplot visualization
    of the eigenvectors. The generated plot is then saved to a file.

    Parameters
    ----------
    branching_graph : object
        An object representing the branching graph. It is expected to
        have the attribute `keigenvec` for precomputed eigenvectors
        or the method `compute_graph_eigenvectors` if the eigenvectors
        need to be computed.
    sort_values : bool, optional
        A flag indicating whether eigenvectors should be sorted. By default,
        it is set to True.
    filename : str, optional
        The name of the file where the eigenvector plot will be saved.
        The default value is "eigenvectors.png".

    Notes
    -----
    The function creates a figure with multiple subplots, where each subplot
    represents an eigenvector visualized as a 1D line plot. A maximum of 50
    eigenvectors is considered and plotted, depending on the size of `keigenvec`.
    """
    if branching_graph.keigenvec is None:
        branching_graph.compute_graph_eigenvectors()
    figure = plt.figure(0)
    figure.clf()
    keigenvec = branching_graph.keigenvec[:, :50]
    if sort_values:
        keigenvec = keigenvec[keigenvec[:, 1].argsort()]
    for i_vec, vec in enumerate(np.transpose(np.around(keigenvec, 10))):
        figure.add_subplot(5, 10, i_vec + 1)
        figure.gca().set_title("Eigenvector " + str(i_vec + 1))
        figure.gca().plot(range(len(vec)), vec, color='blue')
    figure.set_size_inches(20, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)


def save_single_eigenvec_plot(branching_graph, k=2, sort_values=True, filename=None):
    """Generates and saves a plot for a specified eigenvector of a branching graph.

    This function visualizes a specified eigenvector of the given branching graph.
    Eigenvectors are useful for understanding structural or spectral properties
    of the graph. The function provides an option to sort the eigenvector values
    to potentially reveal relationships among graph nodes. The resulting plot
    is color-coded based on the branch IDs of the graph nodes and is saved to
    a specified or default file.

    Parameters
    ----------
    branching_graph : object
        An object representing the branching graph that includes eigenvector
        data and branch ID information in the node attributes. The object must
        have a `keigenvec` property (or compute it through its functionality).
    k : int, optional
        The index of the eigenvector to be plotted (indexed from 1), by default 2.
    sort_values : bool, optional
        Whether to sort the eigenvector values and associated node branches, by
        default True.
    filename : str or None, optional
        The destination filename for saving the plot. If None, a default
        filename `eigenvector_k.png` is used, where `k` is the specified
        eigenvector index.
    """
    if branching_graph.keigenvec is None:
        branching_graph.compute_graph_eigenvectors()
    figure = plt.figure(0)
    figure.clf()
    vec = branching_graph.keigenvec[:, k - 1]
    branches = np.array([branching_graph.nodes[i]['branch_id'] for i in branching_graph.nodes])
    if sort_values:
        vec = vec[branching_graph.keigenvec[:, 1].argsort()]
        branches = branches[branching_graph.keigenvec[:, 1].argsort()]
    figure.gca().set_title("Eigenvector " + str(k))
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    figure.gca().scatter(range(len(vec)), vec, c=branches, cmap='jet')
    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    if filename is None:
        filename = "eigenvector_" + str(k) + ".png"
    figure.savefig(filename)


def save_eigenvector_value_along_stem_plot(branching_graph, k=2, filename="eigenvector_along_stem.png"):
    """
    Save the plot of eigenvector values along the stem of a branching graph.

    This function visualizes the eigenvector values along the stem for different
    branches of a branching graph. It creates a plot where nodes of branches are
    distributed according to their positions along the x-axis, and their corresponding
    eigenvector values are displayed on the y-axis. Branches are colored based on their
    order, and zero-crossing points (nodes where eigenvector values cross zero) are
    highlighted.

    Parameters
    ----------
    branching_graph : object
        The branching graph structure containing nodes and branches. It must provide the
        following attributes:
        - `nodes`: A dictionary-like structure where node data can be accessed and modified.
        - `branch_order`: A dictionary mapping branch IDs to their respective branch order.
        - `branch_linking_node`: A dictionary defining the linking nodes for each branch.
        - `branch_nodes`: A dictionary mapping branch IDs to their respective nodes.

        Additionally, the graph must support the method `add_eigenvector_value_as_attribute(k)`,
        which adds the eigenvector values as attributes to each node.

    k : int, optional
        The eigenvector index to be used for plotting. Defaults to 2.

    filename : str, optional
        The name of the file to save the generated plot. Defaults to "eigenvector_along_stem.png".
    """
    if not 'eigenvector_' + str(k) in branching_graph.nodes[0]:
        branching_graph.add_eigenvector_value_as_attribute(k)

    figure = plt.figure(0)
    figure.clf()

    order_colors = {0: 'darkred', 1: 'darkgoldenrod', 2: 'chartreuse', 3: 'darkcyan'}

    node_x = {}
    for branch_id in np.sort(list(branching_graph.branch_order.keys())):
        link = branching_graph.branch_linking_node[branch_id]
        link_x = node_x[link] if link in node_x else 0
        branch_node_x = [link_x + 1 + (i - np.min(branching_graph.branch_nodes[branch_id])) for i in
                         branching_graph.branch_nodes[branch_id]]
        node_x.update(dict(zip(branching_graph.branch_nodes[branch_id], branch_node_x)))

        branch_node_y = [branching_graph.nodes[i]['eigenvector_' + str(k)] for i in
                         branching_graph.branch_nodes[branch_id]]

        zero_nodes = np.array(branching_graph.branch_nodes[branch_id])[:-1][
            np.array(branch_node_y)[:-1] * np.array(branch_node_y)[1:] < 0]

        for i in zero_nodes:
            figure.gca().scatter(np.mean(branch_node_x[i:i + 2]), np.mean(branch_node_y[i:i + 2]), color='k')

        branch_order = branching_graph.branch_order[branch_id]
        figure.gca().plot(branch_node_x, branch_node_y, color=order_colors[branch_order])

    figure.set_size_inches(20, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)


def save_graph_plot(branching_graph, attribute_names=[None], colormap='jet', node_size=10, attribute_as_size=False,
                    plot_zeros=True, filename="graph.png"):
    """Generates and saves a plot of a graph with customizable node attributes and layout.

    The function creates a visualization of the input graph with options to customize
    node colors based on attributes, specify node sizes, and include zero crossings in
    attribute plots. The generated graph layout is derived using the Kamada-Kawai method.
    Plots can be saved to a specified filename.

    Parameters
    ----------
    branching_graph : networkx.Graph
        The graph to be plotted. Each node may have attributes to be visualized.
    attribute_names : list of str or None, optional
        List of node attribute names to visualize. If None or an attribute name is not
        found in a node, default coloring is applied. Defaults to [None].
    colormap : str, optional
        The name of the matplotlib colormap to use for coloring the nodes. Defaults to 'jet'.
    node_size : int or float, optional
        The default size of the nodes in the plot. Actual sizes may vary depending on
        `attribute_as_size`. Defaults to 10.
    attribute_as_size : bool, optional
        If True, scales node sizes based on the absolute values of the provided attribute.
        Otherwise, uses `node_size` for all nodes. Defaults to False.
    plot_zeros : bool, optional
        If True and the attribute name starts with 'eigenvector_', plots scatter
        points at the location of zero crossings on edges. Defaults to True.
    filename : str, optional
        File path or name where the plotted graph will be saved as an image. Defaults to 'graph.png'.

    """
    figure = plt.figure(0)
    figure.clf()

    graph_layout = nx.kamada_kawai_layout(branching_graph)

    for i_a, attribute_name in enumerate(attribute_names):

        figure.add_subplot(1, len(attribute_names), i_a + 1)

        if attribute_name is None or attribute_name not in branching_graph.nodes[0]:
            node_color = [0 for _ in branching_graph.nodes()]
            node_sizes = node_size
        else:
            node_color = [branching_graph.nodes[i][attribute_name] for i in branching_graph.nodes()]
            if attribute_as_size:
                node_sizes = [node_size * (np.abs(branching_graph.nodes[i][attribute_name]) / np.max(
                    [np.abs(branching_graph.nodes[j][attribute_name]) for j in branching_graph.nodes()])) for i in
                              branching_graph.nodes()]
            else:
                node_sizes = node_size

        nx.drawing.nx_pylab.draw_networkx(branching_graph,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=False,
                                          node_size=node_sizes,
                                          node_color=node_color,
                                          cmap=plt.get_cmap(colormap))

        if attribute_name.startswith('eigenvector_') and plot_zeros:
            zero_edges = np.array(branching_graph.edges())[np.prod(
                [[branching_graph.nodes[i][attribute_name] for i in e] for e in np.array(branching_graph.edges)],
                axis=1) < 0]

            for e in zero_edges:
                edge_points = np.array([graph_layout[i] for i in e])
                figure.gca().scatter(np.mean(edge_points[:, 0]), np.mean(edge_points[:, 1]), color='k')

        figure.gca().set_title(attribute_name, size=24)

    figure.set_size_inches(10 * len(attribute_names), 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)


# def Convert(lst):
#     res_dct = {i : v for i,v in enumerate(lst)}
#     return res_dct

########## Corps du programme : tests

if __name__ == '__main__':

    G = BranchingGraph(100)
    G.add_branch(branch_size=20, linking_node=32)
    G.add_branch(branch_size=20, linking_node=63)
    # G.add_branch(branch_size=20, linking_node=40)
    # G.add_branch(branch_size=20, linking_node=47)
    # G.add_branch(branch_size=20, linking_node=50)
    # G.add_branch(branch_size=20, linking_node=55)
    # G.add_branch(branch_size=20, linking_node=62)
    # G.add_branch(branch_size=20, linking_node=63)
    # G.add_branch(branch_size=20, linking_node=70)

    # G.add_branch(branch_size=100, linking_node=350)
    # G.add_branch(branch_size=20, linking_node=665)
    # G.add_branch(branch_size=10, linking_node=745)
    # G.add_branch(branch_size=20, linking_node=50)
    # G.add_branch(branch_size=20, linking_node=55)
    # G.add_branch(branch_size=20, linking_node=60)
    # G.add_branch(branch_size=20, linking_node=65)
    # G.add_branch(branch_size=20, linking_node=70)
    # G.add_branch(branch_size=20, linking_node=75)

    # G.add_branch(branch_size=10, linking_node=105)
    # G.add_branch(branch_size=8,linking_node=47)
    # G.add_branch(branch_size=20,linking_node=50,y_orientation=-1)
    # G.add_branch(branch_size=25,linking_node=55)
    # G.add_branch(branch_size=15,linking_node=62,y_orientation=-1)
    # G.add_branch(branch_size=20,linking_node=63)
    # G.add_branch(branch_size=3,linking_node=70,y_orientation=-1)
    # G.add_branch(branch_size=18,linking_node=75)
    # G.add_branch(branch_size=20,linking_node=79,y_orientation=-1)

    G.compute_graph_eigenvectors()
    G.add_eigenvector_value_as_attribute(2)
    G.add_eigenvector_value_as_attribute(3)
    save_graph_plot(G, attribute_names=['eigenvector_2', 'branch_relative_eigenvector_2', 'eigenvector_3'],
                    filename='graph_first_eigenvectors.png')
    save_graph_plot(G, attribute_names=['eigenvector_2', 'branch_relative_eigenvector_2'],
                    filename='../../Data/Tests/eigenvector2.png')

    for k in range(12):
        G.add_eigenvector_value_as_attribute(len(G) - k)
    save_graph_plot(G, attribute_names=['eigenvector_' + str(len(G) - k) for k in range(12)], plot_zeros=False,
                    attribute_as_size=True, node_size=50, filename='graph_last_eigenvectors.png')

    # G.clustering_by_fiedler_and_agglomerative(number_of_clusters=11, with_coordinates=True)
    # save_graph_plot(G, attribute_names=['clustering_label'])

    # save_eigenvector_value_along_stem_plot(G, k=2)

    # save_eigenval_plot(G)
    # save_single_eigenvec_plot(G,len(G))
    # save_eigenvec_plot(G)
    save_eigenvec_plot(G, sort_values=True, filename="../../Data/Tests/eigenvectorstopo.png")
    # G.export_eigenvectors_on_pointcloud(k=2)

    pcdtabclassif = G.node_coords
    # Sortie de tous les nuages
    np.savetxt('pcdbranchinggraph', pcdtabclassif, delimiter=",")
