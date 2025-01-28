#!/usr/bin/env python
# -*- coding: utf-8 -*-

import operator
import time

import networkx as nx
import numpy as np
import sklearn.cluster as skc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

from spectral_clustering.display_and_export import export_some_graph_attributes_on_point_cloud
from spectral_clustering.quotientgraph_operations import compute_quotient_graph_mean_attribute_from_points


def opti_energy_dot_product_old(quotientgraph, energy_to_stop=0.13, leaves_out=False, list_graph_node_to_work=[],
                                export_iter=True):
    """
    Optimize the energy dot product in a graph by iteratively fusing nodes with higher energy edges,
    rebuilding the graph, and exporting the results. The method modifies the graph and performs operations
    like edge energy computation, label updates, and graph attribute exports.

    Parameters
    ----------
    quotientgraph : object
        A graph object representing the quotient graph. Assumes it has methods to compute direction info,
        rebuild the graph, and attributes related to the point cloud graph.
    energy_to_stop : float, optional
        Threshold energy value to stop the optimization process. Default is 0.13.
    leaves_out : bool, optional
        If True, treats the graph clusters excluding the leaf nodes during the optimization steps.
        Default is False.
    list_graph_node_to_work : list, optional
        A list of specific graph nodes from the point cloud to consider during the optimization process.
        If empty, the method will compute and process all edges with energy details in the graph. Default is an empty list.
    export_iter : bool, optional
        If True, exports graph attributes and visualizations at every iteration of the optimization process.
        Default is True.

    Notes
    -----
    This function is designed for specialized usage with quotient graphs and associated point cloud graphs.
    The graph manipulation includes operations like merging graph nodes associated by minimum energy,
    updating attributes in the point cloud graph, and regenerating the quotient graph. The energy is
    calculated between nodes or edges, directly impacting the choice of clusters or edges for merging.
    Repeated iterations are performed until the minimum energy exceeds the specified `energy_to_stop`
    value. Optionally, leaf nodes may be excluded during processing to focus on core clusters.

    The function also relies on external calls, such as exporting attributes to a file and displaying
    the resulting quotient graphs using matplotlib. These external utilities must be implemented
    separately, and their proper functioning is assumed.
    """
    # take the edge of higher energy and fuse the two nodes involved, then rebuilt the graph, export and redo
    list_leaves = [x for x in quotientgraph.nodes() if quotientgraph.degree(x) == 1]
    energy_min = 0
    i = 0

    while energy_min < energy_to_stop:
        G = quotientgraph.point_cloud_graph
        if not list_graph_node_to_work:
            # select the edge
            quotientgraph.compute_direction_info(list_leaves)
            energy_per_edges = nx.get_edge_attributes(quotientgraph, 'energy_dot_product')
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
            S = nx.subgraph(quotientgraph, list_of_new_quotient_graph_nodes)
            quotientgraph.compute_direction_info(list_leaves=[])
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

        if export_iter:
            export_some_graph_attributes_on_point_cloud(pcd_g=quotientgraph.point_cloud_graph,
                                                        graph_attribute='quotient_graph_node',
                                                        filename='graph_attribute_quotient_graph_' + str(i) + '.txt')
            quotientgraph.rebuild(G=quotientgraph.point_cloud_graph)
            display_and_export_quotient_graph_matplotlib(quotient_graph=quotientgraph, node_sizes=20,
                                                         filename="quotient_graph_matplotlib_QG_intra_class_number_" + str(
                                                             i),
                                                         data_on_nodes='intra_class_node_number')
        i += 1

    quotientgraph.rebuild(G=quotientgraph.point_cloud_graph)
    export_some_graph_attributes_on_point_cloud(pcd_g=quotientgraph.point_cloud_graph,
                                                graph_attribute='quotient_graph_node',
                                                filename='final_split_and_merge.txt')

    """
    # try to treat the clusters two by two, without the leaves in it. following a shortest path
    for l in list_leaves:
        if l != root:
            path = nx.bidirectional_shortest_path(quotientgraph, root, l)
            path.pop()
            path.pop(0)
            le = len(path)
            if len(path) >= 2:                    
            for i in range(le-1):
                    quotientgraph.segment_several_nodes_using_attribute(list_quotient_node_to_work=[path[i], path[i+1]])
                G = quotientgraph.point_cloud_graph
                quotientgraph.rebuild(G)


    """


def create_a_subgraph_copy(G, list_of_nodes=[]):
    """
    Creates a subgraph as a copy based on the given list of nodes. This function preserves the
    attributes of the original graph and nodes. If the graph is a multigraph, it copies all
    edges and their attributes for the specified nodes; otherwise, it works with simple graphs.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph or networkx.MultiGraph or networkx.MultiDiGraph
        The original graph from which the subgraph is to be created. This can be any NetworkX
        graph instance.

    list_of_nodes : list, optional
        A list of nodes to be included in the subgraph. If not provided, an empty list will
        result in a subgraph with no nodes. Each node in this list must exist in the original
        graph.

    Returns
    -------
    SG : networkx.Graph or networkx.DiGraph or networkx.MultiGraph or networkx.MultiDiGraph
        A deep copy of the subgraph containing only the nodes and edges specified by the
        `list_of_nodes`. The type of the resulting graph matches the type of the input graph.
    """
    largest_wcc = list_of_nodes
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
    if SG.is_multigraph():
        SG.add_edges_from((n, nbr, key, d)
                          for n, nbrs in G.adj.items() if n in largest_wcc
                          for nbr, keydict in nbrs.items() if nbr in largest_wcc
                          for key, d in keydict.items())
    else:
        SG.add_edges_from((n, nbr, d)
                          for n, nbrs in G.adj.items() if n in largest_wcc
                          for nbr, d in nbrs.items() if nbr in largest_wcc)
    SG.graph.update(G.graph)
    return SG


def opti_energy_dot_product(quotientgraph, subgraph_riemannian, angle_to_stop=30,
                            export_iter=True, list_leaves=[]):
    """
    Optimize graph energy by iteratively merging nodes based on dot product energy.

    This function modifies the given quotient graph and its subgraph to iteratively
    optimize energy by merging nodes connected by edges with the lowest energy.
    The merging process involves updating node attributes and subgraph structures.
    Optionally, iteration results can be exported or visualized after each update.

    Parameters
    ----------
    quotientgraph : networkx.Graph
        The quotient graph on which optimization is performed. This is modified in
        place and used in conjunction with its associated point cloud graph.

    subgraph_riemannian : networkx.Graph
        A subgraph of the quotient graph focusing on a particular region of the
        point cloud. This graph plays a key role in computing and organizing
        cluster information.

    angle_to_stop : float, optional
        The angle (in degrees) that determines the stopping criterion for energy
        optimization. The default is 30 degrees. The energy to stop is computed
        as `1 - cos(angle_to_stop)`.

    export_iter : bool, optional
        Boolean flag to specify if intermediate results of the quotient graph
        optimization should be exported. The default is True.

    list_leaves : list, optional
        A list of node identifiers in the graph that are considered leaves.
        These nodes may affect energy calculations during optimization.
    """
    energy_to_stop = 1 - np.cos(np.radians(angle_to_stop))
    # take the edge of higher energy and fuse the two nodes involved, then rebuilt the graph, export and redo
    energy_min = 0
    i = 0
    SG = subgraph_riemannian
    # Create a hard copy subgraph of the quotient graph to compute the energy on a focused part
    list_quotient_node_to_work = []
    for no in SG.nodes:
        list_quotient_node_to_work.append(SG.nodes[no]['quotient_graph_node'])
    list_of_new_quotient_graph_nodes = list(set(list_quotient_node_to_work))
    SQG = create_a_subgraph_copy(quotientgraph, list_of_new_quotient_graph_nodes)
    SQG.point_cloud_graph = SG
    SQG.compute_direction_info(list_leaves=list_leaves)

    while energy_min < energy_to_stop:
        # compute energy on nodes
        start1 = time.time()
        end1 = time.time()
        time1 = end1 - start1
        print(time1)
        # selection of the edge that bears the less energy
        start2 = time.time()
        energy_per_edges = nx.get_edge_attributes(SQG, 'energy_dot_product')
        edge_to_delete = min(energy_per_edges.items(), key=operator.itemgetter(1))[0]
        print(edge_to_delete)
        print(energy_per_edges[edge_to_delete])
        energy_min = energy_per_edges[edge_to_delete]
        end2 = time.time()
        time2 = end2 - start2
        print(time2)

        # Update node direction mean and intra class node number
        m1 = SQG.nodes[edge_to_delete[0]]['dir_gradient_mean']
        m2 = SQG.nodes[edge_to_delete[1]]['dir_gradient_mean']
        n1 = SQG.nodes[edge_to_delete[0]]['intra_class_node_number']
        n2 = SQG.nodes[edge_to_delete[1]]['intra_class_node_number']
        SQG.nodes[edge_to_delete[0]]['dir_gradient_mean'] = (1 / (1 + n2 / n1)) * m1 + (1 / (1 + n1 / n2)) * m2
        SQG.nodes[edge_to_delete[0]]['intra_class_node_number'] = n1 + n2

        # one cluster is merged into the other one, change of labels on the sub pointcloudgraph label
        start3 = time.time()
        list_of_nodes = [x for x, y in SG.nodes(data=True) if y['quotient_graph_node'] == edge_to_delete[1]]
        for j in range(len(list_of_nodes)):
            SG.nodes[list_of_nodes[j]]['quotient_graph_node'] = edge_to_delete[0]
        end3 = time.time()
        time3 = end3 - start3
        print(time3)
        start4 = time.time()

        # Update subgraph_quotient structure
        SQG = nx.contracted_nodes(SQG, edge_to_delete[0], edge_to_delete[1], self_loops=False)
        SQG.point_cloud_graph = SG
        end4 = time.time()
        time4 = end4 - start4
        # Update dot energies
        for e in SQG.edges(edge_to_delete[0]):
            v1 = SQG.nodes[e[0]]['dir_gradient_mean']
            v2 = SQG.nodes[e[1]]['dir_gradient_mean']
            if e[0] in list_leaves or e[1] in list_leaves:
                SQG.edges[e]['energy_dot_product'] = 4
            else:
                dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
                energy_dot_product = 1 - dot
                SQG.edges[e]['energy_dot_product'] = energy_dot_product

        print(time4)
        print('end loop')

        if export_iter:
            export_some_graph_attributes_on_point_cloud(pcd_g=quotientgraph.point_cloud_graph,
                                                        graph_attribute='quotient_graph_node',
                                                        filename='graph_attribute_quotient_graph_' + str(i) + '.txt')
            # display_and_export_quotient_graph_matplotlib(quotient_graph=quotientgraph, node_sizes=20,
            # filename="quotient_graph_matplotlib_QG_intra_class_number_" + str(
            #    i),
            # data_on_nodes='intra_class_node_number')
        i += 1

    quotientgraph.rebuild(G=quotientgraph.point_cloud_graph)
    export_some_graph_attributes_on_point_cloud(pcd_g=quotientgraph.point_cloud_graph,
                                                graph_attribute='quotient_graph_node',
                                                filename='final_opti_energy_dot_product.txt')


def oversegment_part_return_list(quotientgraph, list_quotient_node_to_work=[], average_size_cluster=20):
    """
    Clusters points in a point cloud graph and updates the quotient graph nodes.

    This function performs clustering on nodes within specified quotient graph nodes and updates
    their associated quotient graph labels based on the clustering results. The updated
    quotient graph integrates new labels for further processing.

    Parameters
    ----------
    quotientgraph : object
        The quotient graph, which contains the point cloud graph structure (`point_cloud_graph`)
        and nodes to be updated with new labels.
    list_quotient_node_to_work : list, optional
        A list of quotient graph nodes to be processed. The nodes within these quotient graph
        nodes are clustered and updated. Default is an empty list.
    average_size_cluster : int, optional
        The average number of points per cluster, used to determine the desired number of
        clusters during the clustering process. Default is 20.

    Returns
    -------
    list
        A list of point cloud graph nodes (from the quotient graph node updates)
        that were processed and are available for subsequent operations.
    """
    lw = list_quotient_node_to_work
    G = quotientgraph.point_cloud_graph
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
    number_of_clusters = number_of_points / average_size_cluster

    # clustering via Kmeans to obtain new labels
    clustering = skc.KMeans(n_clusters=int(number_of_clusters), init='k-means++', n_init=20, max_iter=300,
                            tol=0.0001).fit(Xcoord)
    clustering_labels = clustering.labels_[:, np.newaxis] + max(quotientgraph.nodes) + 1

    # Integration of the new labels in the quotient graph and updates
    for i in range(len(list_of_nodes)):
        G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]
    quotientgraph.rebuild(G)

    # return of the list of nodes in pointcloudgraph to keep working with them
    list_of_nodes_pointcloudgraph = list_of_nodes
    return list_of_nodes_pointcloudgraph


def oversegment_part(quotientgraph, subgraph_riemannian, average_size_cluster=20):
    """
    Segments an input subgraph using clustering and updates the corresponding quotient graph.

    This function performs oversegmentation of a given subgraph using the KMeans clustering
    algorithm. It determines a number of clusters based on the average cluster size provided.
    The resulting cluster labels are then integrated into the quotient graph to represent
    the updated segmentation.

    Parameters
    ----------
    quotientgraph : object
        The input quotient graph to be updated with new segmentation labels.

    subgraph_riemannian : networkx.Graph
        The subgraph to be segmented, where each node must have a 'pos' attribute representing
        its spatial coordinates.

    average_size_cluster : int, optional
        The target average number of nodes per cluster. Defaults to 20.

    Notes
    -----
    This function expects a Riemannian subgraph that includes spatial positions stored under the
    'pos' node attribute. KMeans clustering is performed on these positions. The function also
    assumes that the quotient graph has methods to update itself with new node labels. The
    clustering process includes configuring KMeans to perform up to 20 runs (`n_init=20`) and
    allow 300 iterations per run with specified tolerance.
    """
    SG = subgraph_riemannian

    # get all the x,y,z coordinates in an np.array
    t1 = time.time()
    Xcoord = np.zeros((len(SG), 3))
    for u in range(len(SG)):
        Xcoord[u] = SG.nodes[list(SG.nodes)[u]]['pos']
    t2 = time.time()
    print(t2 - t1)
    # determine a number of cluster we would like to have
    number_of_points = len(SG)
    number_of_clusters = number_of_points / average_size_cluster

    t3 = time.time()
    # clustering via Kmeans to obtain new labels
    clustering = skc.KMeans(n_clusters=int(number_of_clusters), init='k-means++', n_init=20, max_iter=300,
                            tol=0.0001).fit(Xcoord)
    clustering_labels = clustering.labels_[:, np.newaxis] + max(quotientgraph.nodes) + 1
    t4 = time.time()
    print(t4 - t3)
    # Integration of the new labels in the quotient graph and updates
    t5 = time.time()
    num = 0
    for i in SG.nodes:
        SG.nodes[i]['quotient_graph_node'] = clustering_labels[num]
        num += 1
    quotientgraph.rebuild(quotientgraph.point_cloud_graph)
    t6 = time.time()
    print(t6 - t5)


def create_subgraphs_to_work(quotientgraph, list_quotient_node_to_work=[]):
    """
    Creates subgraphs from the given quotient graph by focusing on specific nodes to work with.
    This function builds a Riemannian subgraph using the original graph stored in the given
    quotient graph, and includes the nodes mapped to specified quotient graph nodes.

    Parameters
    ----------
    quotientgraph : Any
        The quotient graph object that contains the point cloud graph.
    list_quotient_node_to_work : list, optional
        List of nodes in the quotient graph for which the corresponding nodes in the point
        cloud graph will be included in the subgraph. Defaults to an empty list.

    Returns
    -------
    networkx.Graph
        The created subgraph that includes all the nodes from the original graph associated
        with the given list of quotient graph nodes.
    """
    lqg = list_quotient_node_to_work
    G = quotientgraph.point_cloud_graph

    lpcd = []
    for qnode in lqg:
        list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
        lpcd += list_of_nodes_each

    subgraph_riemannian = nx.subgraph(G, lpcd)

    return subgraph_riemannian


# RESEGMENTATION VIA ELBOW METHOD

def select_minimum_centroid_class(clusters_centers):
    """
    Selects the label of the cluster with the minimum centroid value.

    This function identifies the cluster with the smallest centroid
    value from the provided array of cluster centroids and returns
    the corresponding cluster label.

    Parameters
    ----------
    clusters_centers : numpy.ndarray
        A 1D numpy array containing the centroid values of all clusters.

    Returns
    -------
    int
        The label of the cluster with the smallest centroid value.
    """
    mincluster = np.where(clusters_centers == np.amin(clusters_centers))
    labelmincluster = mincluster[0][0]
    return labelmincluster


def select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, labelpointcloudgraph, attribute='kmeans_labels'):
    """
    Select all quotient graph nodes corresponding to a specific point cloud graph cluster.

    This function identifies and selects the quotient graph nodes which correspond to a
    specific cluster in the point cloud graph. The selection is based on the provided
    clustering labels and a specified attribute.

    Parameters
    ----------
    G : networkx.Graph
        The original point cloud graph containing the clustering labels.
    QG : networkx.Graph
        The quotient graph derived from the point cloud graph.
    labelpointcloudgraph : int
        The label of the cluster in the point cloud graph to be matched.
    attribute : str, optional
        The clustering attribute name used for matching, default is 'kmeans_labels'.

    Returns
    -------
    list_leaves : list
        A list of quotient graph nodes associated with the specific cluster label.

    """
    # Select clusters that were associated with the smallest value of kmeans centroid.
    compute_quotient_graph_mean_attribute_from_points(G, QG, attribute=attribute)
    list_leaves = [x for x in QG.nodes() if QG.nodes[x][attribute + '_mean'] == labelpointcloudgraph]
    return list_leaves


def resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=[], number_of_cluster_tested=10,
                                      attribute='norm_gradient', number_attribute=1, standardization=False, numer=1,
                                      G_mod=True, export_div=False, ret=False):
    """Resegments the nodes of a given quotient graph using the elbow method to find the optimal number of clusters.

    This function takes a quotient graph and a list of nodes to rework, applies the elbow method to
    resegment them into a more optimal clustering structure, and updates the graph accordingly. The
    function uses k-means clustering for segmentation and provides an option to standardize certain
    attributes prior to clustering.

    Parameters
    ----------
    QG : any
        The quotient graph on which the resegmentation will be performed.
    QG_nodes_to_rework : list, optional
        A list of quotient graph nodes to rework their clustering.
        Defaults to an empty list.
    number_of_cluster_tested : int, optional
        The maximum number of clusters to test using the elbow method.
        Defaults to `10`.
    attribute : str, optional
        The node attribute used for clustering.
        Defaults to ``'norm_gradient'``.
    number_attribute : int, optional
        The dimensionality of the node attribute used for clustering.
        Defaults to ``1``.
    standardization : bool, optional
        If True, the attributes used for clustering are standardized using StandardScaler.
        Defaults to ``False``.
    numer : int, optional
        An offset added to the new cluster labels.
        Defaults to ``1``.
    G_mod : bool, optional
        If True, modifies the original graph by updating the nodes with the new cluster labels.
        Defaults to ``True``.
    export_div : bool, optional
        If True, exports the new clustered labels to text files.
        Defaults to ``False``.
    ret : bool, optional
        If True, returns the cluster centers of the new segmentation.
        Defaults to ``False``.

    Returns
    -------
    numpy.ndarray, optional
        An array of the cluster centers, if `ret` is set to True and clustering is performed successfully.
    """
    G = QG.point_cloud_graph
    qg_values = nx.get_node_attributes(G, 'quotient_graph_node')
    maxi = max(qg_values.values())
    num = maxi + numer
    for i in QG_nodes_to_rework:
        sub = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=[i])
        SG = sub
        list_nodes = list(SG.nodes)
        # creation of matrices to work with in the elbow_method package
        if len(SG.nodes) > number_of_cluster_tested:
            Xcoord = np.zeros((len(SG), 3))
            for u in range(len(SG)):
                Xcoord[u] = SG.nodes[list(SG.nodes)[u]]['pos']
            Xnorm = np.zeros((len(SG), number_attribute))
            for u in range(len(SG)):
                Xnorm[u] = SG.nodes[list(SG.nodes)[u]][attribute]

            if standardization:
                sc = StandardScaler()
                Xnorm = sc.fit_transform(Xnorm)

            # Instantiate the clustering model and visualizer
            model = KMeans()
            visualizer = KElbowVisualizer(model, k=(1, number_of_cluster_tested), metric='distortion')

            visualizer.fit(Xnorm)  # Fit the data to the visualizer
            # visualizer.show()        # Finalize and render the figure
            # Resegment and actualize the pointcloudgraph
            k_opt = visualizer.elbow_value_
            print(k_opt)
            if k_opt > 1:
                clustering = skc.KMeans(n_clusters=k_opt, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(
                    Xnorm)
                new_labels = np.zeros((len(SG.nodes), 4))
                new_labels[:, 0:3] = Xcoord
                for pt in range(len(list_nodes)):
                    new_labels[pt, 3] = clustering.labels_[pt] + num
                    if G_mod:
                        G.nodes[list_nodes[pt]]['quotient_graph_node'] = new_labels[pt, 3]
                num += k_opt + len(list_nodes)

                # print(i, ' divided in ', k_opt)
                if export_div:
                    np.savetxt('pcd_new_labels_' + str(i) + '.txt', new_labels, delimiter=",")
                    indices = np.argsort(new_labels[:, 3])
                    arr_temp = new_labels[indices]
                    np.array_split(arr_temp, np.where(np.diff(arr_temp[:, 3]) != 0)[0] + 1)
                    v = 0
                    for arr in arr_temp:
                        np.savetxt('pcd_new_labels_' + str(i) + str(v) + '.txt', arr, delimiter=",")
                        v += 1
                if ret is True:
                    cluster_c = clustering.cluster_centers_
                    return cluster_c

            else:
                print("nothing to cluster")
