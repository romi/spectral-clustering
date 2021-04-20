import numpy as np
import sklearn.cluster as skc
import networkx as nx


def segment_a_node_using_attribute(quotientgraph, quotient_node_to_work, attribute_to_work_with='direction_gradient', method='OPTICS'):
    """Compute a new clustering on a specified node of the QuotientGraph by changing the class of the underlying nodes
    of the PointCloudGraph associated.

    Parameters
    ----------
    quotientgraph : QuotientGraph class object.
    quotient_node_to_work : the number of the quotientgraph node we want to cluster.
    attribute_to_work_with : which attribute of the pointcloudgraph associated to the quotientgraph mentioned to use to
    part the quotient_node_to_work.
    method : method of clustering to use.

    Returns
    -------
    Nothing
    Update the attribute 'quotient_graph_node' on each node of the PointCloudGraph object according to the new clustering

    """
    G = quotientgraph.point_cloud_graph
    nw = quotient_node_to_work

    # make list of nodes inside the quotient graph node to work with
    list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == nw]

    # Use of a function to segment
    X = np.zeros((len(list_of_nodes), 3))

    for i in range(len(list_of_nodes)):
        X[i] = G.nodes[list_of_nodes[i]][attribute_to_work_with]

    if method=='KMEANS':
        clustering = skc.KMeans(n_clusters=2, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(X)
    if method=='OPTICS':
        clustering = skc.OPTICS(min_samples=100).fit(X)

    # Instead of having 0 or 1, this gives a number not already used in the quotient graph to name the nodes
    clustering_labels = clustering.labels_[:, np.newaxis] + max(quotientgraph.nodes) + 1

    # Integration of the new labels in the quotient graph and updates

    for i in range(len(list_of_nodes)):
        G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]


def segment_several_nodes_using_attribute(quotientgraph, list_quotient_node_to_work=[], algo='OPTICS',
                                          para_clustering_meth=100, attribute='direction_gradient'):
    """Compute a new clustering on a specified node of the QuotientGraph by changing the class of the underlying nodes
    of the PointCloudGraph associated.

    Parameters
    ----------
    quotientgraph : QuotientGraph class object.
    list_quotient_node_to_work : All the numbers of the quotientgraph nodes we want to cluster in a list.
    attribute : which attribute of the pointcloudgraph associated to the quotientgraph mentioned to use to
    part the quotient_node_to_work.
    algo : method of clustering to use 'kmeans' or 'OPTICS'.
    para_clustering_meth : the parameter needed for the algorithm used. For 'OPTICS' it is the minimum samples size of
    the new clusters ('min_samples'). For 'kmeans' it is the number of clusters to get at the end ('n_clusters')

    Returns
    -------
    Nothing
    Update the attribute 'quotient_graph_node' on each node of the PointCloudGraph object according to the new clustering
    Only the PointCloudGraph is updated.

    """

    lw = list_quotient_node_to_work
    G = quotientgraph.point_cloud_graph
    # make list of nodes inside the different quotient graph nodes to work with
    list_of_nodes = []
    for qnode in lw:
        list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
        list_of_nodes += list_of_nodes_each

    # Use of a function to segment
    X = np.zeros((len(list_of_nodes), 3))

    for i in range(len(list_of_nodes)):
        X[i] = G.nodes[list_of_nodes[i]][attribute]

    if algo == 'OPTICS':
        clustering = skc.OPTICS(min_samples=para_clustering_meth).fit(X)
    elif algo == 'kmeans':
        clustering = skc.KMeans(n_clusters=para_clustering_meth, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(X)

    # Selection of max value for 'quotient_graph_node' in G
    q_g = nx.get_node_attributes(G, 'quotient_graph_node')
    q_g_values = list(q_g.values())

    # Instead of having 0 or 1, this gives a number not already used in the quotient graph to name the nodes
    clustering_labels = clustering.labels_[:, np.newaxis] + max(q_g_values) + 2

    # Integration of the new labels in the quotient graph and updates
    for i in range(len(list_of_nodes)):
        G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]

    quotientgraph.point_cloud_graph = G
    
    
def segment_each_cluster_by_optics(quotientgraph, leaves_out=True, attribute_to_work='direction_gradient'):
    """Compute a new clustering by treating one node of the QuotientGraph after the other and changing the class of the
    underlying nodes of the PointCloudGraph associated.

    Parameters
    ----------
    quotientgraph : QuotientGraph class object.
    leaves_out : boolean. If True, it means that the end nodes of the QuotientGraph will not be treated.
    attribute_to_work : which attribute of the pointcloudgraph associated to the quotientgraph mentioned to use to
    part the quotient_node_to_work.

    Returns
    -------
    Nothing
    Update the attribute 'quotient_graph_node' on each node of the PointCloudGraph object according to the new clustering
    Update the quotientgraph accordingly to the new clustering.

    """
    # Put the "end" nodes in a list
    list_leaves = [x for x in quotientgraph.nodes() if quotientgraph.degree(x) == 1]
    for n in quotientgraph:
        if leaves_out:
            if n not in list_leaves and quotientgraph.nodes[n]['intra_class_node_number'] > 100:
                segment_several_nodes_using_attribute(quotientgraph=quotientgraph, list_quotient_node_to_work=[n],
                                                      attribute=attribute_to_work)
        elif leaves_out is False and quotientgraph.nodes[n]['intra_class_node_number'] > 100:
            segment_several_nodes_using_attribute(quotientgraph=quotientgraph, list_quotient_node_to_work=[n],
                                                  attribute=attribute_to_work)

    quotientgraph.rebuild_quotient_graph(quotientgraph.point_cloud_graph, filename='optics_seg_directions.txt')
