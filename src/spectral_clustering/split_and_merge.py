import networkx as nx
import operator
import sklearn.cluster as skc
import numpy as np
from spectral_clustering.display_and_export import *

def opti_energy_dot_product(quotientgraph, energy_to_stop=0.13, leaves_out=False, list_graph_node_to_work=[], export_iter=True):
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

        quotientgraph.point_cloud_graph = G
        # quotientgraph.rebuild_quotient_graph(G=quotientgraph.point_cloud_graph,filename='graph_attribute_quotient_graph_' + str(i) + '.txt')
        if export_iter:
            quotientgraph.rebuild_quotient_graph(G=quotientgraph.point_cloud_graph,
                                        filename='graph_attribute_quotient_graph_' + str(i) + '.txt')
            display_and_export_quotient_graph_matplotlib(quotient_graph=quotientgraph, node_sizes=20,
                                                         filename="quotient_graph_matplotlib_QG_intra_class_number_" + str(i),
                                                         data_on_nodes='intra_class_node_number')
        i += 1


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
                quotientgraph.rebuild_quotient_graph(G)


    """
def oversegment_part(quotientgraph, list_quotient_node_to_work=[], average_size_cluster=20):
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
    number_of_clusters = number_of_points /average_size_cluster

    # clustering via Kmeans to obtain new labels
    clustering = skc.KMeans(n_clusters=int(number_of_clusters), init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(Xcoord)
    clustering_labels = clustering.labels_[:, np.newaxis] + max(quotientgraph.nodes) + 1

    # Integration of the new labels in the quotient graph and updates
    for i in range(len(list_of_nodes)):
        G.nodes[list_of_nodes[i]]['quotient_graph_node'] = clustering_labels[i]
    quotientgraph.rebuild_quotient_graph(G)

    # return of the list of nodes in pointcloudgraph to keep working with them
    list_of_nodes_pointcloudgraph = list_of_nodes
    return list_of_nodes_pointcloudgraph