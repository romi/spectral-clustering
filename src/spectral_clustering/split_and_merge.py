import networkx as nx
import operator
import sklearn.cluster as skc
import numpy as np
from spectral_clustering.display_and_export import *


def opti_energy_dot_product_old(quotientgraph, energy_to_stop=0.13, leaves_out=False, list_graph_node_to_work=[], export_iter=True):
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
            export_some_graph_attributes_on_point_cloud(pointcloudgraph=quotientgraph.point_cloud_graph,
                                                        graph_attribute='quotient_graph_node',
                                                        filename = 'graph_attribute_quotient_graph_' + str(i) + '.txt')
            quotientgraph.rebuild(G=quotientgraph.point_cloud_graph)
            display_and_export_quotient_graph_matplotlib(quotient_graph=quotientgraph, node_sizes=20,
                                                         filename="quotient_graph_matplotlib_QG_intra_class_number_" + str(i),
                                                         data_on_nodes='intra_class_node_number')
        i += 1


    quotientgraph.rebuild(G=quotientgraph.point_cloud_graph)
    export_some_graph_attributes_on_point_cloud(pointcloudgraph=quotientgraph.point_cloud_graph,
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

def opti_energy_dot_product(quotientgraph, subgraph_riemannian, energy_to_stop=0.13,
                            export_iter=True):

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

    while energy_min < energy_to_stop:
        # compute energy on nodes
        SQG.compute_direction_info(list_leaves=[])

        # selection of the edge that bears the less energy
        energy_per_edges = nx.get_edge_attributes(SQG, 'energy_dot_product')
        edge_to_delete = min(energy_per_edges.items(), key=operator.itemgetter(1))[0]
        print(edge_to_delete)
        print(energy_per_edges[edge_to_delete])
        energy_min = energy_per_edges[edge_to_delete]

        # one cluster is merged into the other one, change of labels on the sub pointcloudgraph label
        list_of_nodes = [x for x, y in SG.nodes(data=True) if y['quotient_graph_node'] == edge_to_delete[1]]
        for j in range(len(list_of_nodes)):
            SG.nodes[list_of_nodes[j]]['quotient_graph_node'] = edge_to_delete[0]
        # Update subgraph_quotient structure
        SQG = nx.contracted_nodes(SQG, edge_to_delete[0], edge_to_delete[1], self_loops=False)
        SQG.point_cloud_graph = SG
        print('end loop')


        if export_iter:
            export_some_graph_attributes_on_point_cloud(pointcloudgraph=quotientgraph.point_cloud_graph,
                                                        graph_attribute='quotient_graph_node',
                                                        filename='graph_attribute_quotient_graph_' + str(i) + '.txt')
            #display_and_export_quotient_graph_matplotlib(quotient_graph=quotientgraph, node_sizes=20,
                                                         #filename="quotient_graph_matplotlib_QG_intra_class_number_" + str(
                                                         #    i),
                                                         #data_on_nodes='intra_class_node_number')
        i += 1


    quotientgraph.rebuild(G=quotientgraph.point_cloud_graph)
    export_some_graph_attributes_on_point_cloud(pointcloudgraph=quotientgraph.point_cloud_graph,
                                                graph_attribute='quotient_graph_node',
                                                filename='final_split_and_merge.txt')


def oversegment_part_return_list(quotientgraph, list_quotient_node_to_work=[], average_size_cluster=20):
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
    quotientgraph.rebuild(G)

    # return of the list of nodes in pointcloudgraph to keep working with them
    list_of_nodes_pointcloudgraph = list_of_nodes
    return list_of_nodes_pointcloudgraph

def oversegment_part(quotientgraph, subgraph_riemannian, average_size_cluster=20):

    SG = subgraph_riemannian

    # get all the x,y,z coordinates in an np.array
    Xcoord = np.zeros((len(list(SG.nodes)), 3))
    for i in range(len(list(SG.nodes))):
        Xcoord[i] = SG.nodes[list(SG.nodes)[i]]['pos']

    # determine a number of cluster we would like to have
    number_of_points = len(list(SG.nodes))
    number_of_clusters = number_of_points /average_size_cluster

    # clustering via Kmeans to obtain new labels
    clustering = skc.KMeans(n_clusters=int(number_of_clusters), init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(Xcoord)
    clustering_labels = clustering.labels_[:, np.newaxis] + max(quotientgraph.nodes) + 1

    # Integration of the new labels in the quotient graph and updates
    for i in range(len(list(SG.nodes))):
        SG.nodes[list(SG.nodes)[i]]['quotient_graph_node'] = clustering_labels[i]
    quotientgraph.rebuild(quotientgraph.point_cloud_graph)



def create_subgraphs_to_work(quotientgraph, list_quotient_node_to_work=[]):
    lqg = list_quotient_node_to_work
    G = quotientgraph.point_cloud_graph

    lpcd = []
    for qnode in lqg:
        list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
        lpcd += list_of_nodes_each

    subgraph_riemannian = nx.subgraph(G, lpcd)

    return subgraph_riemannian