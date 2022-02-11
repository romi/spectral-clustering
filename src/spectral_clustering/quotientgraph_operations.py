import networkx as nx
import operator
from spectral_clustering.display_and_export import *

def delete_small_clusters(quotientgraph, min_number_of_element_in_a_quotient_node=50):
    """Compute a new clustering by treating one node of the QuotientGraph after the other and changing the class of the
    underlying nodes of the PointCloudGraph associated.

    Parameters
    ----------
    quotientgraph : QuotientGraph class object.
    min_number_of_element_in_a_quotient_node

    Returns
    -------
    Nothing
    Update the quotientgraph by deleting the nodes with intra_class_node_number < min_number_of_element_in_a_quotient_node
    Update the PointCloudGraph associated to the QuotientGraph by changing the points from the small class to the most
    bigger adjacent class.

    """

    G = quotientgraph.point_cloud_graph
    nodes_to_remove = []
    for u in quotientgraph.nodes:
        if quotientgraph.nodes[u]['intra_class_node_number'] < min_number_of_element_in_a_quotient_node:
            adjacent_clusters = [n for n in quotientgraph[u]]
            max_number_of_nodes_in_adjacent_clusters = 0
            for i in range(len(adjacent_clusters)):
                if quotientgraph.nodes[adjacent_clusters[i]]['intra_class_node_number'] > max_number_of_nodes_in_adjacent_clusters:
                    max_number_of_nodes_in_adjacent_clusters = quotientgraph.nodes[adjacent_clusters[i]]['intra_class_node_number']
                    new_cluster = adjacent_clusters[i]
            # Opération de fusion du petit cluster avec son voisin le plus conséquent.
            # Mise à jour des attributs de la grande classe et suppression de la petite classe
            quotientgraph.nodes[new_cluster]['intra_class_node_number'] += quotientgraph.nodes[u]['intra_class_node_number']
            quotientgraph.nodes[new_cluster]['intra_class_edge_weight'] += (quotientgraph.nodes[u]['intra_class_edge_weight']
                                                                   + quotientgraph.edges[new_cluster, u][
                                                                       'inter_class_edge_weight'])
            quotientgraph.nodes[new_cluster]['intra_class_edge_number'] += (quotientgraph.nodes[u]['intra_class_edge_number']
                                                                   + quotientgraph.edges[new_cluster, u][
                                                                       'inter_class_edge_number'])

            # Mise à jour du lien avec le PointCloudGraph d'origine
            for v in G.nodes:
                if G.nodes[v]['quotient_graph_node'] == u:
                    G.nodes[v]['quotient_graph_node'] = new_cluster

            # Mise à jour des edges
            for i in range(len(adjacent_clusters)):
                if quotientgraph.has_edge(new_cluster, adjacent_clusters[i]) is False:
                    quotientgraph.add_edge(new_cluster, adjacent_clusters[i],
                                           inter_class_edge_weight=quotientgraph.edges[u, adjacent_clusters[i]]['inter_class_edge_weight'],
                                           inter_class_edge_number=quotientgraph.edges[u, adjacent_clusters[i]]['inter_class_edge_number'])
                elif quotientgraph.has_edge(new_cluster, adjacent_clusters[i]) and new_cluster != adjacent_clusters[i]:
                    quotientgraph.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                        quotientgraph.edges[u, adjacent_clusters[i]]['inter_class_edge_weight']
                    quotientgraph.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                        quotientgraph.edges[u, adjacent_clusters[i]]['inter_class_edge_number']
            nodes_to_remove.append(u)

    quotientgraph.remove_nodes_from(nodes_to_remove)
    
    
def update_quotient_graph_attributes_when_node_change_cluster(quotientgraph, old_cluster, new_cluster, node_to_change):
    """When the node considered change of cluster, this function updates all the attributes associated to the
    quotient graph.

    Parameters
    ----------
    node_to_change : int
    The node considered and changed in an other cluster
    old_cluster : int
    The original cluster of the node
    new_cluster : int
    The new cluster of the node
    G : PointCloudGraph

    Returns
    -------
    """

    G = quotientgraph.point_cloud_graph
    # update Quotient Graph attributes
    # Intra_class_node_number
    #print(old_cluster)
    #print(new_cluster)
    quotientgraph.nodes[old_cluster]['intra_class_node_number'] += -1
    quotientgraph.nodes[new_cluster]['intra_class_node_number'] += 1

    # intra_class_edge_weight, inter_class_edge_number
    for ng in G[node_to_change]:
        if old_cluster != G.nodes[ng]['quotient_graph_node'] and new_cluster == G.nodes[ng]['quotient_graph_node']:
            quotientgraph.nodes[new_cluster]['intra_class_edge_weight'] += G.edges[ng, node_to_change]['weight']
            quotientgraph.nodes[new_cluster]['intra_class_edge_number'] += 1
            quotientgraph.edges[old_cluster, new_cluster]['inter_class_edge_weight'] += - G.edges[ng, node_to_change][
                'weight']
            quotientgraph.edges[old_cluster, new_cluster]['inter_class_edge_number'] += - 1

        if old_cluster == G.nodes[ng]['quotient_graph_node'] and new_cluster != G.nodes[ng]['quotient_graph_node']:
            quotientgraph.nodes[old_cluster]['intra_class_edge_weight'] += -G.edges[ng, node_to_change]['weight']
            quotientgraph.nodes[old_cluster]['intra_class_edge_number'] += -1
            cluster_adj = G.nodes[ng]['quotient_graph_node']
            if quotientgraph.has_edge(new_cluster, cluster_adj) is False:
                quotientgraph.add_edge(new_cluster, cluster_adj, inter_class_edge_weight=None, inter_class_edge_number=None)
            quotientgraph.edges[new_cluster, cluster_adj]['inter_class_edge_weight'] += G.edges[ng, node_to_change]['weight']
            quotientgraph.edges[new_cluster, cluster_adj]['inter_class_edge_number'] += 1

        if old_cluster != G.nodes[ng]['quotient_graph_node'] and new_cluster != G.nodes[ng]['quotient_graph_node']:
            cluster_adj = G.nodes[ng]['quotient_graph_node']
            if quotientgraph.has_edge(new_cluster, cluster_adj) is False:
                quotientgraph.add_edge(new_cluster, cluster_adj,
                                       inter_class_edge_weight=G.edges[ng, node_to_change]['weight'],
                                       inter_class_edge_number=1)
            else:
                quotientgraph.edges[new_cluster, cluster_adj]['inter_class_edge_weight'] += G.edges[ng, node_to_change][
                    'weight']
                quotientgraph.edges[new_cluster, cluster_adj]['inter_class_edge_number'] += 1
            if new_cluster != old_cluster:
                quotientgraph.edges[old_cluster, cluster_adj]['inter_class_edge_weight'] += - \
                    G.edges[ng, node_to_change]['weight']
                quotientgraph.edges[old_cluster, cluster_adj]['inter_class_edge_number'] += - 1


def check_connectivity_of_modified_cluster(pointcloudgraph, old_cluster, new_cluster, node_to_change):
    """When the node considered change of cluster, this function checks if the old cluster of the node is still
    connected or forms two different parts.
    This function is to use after the change occurred in the distance-based graph

    Parameters
    ----------
    node_to_change : int
    The node considered and changed in an other cluster
    old_cluster : int
    The original cluster of the node
    new_cluster : int
    The new cluster of the node
    G : PointCloudGraph

    Returns
    -------
    Boolean
    True if the connectivity was conserved
    """
    G = pointcloudgraph
    # Build a dictionary which associate each node of the PointCloudGraph with its cluster/node in the QuotientGraph
    d = dict(G.nodes(data='quotient_graph_node', default=None))
    # Extract points of the old_cluster
    points_of_interest = [key for (key, value) in d.items() if value == old_cluster]
    # Handling the case where this cluster has completely disappeared
    if not points_of_interest:
        result = True
    else:
        # Creation of the subgraph and test
        subgraph = G.subgraph(points_of_interest)
        result = nx.is_connected(subgraph)

    return result

#def rebuilt_topological_quotient_graph


def collect_quotient_graph_nodes_from_pointcloudpoints(quotient_graph, list_of_points=[]):
    G = quotient_graph.point_cloud_graph
    listi = []
    for node in list_of_points:
        listi.append(G.nodes[node]['quotient_graph_node'])

    list_of_clusters = list(set(listi))

    return list_of_clusters

def collect_quotient_graph_nodes_from_pointcloudpoints_majority(quotient_graph, list_of_points=[]):
    G = quotient_graph.point_cloud_graph
    listi = []

    for qgnode in quotient_graph.nodes:
        countyes = 0
        countno = 0
        list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qgnode]
        for n in list_of_nodes_each:
            if n in list_of_points:
                countyes += 1
            else:
                countno += 1
        if countyes > countno:
            listi.append(qgnode)

    list_of_clusters = list(set(listi))

    return list_of_clusters

def compute_quotientgraph_mean_attribute_from_points(G, QG, attribute='clustering_labels'):
    for n in QG.nodes:
        moy = 0
        list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        for e in list_of_nodes:
            moy += G.nodes[e][attribute]
        QG.nodes[n][attribute+'_mean'] = moy/len(list_of_nodes)

def merge_similar_class_QG_nodes(QG, attribute='viterbi_class', export=True):
    G = QG.point_cloud_graph
    energy_similarity = 0
    for e in QG.edges():
        v1 = QG.nodes[e[0]][attribute]
        v2 = QG.nodes[e[1]][attribute]
        if v1 == v2:
            QG.edges[e]['similarity_class'] = 1
        else:
            QG.edges[e]['similarity_class'] = 0
        energy_similarity += QG.edges[e]['similarity_class']

    energy_per_edges = nx.get_edge_attributes(QG, 'similarity_class')
    edge_to_delete = max(energy_per_edges.items(), key=operator.itemgetter(1))[0]
    energy_max = energy_per_edges[edge_to_delete]
    while energy_max != 0:
        n1 = QG.nodes[edge_to_delete[0]]['intra_class_node_number']
        n2 = QG.nodes[edge_to_delete[1]]['intra_class_node_number']
        viterbi = QG.nodes[edge_to_delete[0]][attribute]
        list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == edge_to_delete[1]]
        for j in range(len(list_of_nodes)):
            G.nodes[list_of_nodes[j]]['quotient_graph_node'] = edge_to_delete[0]

        QG = nx.contracted_nodes(QG, edge_to_delete[0], edge_to_delete[1], self_loops=False)
        QG.point_cloud_graph = G
        QG.nodes[edge_to_delete[0]]['intra_class_node_number'] = n1 + n2
        QG.nodes[edge_to_delete[0]][attribute] = viterbi
        for e in QG.edges(edge_to_delete[0]):
            v1 = QG.nodes[e[0]][attribute]
            v2 = QG.nodes[e[1]][attribute]
            if v1 == v2:
                QG.edges[e]['similarity_class'] = 1
            else:
                QG.edges[e]['similarity_class'] = 0

        energy_per_edges = nx.get_edge_attributes(QG, 'similarity_class')
        edge_to_delete = max(energy_per_edges.items(), key=operator.itemgetter(1))[0]
        energy_max = energy_per_edges[edge_to_delete]
    #QG.rebuild(G=G)
    if export is True:
        export_some_graph_attributes_on_point_cloud(pointcloudgraph=QG.point_cloud_graph,
                                                    graph_attribute='quotient_graph_node',
                                                    filename='Merge_after_viterbi.txt')
        display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
                                                     filename="merge_quotient_graph_viterbi", data_on_nodes='viterbi_class',
                                                     data=True, attributekmeans4clusters=False)
    return QG

def merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass = [1]):
    G = QG.point_cloud_graph
    energy_similarity = 0
    for e in QG.edges():
        v1 = QG.nodes[e[0]][attribute]
        v2 = QG.nodes[e[1]][attribute]
        if v1 == v2 and v1 in viterbiclass :
            QG.edges[e]['similarity_class'] = 1
        else:
            QG.edges[e]['similarity_class'] = 0
        energy_similarity += QG.edges[e]['similarity_class']

    energy_per_edges = nx.get_edge_attributes(QG, 'similarity_class')
    edge_to_delete = max(energy_per_edges.items(), key=operator.itemgetter(1))[0]
    energy_max = energy_per_edges[edge_to_delete]
    while energy_max == 1:
        n1 = QG.nodes[edge_to_delete[0]]['intra_class_node_number']
        n2 = QG.nodes[edge_to_delete[1]]['intra_class_node_number']
        viterbi = QG.nodes[edge_to_delete[0]][attribute]
        list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == edge_to_delete[1]]
        for j in range(len(list_of_nodes)):
            G.nodes[list_of_nodes[j]]['quotient_graph_node'] = edge_to_delete[0]

        QG = nx.contracted_nodes(QG, edge_to_delete[0], edge_to_delete[1], self_loops=False)
        QG.point_cloud_graph = G
        QG.nodes[edge_to_delete[0]]['intra_class_node_number'] = n1 + n2
        QG.nodes[edge_to_delete[0]][attribute] = viterbi
        for e in QG.edges(edge_to_delete[0]):
            v1 = QG.nodes[e[0]][attribute]
            v2 = QG.nodes[e[1]][attribute]
            if v1 == v2 and v1 in viterbiclass:
                QG.edges[e]['similarity_class'] = 1
            else:
                QG.edges[e]['similarity_class'] = 0

        energy_per_edges = nx.get_edge_attributes(QG, 'similarity_class')
        edge_to_delete = max(energy_per_edges.items(), key=operator.itemgetter(1))[0]
        energy_max = energy_per_edges[edge_to_delete]
     # QG.rebuild(G=G)

    return QG


def quotient_graph_compute_direction_mean(quotient_graph):
    G = quotient_graph.point_cloud_graph
    for l in quotient_graph:
        quotient_graph.nodes[l]['dir_gradient_mean'] = 0
        list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l]
        for n in list_of_nodes_in_qnode:
            quotient_graph.nodes[l]['dir_gradient_mean'] += G.nodes[n]['direction_gradient']
        quotient_graph.nodes[l]['dir_gradient_mean'] /= len(list_of_nodes_in_qnode)


def quotient_graph_compute_direction_standard_deviation(quotient_graph, mean='dir_gradient_mean'):
    G = quotient_graph.point_cloud_graph
    for l in quotient_graph:
        quotient_graph.nodes[l]['dir_gradient_angle_mean'] = 0
        quotient_graph.nodes[l]['dir_gradient_stdv'] = 0
        list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l]
        for n in list_of_nodes_in_qnode:
            v1 = quotient_graph.nodes[l][mean]
            v2 = G.nodes[n]['direction_gradient']
            G.nodes[n]['dir_gradient_angle'] = abs(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
            quotient_graph.nodes[l]['dir_gradient_angle_mean'] += G.nodes[n]['dir_gradient_angle']
        quotient_graph.nodes[l]['dir_gradient_angle_mean'] /= len(list_of_nodes_in_qnode)
        for n in list_of_nodes_in_qnode:
            v1 = quotient_graph.nodes[l][mean]
            v2 = G.nodes[n]['direction_gradient']
            quotient_graph.nodes[l]['dir_gradient_stdv'] += np.power(quotient_graph.nodes[l]['dir_gradient_angle_mean'] - G.nodes[n]['dir_gradient_angle'], 2)
        quotient_graph.nodes[l]['dir_gradient_stdv'] /= len(list_of_nodes_in_qnode)


def transfer_quotientgraph_infos_on_riemanian_graph(QG,info='viterbi_class'):
    G = QG.point_cloud_graph
    for n in G.nodes:
        G.nodes[n][info] = QG.nodes[G.nodes[n]['quotient_graph_node']][info]


def calcul_quotite_feuilles(QG, list_leaves, list_of_linear, root_point_riemanian):
    G = QG.point_cloud_graph
    for e in QG.edges:
        QG.edges[e]['quotite_feuille_e'] = 0
        QG.edges[e]['useful_path'] = 50
    for n in QG.nodes:
        QG.nodes[n]['quotite_feuille_n'] = 0

    for leaf in list_leaves:
        sub_qg = nx.subgraph(QG, list_of_linear + [leaf])
        if nx.has_path(sub_qg, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"]):
            path = nx.dijkstra_path(sub_qg, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"], weight='distance_centroides')
        else:
            path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"], weight='distance_centroides')
        for n in path:
            QG.nodes[n]['quotite_feuille_n'] += 1
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            QG.edges[e]['quotite_feuille_e'] += 1
            QG.edges[e]['useful_path'] = 0


