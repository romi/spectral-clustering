import networkx as nx




def delete_small_clusters(quotientgraph, min_number_of_element_in_a_quotient_node=50):
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
    list = []
    for node in list_of_points:
        list.pop(G.nodes[node]['quotient_graph_node'])

    list_of_clusters = list(set(list))

    return list_of_clusters