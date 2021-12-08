import scipy as sp
import numpy as np
import networkx as nx

from spectral_clustering.split_and_merge import *
from spectral_clustering.quotientgraph_operations import *

def define_leaves_by_topo(quotientgraph):
    G = quotientgraph.point_cloud_graph
    # Put the "end" nodes in a list
    list_leaves = [x for x in quotientgraph.nodes() if quotientgraph.degree(x) == 1]

    # Pick the node with the most average norm gradient (most likely to be part of the stem)
    max_norm = 0
    max_norm_node = -1
    for l in list_leaves:
        quotientgraph.nodes[l]['norm_gradient_mean'] = 0
        list_of_nodes_in_qnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == l]
        for n in list_of_nodes_in_qnode:
            quotientgraph.nodes[l]['norm_gradient_mean'] += G.nodes[n]['norm_gradient']
        quotientgraph.nodes[l]['norm_gradient_mean'] /= len(list_of_nodes_in_qnode)
        if quotientgraph.nodes[l]['norm_gradient_mean'] > max_norm:
            max_norm = quotientgraph.nodes[l]['norm_gradient_mean']
            max_norm_node = l

    # Finally I chose randomly the root among the leaves but I could choose the max_norm_node
    root = np.random.choice(list_leaves)
    # cluster again the branches by following a path
    # to_cluster = []
    # for n in self:
    #    if n not in list_leaves:
    #       to_cluster.append(n)

    # self.segment_several_nodes_using_attribute(G=self.point_cloud_graph, list_quotient_node_to_work=to_cluster,
    #                                          algo='OPTICS', para_clustering_meth=100, attribute='direction_gradient')
    # self.delete_small_clusters(min_number_of_element_in_a_quotient_node=100)

def define_semantic_classes(quotientgraph):

    # leaves : one adjacency
    list_leaves = [x for x in quotientgraph.nodes() if quotientgraph.degree(x) == 1]

    # the node with the most number of neighborhood is the stem
    degree_sorted = sorted(quotientgraph.degree, key=lambda x: x[1], reverse=True)
    stem = [degree_sorted[0][0]]

    # the rest is defined as branches
    for n in quotientgraph.nodes:
        if n in list_leaves:
            quotientgraph.nodes[n]['semantic_label'] = 'leaf'
        if n in stem:
            quotientgraph.nodes[n]['semantic_label'] = 'stem'
        elif n not in list_leaves and n not in stem:
            quotientgraph.nodes[n]['semantic_label'] = 'petiole'

def define_semantic_scores(quotientgraph, method='similarity_dist'):
    quotientgraph.compute_local_descriptors(quotientgraph.point_cloud_graph, method='all_qg_cluster', data='coords')
    quotientgraph.compute_silhouette()
    degree_sorted = sorted(quotientgraph.degree, key=lambda x: x[1], reverse=True)
    stem = [degree_sorted[0][0]]

    if method == 'condition_list':
        for n in quotientgraph.nodes:
            score_vector_leaf = [quotientgraph.nodes[n]['planarity'] > 0.5, quotientgraph.nodes[n]['linearity'] < 0.5, quotientgraph.degree(n) == 1]
            score_vector_petiole = [quotientgraph.nodes[n]['planarity'] < 0.5, quotientgraph.nodes[n]['linearity'] > 0.5, quotientgraph.degree(n) == 2, quotientgraph.nodes[n]['silhouette'] > 0.2]
            score_vector_stem = [quotientgraph.nodes[n]['planarity'] < 0.5, quotientgraph.nodes[n]['linearity'] > 0.5, quotientgraph.degree(n) == degree_sorted[0][1]]
            quotientgraph.nodes[n]['score_leaf'] = score_vector_leaf.count(True) / len(score_vector_leaf)
            quotientgraph.nodes[n]['score_petiole'] = score_vector_petiole.count(True) / len(score_vector_petiole)
            quotientgraph.nodes[n]['score_stem'] = score_vector_stem.count(True) / len(score_vector_stem)

    if method == 'similarity_dist':
        def smoothstep(x):
            if x <= 0:
                res = 0
            elif x >= 1:
                res = 1
            else:
                res = 3 * pow(x, 2) - 2 * pow(x, 3)

            return res

        def identity(x):
            if x <= 0:
                res = 0
            elif x >= 1:
                res = 1
            else:
                res = x

            return res

        score_vector_leaf_ref = [0.81, 0, -0.1]
        score_vector_linea_ref = [0, 1, 0.3]
        for n in quotientgraph.nodes:
            # score_vector_n = sk.preprocessing.normalize(np.asarray([QG.nodes[n]['planarity'], QG.nodes[n]['linearity']]).reshape(1,len(score_vector_leaf_ref)), norm='l2')
            score_vector_n = [quotientgraph.nodes[n]['planarity'], quotientgraph.nodes[n]['linearity'], quotientgraph.nodes[n]['silhouette']]
            print(score_vector_n)
            d1_leaf = sp.spatial.distance.euclidean(score_vector_leaf_ref, score_vector_n)
            # print(d1_leaf)
            d2_linea = sp.spatial.distance.euclidean(score_vector_linea_ref, score_vector_n)
            # print(d2_leaf)
            f1 = 1 - d1_leaf
            f2 = 1 - d2_linea
            quotientgraph.nodes[n]['score_leaf'] = identity(f1)
            quotientgraph.nodes[n]['score_petiole'] = identity(f2)

def minimum_spanning_tree_quotientgraph_semantics(quotientgraph):
    define_semantic_classes(quotientgraph)
    for (u,v) in quotientgraph.edges:
        if quotientgraph.nodes[u]['semantic_label'] == quotientgraph.nodes[v]['semantic_label']:
            quotientgraph.edges[u, v]['semantic_weight'] = 50
        else:
            quotientgraph.edges[u, v]['semantic_weight'] = 1
    compute_quotientgraph_mean_attribute_from_points(quotientgraph.point_cloud_graph, quotientgraph, attribute='quotient_graph_node')
    QG_t2 = nx.minimum_spanning_tree(quotientgraph, algorithm='kruskal', weight='semantic_weight')

    return QG_t2



def determination_main_stem_shortest_paths_improved(QG, ptsource, list_of_linear_QG_nodes, angle_to_stop=45, minimumpoint=5):
    import copy
    sub_riemanian = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=list_of_linear_QG_nodes)
    G = QG.point_cloud_graph
    energy_to_stop = 1 - np.cos(np.radians(angle_to_stop))

    dict = nx.single_source_dijkstra_path_length(sub_riemanian, ptsource, weight='weight')
    ptarrivee = max(dict.items(), key=operator.itemgetter(1))[0]
    segmsource = nx.dijkstra_path(sub_riemanian, ptsource, ptarrivee, weight='weight')
    list_clusters_qg_traversed = []
    for i in segmsource:
        n = G.nodes[i]['quotient_graph_node']
        list_clusters_qg_traversed.append(n)

    def count_to_dict(lst):
        return {k: lst.count(k) for k in lst}
    dict = count_to_dict(list_clusters_qg_traversed)
    list_stem2=[]
    for key, value in dict.items():
        if value > minimumpoint:
            list_stem2.append(key)
    list_stem1 = list(dict.fromkeys(list_clusters_qg_traversed))

    quotient_graph_compute_direction_mean(QG)
    quotient_graph_compute_direction_standard_deviation(QG)
    final_list_stem = copy.deepcopy(list_stem2)
    for node in list_stem2:
      if QG.nodes[node]['dir_gradient_angle_mean'] < np.cos(np.radians(angle_to_stop)):
          final_list_stem.remove(node)
          QG.nodes[node]['viterbi_class'] = -1

    nodi = 0
    final_list_stem_clean = []
    while nodi <= (len(final_list_stem)-2):
        e = (final_list_stem[nodi], final_list_stem[nodi + 1])
        v1 = QG.nodes[e[0]]['dir_gradient_mean']
        v2 = QG.nodes[e[1]]['dir_gradient_mean']
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        energy = 1 - dot
        print(e)
        print(energy)
        if energy > energy_to_stop:
            final_list_stem_clean = final_list_stem[:nodi+1]
            nodi = len(final_list_stem) + 2
        else:
            final_list_stem_clean = final_list_stem
            nodi += 1

    final_list_stem_clean.insert(0, G.nodes[ptsource]['quotient_graph_node'])
    for n in final_list_stem_clean:
        QG.nodes[n]['viterbi_class'] = 3

    transfer_quotientgraph_infos_on_riemanian_graph(QG=QG, info='viterbi_class')

    return final_list_stem_clean