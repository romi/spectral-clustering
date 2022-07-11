import scipy as sp
import numpy as np
import networkx as nx
import copy as copy

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

def determination_main_stem(QG, list_of_linear_QG_nodes, stemroot, list_leaves, angle_to_stop=45, new_viterbi_class_number=3):
    QG.compute_direction_info(list_leaves=list_leaves)
    G = QG.point_cloud_graph
    energy_to_stop = 1 - np.cos(np.radians(angle_to_stop))
    keepgoing = True
    node_stem = stemroot
    list_stem = []
    list_stem.append(node_stem)
    while keepgoing:
        list_energies = []
        list_nodes = []
        print(node_stem)
        for n in QG[node_stem]:
            if n not in list_stem:
                list_energies.append(QG.edges[(n, node_stem)]['energy_dot_product'])
                list_nodes.append(n)
        node = list_energies.index(min(list_energies))
        node_stem = list_nodes[node]
        if list_nodes:
            if min(list_energies) > energy_to_stop or node_stem not in list_of_linear_QG_nodes:
                keepgoing = False
            else:
                list_stem.append(node_stem)
        else:
            keepgoing is False

    for n in list_stem:
        QG.nodes[n]['viterbi_class'] = new_viterbi_class_number
        list_gnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        for i in list_gnode:
            G.nodes[i]['viterbi_class'] = new_viterbi_class_number

def determination_main_stem_shortest_paths(QG, list_of_linear_QG_nodes):
    sub_riemanian = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=list_of_linear_QG_nodes)
    G = QG.point_cloud_graph
    segmsource, ptsource, ptarrivee = initdijkstra(sub_riemanian)
    list_clusters_qg_traversed = []
    for i in segmsource:
        n = G.nodes[i]['quotient_graph_node']
        list_clusters_qg_traversed.append(n)
    final_list_stem = list(set(list_clusters_qg_traversed))
    for n in final_list_stem:
        QG.nodes[n]['viterbi_class'] = 3
        list_gnode = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        for i in list_gnode:
            G.nodes[i]['viterbi_class'] = 3
    return final_list_stem


def determination_main_stem_shortest_paths_improved(QG, ptsource, list_of_linear_QG_nodes, angle_to_stop=45, minimumpoint=5, classnumberstem=3, classnumberanomaly=3):
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
          QG.nodes[node]['viterbi_class'] = classnumberanomaly

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
        QG.nodes[n]['viterbi_class'] = classnumberstem

    transfer_quotientgraph_infos_on_riemanian_graph(QG=QG, info='viterbi_class')

    return final_list_stem_clean

def stem_detection_with_quotite_leaves(QG, list_leaves3, list_apex, list_of_linear, root_point_riemanian, new_class_stem=3):
    calcul_quotite_feuilles(QG, list_leaves3, list_of_linear, root_point_riemanian)
    G = QG.point_cloud_graph
    list_length = []
    if list_apex:
        list_end = list_apex
    else:
        list_end = list_leaves3

    for leaf in list_end:
        sub_qg = nx.subgraph(QG, list_of_linear + [leaf] +  [G.nodes[root_point_riemanian]["quotient_graph_node"]])
        display_and_export_quotient_graph_matplotlib(quotient_graph=sub_qg, node_sizes=20,
                                                     filename="sub_graphsclass_feuilles_sur_noeuds"+str(leaf),
                                                     data_on_nodes='viterbi_class', data=True,
                                                     attributekmeans4clusters=False)

        if nx.has_path(sub_qg, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"]):
            path = nx.dijkstra_path(sub_qg, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"],
                                    weight='distance_centroides')
        else:
            path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"],
                                    weight='distance_centroides')
        pathleafquantity = []
        paths = []
        for p in path:
            pathleafquantity.append(QG.nodes[p]['quotite_feuille_n'])
            paths.append(path)
        print(pathleafquantity)
        print(set(pathleafquantity))
        lengthpath = sum(list(set(pathleafquantity)))
        list_length.append(lengthpath)
        print(path)
        print(lengthpath)

    leafend = list_end[list_length.index(max(list_length))]
    #path_final = paths[]

    list_stem_full = nx.dijkstra_path(QG, G.nodes[root_point_riemanian]["quotient_graph_node"], leafend, weight='distance_centroides')
    list_stem = list_stem_full
    print('list_stem')
    print(list_stem)
    previous = 0
    """
    maxim = QG.nodes[G.nodes[root_point_riemanian]["quotient_graph_node"]]['quotite_feuille_n']
    for n in list_stem_full:
        if maxim >= QG.nodes[n]['quotite_feuille_n']:
            maxim = QG.nodes[n]['quotite_feuille_n']
            prev = n
        else:
            list_stem.pop(prev)
    """
    del list_stem[-1]
    for n in list_stem:
        #if n not in list_leaves3:
        QG.nodes[n]["viterbi_class"] = new_class_stem


    export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_')
    return list_stem

def differenciate_apex_limb(QG, attribute_class = 'viterbi_class', number_leaves_limb = 1, new_apex_class = 4):
    QG.count_local_extremum_of_Fiedler()
    list_limbs = [x for x, y in QG.nodes(data=True) if y[attribute_class] == number_leaves_limb]
    for l in list_limbs:
        if QG.nodes[l]['number_of_local_Fiedler_extremum'] > 1:
            QG.nodes[l][attribute_class] = new_apex_class

def topology_control(quotient_graph, attribute_class_control ='viterbi_class', class_stem = 3, class_petiols = 5, class_limb = 1, class_apex= 4, error_norm = 10, error_dir=11):
    QG = quotient_graph
    # check leaves connected to stem
    stem = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == class_stem]
    for n in QG[stem[0]]:
        if QG.nodes[n][attribute_class_control] == class_limb or QG.nodes[n][attribute_class_control] == class_apex:
            QG.nodes[n][attribute_class_control] = error_norm

    petiols = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == class_petiols]
    for n in petiols:
        if QG.degree(n) >= 3:
            QG.nodes[n][attribute_class_control] = error_dir

def treat_topology_error(quotient_graph, attribute_class_control='viterbi_class',error = 5, way_to_treat='direction', number_of_cluster_tested=20):
    QG = quotient_graph
    nodes_to_treat = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == error]

    if way_to_treat == 'direction':
        resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=nodes_to_treat, number_of_cluster_tested=number_of_cluster_tested,
                                          attribute='direction_gradient', number_attribute=3, standardization=False)
    elif way_to_treat == 'norm':
        resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=nodes_to_treat, number_of_cluster_tested=number_of_cluster_tested,
                                          attribute='norm_gradient', number_attribute=1, standardization=False, numer = 1000)

def weight_with_semantic(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_linear=0, class_apex=4,
                         weight_limb_limb=10000, weight_apex_apex=10000, weight_apex_mainstem=10000,
                         weight_limb_apex=10000, weight_limb_mainstem=10000, weight_linear_mainstem=0, weight_linear_linear=0,
                         weight_linear_limb=0, weight_linear_apex=0):
    for e in QG.edges():
        QG.edges[e]['weight_sem_paths'] = np.nan
        c1 = QG.nodes[e[0]][class_attribute]
        c2 = QG.nodes[e[1]][class_attribute]
        l = [c1,c2]
        if set(l) == set([class_limb,class_limb]):
            QG.edges[e]['weight_sem_paths']= weight_limb_limb
        if set(l) == set([class_limb,class_mainstem]):
            QG.edges[e]['weight_sem_paths']= weight_limb_mainstem
        if set(l) == set([class_limb,class_apex]):
            QG.edges[e]['weight_sem_paths']= weight_limb_apex
        if set(l) == set([class_limb,class_linear]):
            QG.edges[e]['weight_sem_paths']= weight_linear_limb

        if set(l) == set([class_apex,class_apex]):
            QG.edges[e]['weight_sem_paths']= weight_apex_apex
        if set(l) == set([class_apex,class_mainstem]):
            QG.edges[e]['weight_sem_paths']= weight_apex_mainstem
        if set(l) == set([class_apex,class_linear]):
            QG.edges[e]['weight_sem_paths']= weight_linear_apex

        if set(l) == set([class_linear, class_linear]):
            QG.edges[e]['weight_sem_paths'] = weight_linear_linear
        if set(l) == set([class_linear, class_mainstem]):
            QG.edges[e]['weight_sem_paths'] = weight_linear_mainstem


def shortest_paths_from_limbs_det_petiol(QG, root_point_riemanian, class_attribute='viterbi_class', weight='weight_sem_paths', class_limb=1, class_linear=0, class_mainstem=3, new_class_petiol=5):
    G = QG.point_cloud_graph

    list_limb = [x for x, y in QG.nodes(data=True) if y[class_attribute] == class_limb]
    for leaf in list_limb:
        path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"], weight=weight)
        for n in path:
            if QG.nodes[n][class_attribute] == class_linear:
                QG.nodes[n][class_attribute] = new_class_petiol
        for i in range(len(path) - 1):
            if QG.has_edge(path[i], path[i + 1]):
                e = (path[i], path[i + 1])
            else:
                e = (path[i + 1], path[i])
            QG.edges[e]['useful_path_shortest'] = True

def maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5, class_apex=4, class_branch=6,
                         weight_petiol_petiol=0, weight_petiol_apex=10000, weight_branch_limb=10000):
    for e in QG.edges():
        c1 = QG.nodes[e[0]][class_attribute]
        c2 = QG.nodes[e[1]][class_attribute]
        l = [c1,c2]
        if set(l) == set([class_branch, class_limb]):
            QG.edges[e]['weight_sem_paths'] = weight_branch_limb


def shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class', weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3, new_class_branch=6):
    G = QG.point_cloud_graph
    list_apex = [x for x, y in QG.nodes(data=True) if y[class_attribute] == class_apex]
    for e in QG.edges():
        QG.edges[e]['useful_path_shortest'] = False
    for leaf in list_apex:
        path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"], weight=weight)
        print(path)
        for n in path:
            if QG.nodes[n][class_attribute] == class_branch:
                QG.nodes[n][class_attribute] = new_class_branch
        for i in range(len(path) - 1):
            if QG.has_edge(path[i], path[i + 1]):
                e = (path[i], path[i + 1])
            else:
                e = (path[i + 1], path[i])
            QG.edges[e]['useful_path_shortest'] = True

def merge_remaining_clusters(quotientgraph, remaining_clusters_class=0, class_attribute='viterbi_class'):
    list_clusters = [x for x, y in quotientgraph.nodes(data=True) if y[class_attribute] == remaining_clusters_class]
    print(list_clusters)
    G = quotientgraph.point_cloud_graph

    for u in list_clusters:
        count = quotientgraph.compute_quotientgraph_metadata_on_a_node_interclass(u)
        print(count)
        max_class_o = max(count, key=count.get)
        max_class = max_class_o
        # deal with possibility of max_class_ another class to merge. In that case, second best linked is chosent
        # if only linked to another class to merge, merge with this one.
        while max_class in list_clusters and bool(count) and len(count)>1:
            count.pop(max_class)
            max_class = max(count, key=count.get)
        if bool(count) is False or len(count) == 1:
            max_class = max_class_o

        new_cluster = max_class
        print(new_cluster)
        list_G_nodes_to_change = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == u]
        for g in list_G_nodes_to_change:
            G.nodes[g]['quotient_graph_node'] = new_cluster

        dict1 = {}
        for e in quotientgraph.edges(new_cluster):
            dict1[e] = quotientgraph.edges[e]['useful_path_shortest']
        quotientgraph = nx.contracted_nodes(quotientgraph, new_cluster, u, self_loops=False)
        quotientgraph.point_cloud_graph = G
        dict2 = {}
        for e in quotientgraph.edges(new_cluster):
            dict2[e] = quotientgraph.edges[e]['useful_path_shortest']
            if e in dict1.keys():
                quotientgraph.edges[e]['useful_path_shortest'] = dict1[e]
        print(dict1)
        print(dict2)

    return quotientgraph

def remove_edges_useful_paths(QG):
    FG = copy.deepcopy(QG)
    d = FG.edges()
    for e in d:
        if FG.edges[e]['useful_path_shortest'] is False:
            QG.remove_edge(*e)
