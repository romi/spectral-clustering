########### Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from treex import *

import scipy.sparse as spsp
import scipy as sp
import sklearn as sk
import spectral_clustering.similarity_graph as sgk
import open3d as open3d
import spectral_clustering.point_cloud_graph as kpcg

import time
import copy

from collections import Counter

from spectral_clustering.point_cloud_graph import *
from spectral_clustering.quotient_graph import *
from spectral_clustering.topological_energy import *
from spectral_clustering.segmentation_algorithms import *
from spectral_clustering.split_and_merge import *
from spectral_clustering.display_and_export import *
from spectral_clustering.quotientgraph_semantics import *
from spectral_clustering.Viterbi import *
from spectral_clustering.dijkstra_segmentation import *
from spectral_clustering.quotientgraph_operations import *

from importlib import reload

begin = time.time()

pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenos/cheno_A_2021_04_19.ply", format='ply')
r = 18
SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r)
G = PointCloudGraph(SimG)
G.pcd = pcdfinal
# LOWER POINT, USER INPUT
root_point_riemanian = 36847
# In cloudcompare : Edit > ScalarFields > Add point indexes on SF

G.compute_graph_eigenvectors()

export_fiedler_vector_on_pointcloud(G, filename="pcd_vp2.txt")
G.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
clusters_centers = G.clustering_by_kmeans_using_gradient_norm(export_in_labeled_point_cloud=True, number_of_clusters=4)
QG = QuotientGraph()
QG.build_from_pointcloudgraph(G, G.kmeans_labels_gradient)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_attribute.txt")
time1 = time.time()


label_leaves = select_minimum_centroid_class(clusters_centers)
list_leaves1 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves)

list_of_linear = []
for n in QG.nodes:
    QG.nodes[n]["viterbi_class"] = np.nan
for n in QG.nodes:
    if n in list_leaves1:
        QG.nodes[n]["viterbi_class"] = 1
    if n not in list_leaves1:
        list_of_linear.append(n)
        QG.nodes[n]["viterbi_class"] = 0
list_leaves1.remove(G.nodes[root_point_riemanian]["quotient_graph_node"])
list_of_linear.append(G.nodes[root_point_riemanian]["quotient_graph_node"])
G.nodes[root_point_riemanian]["viterbi_class"] = 0

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm1')



list_leaves_point = []
for qnode in list_leaves1:
    list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
    list_leaves_point += list_of_nodes_each

list_of_linear = []
for n in QG.nodes:
    if n not in list_leaves1:
        list_of_linear.append(n)



resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = list_of_linear, number_of_cluster_tested = 10, attribute='direction_gradient', number_attribute=3, standardization=False)
QG.rebuild(QG.point_cloud_graph)

export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="reseg_linear.txt")

list_leaves2 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves)

list_of_linear = []
for n in QG.nodes:
    QG.nodes[n]["viterbi_class"] = np.nan
for n in QG.nodes:
    if n in list_leaves2:
        QG.nodes[n]["viterbi_class"] = 1
    if n not in list_leaves2:
        list_of_linear.append(n)
        QG.nodes[n]["viterbi_class"] = 0
list_leaves2.remove(G.nodes[root_point_riemanian]["quotient_graph_node"])
list_of_linear.append(G.nodes[root_point_riemanian]["quotient_graph_node"])
G.nodes[root_point_riemanian]["viterbi_class"] = 0

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm2')

QG.compute_direction_info(list_leaves=list_leaves2)
opti_energy_dot_product(quotientgraph=QG, subgraph_riemannian=G, angle_to_stop=30, export_iter=True, list_leaves=list_leaves2)

list_leaves3 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves)
list_of_linear = []
for n in QG.nodes:
    QG.nodes[n]["viterbi_class"] = np.nan
for n in QG.nodes:
    if n in list_leaves3:
        QG.nodes[n]["viterbi_class"] = 1
    if n not in list_leaves3:
        list_of_linear.append(n)
        QG.nodes[n]["viterbi_class"] = 0
if G.nodes[root_point_riemanian]["quotient_graph_node"] in list_leaves3:
    list_leaves3.remove(G.nodes[root_point_riemanian]["quotient_graph_node"])
list_of_linear.append(G.nodes[root_point_riemanian]["quotient_graph_node"])
G.nodes[root_point_riemanian]["viterbi_class"] = 0

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm3')


# arbre couvrant sur QG.
# Calcul des centroîdes, stockage dans un attribut 'centroide_coordinates'
QG.ponderate_with_coordinates_of_points()
delete_small_clusters(QG)
list_leaves3 = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
list_of_linear = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 0]


calcul_quotite_feuilles(QG, list_leaves3, list_of_linear, root_point_riemanian)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='quotite_feuille_n', name='quotite_feuille_sur_cluster')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotite_feuilles_sur_noeuds", data_on_nodes='quotite_feuille_n', data=True, attributekmeans4clusters = False)

#Selection du plus long des plus courts chemins avec la pondération sur les edges en fonction de la quotité de feuilles
stem_detection_with_quotite_leaves(QG, list_leaves3, list_of_linear, root_point_riemanian)

QG = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[3])
QG.point_cloud_graph = G
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="QG_merge_tige", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)

list_leaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
list_notleaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] != 1]
calcul_quotite_feuilles(QG, list_leaves, list_notleaves, root_point_riemanian)
for n in list_leaves:
    QG.nodes[n]["quotite_feuille_n"] = -1



G.find_local_extremum_of_Fiedler()
export_some_graph_attributes_on_point_cloud(G,
                                            graph_attribute="extrem_local_Fiedler",
                                            filename="pcd_exttrem_local.txt")

QG.count_local_extremum_of_Fiedler()
export_quotient_graph_attribute_on_point_cloud(QG, attribute='number_of_local_Fiedler_extremum', name='number_extrema_QG_cluster')


differenciate_apex_limb(QG, attribute_class = 'viterbi_class', number_leaves_limb = 1, new_apex_class = 4)

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb')

display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_classlqg2", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_seg.txt")

def weight_with_semantic(QG=QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_linear=0, class_apex=4,
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
    list_limb = [x for x, y in QG.nodes(data=True) if y[class_attribute] == class_limb]
    for leaf in list_limb:
        path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"], weight=weight)
        for n in path:
            if QG.nodes[n][class_attribute] == class_linear:
                QG.nodes[n][class_attribute] = new_class_petiol


weight_with_semantic(QG=QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_linear=0, class_apex=4,
                         weight_limb_limb=10000, weight_apex_apex=10000, weight_apex_mainstem=10000,
                         weight_limb_apex=10000, weight_limb_mainstem=10000, weight_linear_mainstem=0, weight_linear_linear=0,
                         weight_linear_limb=0, weight_linear_apex=0)

shortest_paths_from_limbs_det_petiol(QG, root_point_riemanian, class_attribute='viterbi_class', weight='weight_sem_paths', class_limb=1, class_linear=0, class_mainstem=3, new_class_petiol=5)

display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole')

def maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5, class_apex=4,
                         weight_petiol_petiol=0, weight_petiol_apex=10000):
    for e in QG.edges():
        c1 = QG.nodes[e[0]][class_attribute]
        c2 = QG.nodes[e[1]][class_attribute]
        l = [c1,c2]
        if set(l) == set([class_petiol, class_apex]):
            QG.edges[e]['weight_sem_paths']= weight_petiol_apex


def shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class', weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3, new_class_branch=6):
    list_apex = [x for x, y in QG.nodes(data=True) if y[class_attribute] == class_apex]
    for leaf in list_apex:
        path = nx.dijkstra_path(QG, leaf, G.nodes[root_point_riemanian]["quotient_graph_node"], weight=weight)
        print(path)
        for n in path:
            if QG.nodes[n][class_attribute] == class_branch:
                QG.nodes[n][class_attribute] = new_class_branch

maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5, class_apex=4,
                         weight_petiol_petiol=0, weight_petiol_apex=10000)

shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class', weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3, new_class_branch=6)

display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet_branches", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole_branches')

def merge_remaining_clusters(quotientgraph, remaining_clusters_class=0, class_attribute='viterbi_class'):
    list_clusters = [x for x, y in QG.nodes(data=True) if y[class_attribute] == remaining_clusters_class]

    G = quotientgraph.point_cloud_graph
    nodes_to_remove = []
    for u in list_clusters:
        adjacent_clusters = [n for n in quotientgraph[u]]
        max_number_of_connections = 0
        for i in range(len(adjacent_clusters)):
            if quotientgraph.nodes[adjacent_clusters[i]]['intra_class_node_number'] > max_number_of_nodes_in_adjacent_clusters:
                max_number_of_nodes_in_adjacent_clusters = quotientgraph.nodes[adjacent_clusters[i]]['intra_class_node_number']
                new_cluster = adjacent_clusters[i]
            # Opération de fusion du petit cluster avec son voisin le plus conséquent.
            # Mise à jour des attributs de la grande classe et suppression de la petite classe
            quotientgraph.nodes[new_cluster]['intra_class_node_number'] += quotientgraph.nodes[u][
                'intra_class_node_number']
            quotientgraph.nodes[new_cluster]['intra_class_edge_weight'] += (
                        quotientgraph.nodes[u]['intra_class_edge_weight']
                        + quotientgraph.edges[new_cluster, u][
                            'inter_class_edge_weight'])
            quotientgraph.nodes[new_cluster]['intra_class_edge_number'] += (
                        quotientgraph.nodes[u]['intra_class_edge_number']
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
                                       inter_class_edge_weight=quotientgraph.edges[u, adjacent_clusters[i]][
                                           'inter_class_edge_weight'],
                                       inter_class_edge_number=quotientgraph.edges[u, adjacent_clusters[i]][
                                           'inter_class_edge_number'])
            elif quotientgraph.has_edge(new_cluster, adjacent_clusters[i]) and new_cluster != adjacent_clusters[i]:
                quotientgraph.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                    quotientgraph.edges[u, adjacent_clusters[i]]['inter_class_edge_weight']
                quotientgraph.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                    quotientgraph.edges[u, adjacent_clusters[i]]['inter_class_edge_number']
        nodes_to_remove.append(u)

    quotientgraph.remove_nodes_from(nodes_to_remove)


#def create_tree_with_semantic_weights_and_shortest_paths()

#def detect_anomalies()

"""
# Plus courts chemins avec quotite feuille, remove les points par lesquels ça passe pas.
selected_edges = [(u,v) for u,v,e in QG.edges(data=True) if e['useful_path'] == 50]
QG.remove_edges_from(selected_edges)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="span-path", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="span-path_intra", data_on_nodes='intra_class_node_number', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="span-path_quotite", data_on_nodes='quotite_feuille_n', data=True, attributekmeans4clusters = False)

selected_nodes_isolated = [x for x in QG.nodes() if QG.degree(x) == 0]
for n in selected_nodes_isolated:
    QG.nodes[n]['viterbi_class'] = 30

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='isolated_nodes')

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm3')
"""