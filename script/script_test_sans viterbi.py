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
"""
for i in range(50):
    sgk.export_eigenvectors_on_pointcloud(pcd = pcdfinal, keigenvec = G.keigenvec, k=i, filename='vecteurproprecol'+ str(i)+'.txt')
"""
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

#oversegment_part_return_list(QG, list_quotient_node_to_work=list_of_linear, average_size_cluster=100)
#export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
#                                            graph_attribute="quotient_graph_node",
#                                            filename="reseg_test_kmeans.txt")




resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = list_of_linear, number_of_cluster_tested = 10, attribute='direction_gradient', number_attribute=3, standardization=False)
QG.rebuild(QG.point_cloud_graph)

export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="reseg_linear.txt")
#delete_small_clusters(quotientgraph=QG, min_number_of_element_in_a_quotient_node=50)
#QG.rebuild(QG.point_cloud_graph)

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
QG.compute_nodes_coordinates()
# Calcul des poids
for e in QG.edges:
    n1 = e[0]
    n2 = e[1]
    c1 = QG.nodes[n1]['centroide_coordinates']
    c2 = QG.nodes[n2]['centroide_coordinates']
    dist = np.linalg.norm(c1-c2)
    QG.edges[e]['distance_centroides'] = dist
#arbre couvrant dur distances les plus courtes.
tree_centroide = nx.minimum_spanning_tree(QG, algorithm='kruskal', weight='centroide_coordinates')
display_and_export_quotient_graph_matplotlib(quotient_graph=tree_centroide, node_sizes=20, filename="span-tree_centroides", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)

delete_small_clusters(QG)
list_leaves3 = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
list_of_linear = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 0]

def calcul_quotite_feuilles(QG, list_leaves, list_of_linear):
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

calcul_quotite_feuilles(QG, list_leaves3, list_of_linear)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='quotite_feuille_n', name='quotite_feuille_sur_cluster')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotite_feuilles_sur_noeuds", data_on_nodes='quotite_feuille_n', data=True, attributekmeans4clusters = False)

#Selection du plus long des plus courts chemins avec la pondération sur les edges en fonction de la quotité de feuilles
list_length = []
for leaf in list_leaves3:
    sub_qg = nx.subgraph(QG, list_of_linear + [leaf])
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
    for p in path:
        pathleafquantity.append(QG.nodes[p]['quotite_feuille_n'])
    lengthpath = sum(list(set(pathleafquantity)))
    list_length.append(lengthpath)

leafend = list_leaves3[list_length.index(max(list_length))]

list_stem = nx.dijkstra_path(QG, leafend, G.nodes[root_point_riemanian]["quotient_graph_node"], weight='distance_centroides')

for n in list_stem:
    if n not in list_leaves3:
        QG.nodes[n]["viterbi_class"] = 3

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_')

quotient_graph_compute_direction_mean(QG)
quotient_graph_compute_direction_standard_deviation(QG)

export_quotient_graph_attribute_on_point_cloud(QG, attribute='dir_gradient_angle_mean', name='dir_gradient_angle_mean')
export_quotient_graph_attribute_on_point_cloud(QG, attribute='dir_gradient_stdv', name='dir_gradient_stdv')




QG2 = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[3])
display_and_export_quotient_graph_matplotlib(quotient_graph=QG2, node_sizes=20, filename="QG_merge_tige", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
tree_centroide = nx.minimum_spanning_tree(QG2, algorithm='kruskal', weight='centroide_coordinates')
display_and_export_quotient_graph_matplotlib(quotient_graph=tree_centroide, node_sizes=20, filename="span-tree_centroides", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)



list_leaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]


list_notleaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] != 1]
calcul_quotite_feuilles(QG2, list_leaves, list_notleaves)
for n in list_leaves:
    QG2.nodes[n]["quotite_feuille_n"] = -1

selected_edges = [(u,v) for u,v,e in QG2.edges(data=True) if e['useful_path'] == 50]
QG2.remove_edges_from(selected_edges)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG2, node_sizes=20, filename="span-path", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG2, node_sizes=20, filename="span-path_intra", data_on_nodes='intra_class_node_number', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG2, node_sizes=20, filename="span-path_quotite", data_on_nodes='quotite_feuille_n', data=True, attributekmeans4clusters = False)

selected_nodes_isolated = [x for x in QG2.nodes() if QG2.degree(x) == 0]
for n in selected_nodes_isolated:
    QG2.nodes[n]['viterbi_class'] = 30

export_quotient_graph_attribute_on_point_cloud(QG2, attribute='viterbi_class', name='isolated_nodes')

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm3')


G.find_local_extremum_of_Fiedler()
export_some_graph_attributes_on_point_cloud(G,
                                            graph_attribute="extrem_local_Fiedler",
                                            filename="pcd_exttrem_local.txt")

QG.count_local_extremum_of_Fiedler()

export_quotient_graph_attribute_on_point_cloud(QG, attribute='number_of_local_Fiedler_extremum', name='number_extrema_QG_cluster')

def differenciate_apex_limb(QG):
    QG.count_local_extremum_of_Fiedler()
    list_limbs = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
    for l in list_limbs:
        if QG.nodes[l]['number_of_local_Fiedler_extremum'] > 1:
            QG.nodes[l]['viterbi_class'] = 4

differenciate_apex_limb(QG2)

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb')

display_and_export_quotient_graph_matplotlib(quotient_graph=QG2, node_sizes=20, filename="apex_classlqg2", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_seg.txt")

"""

determination_main_stem_shortest_paths_improved(QG=QG, ptsource=root_point_riemanian, list_of_linear_QG_nodes=list_of_linear, angle_to_stop=45, minimumpoint=3, classnumberstem=3, classnumberanomaly=2)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='stem_detect')

export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_attribute_afterstem.txt")


delete_small_clusters(quotientgraph=QG, min_number_of_element_in_a_quotient_node=30)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_attribute_afterstem_delete_small_clusters.txt")

display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_viterbi_stem", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
compute_quotientgraph_mean_attribute_from_points(G, QG, attribute='quotient_graph_node')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_nodesnumber", data_on_nodes='quotient_graph_node_mean', data=True, attributekmeans4clusters = False)

QG = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass = [3])
export_some_graph_attributes_on_point_cloud(pointcloudgraph=QG.point_cloud_graph,
                                                    graph_attribute='quotient_graph_node',
                                                    filename='Merge_stem.txt')



transfer_quotientgraph_infos_on_riemanian_graph(QG=QG,info='viterbi_class')
QG.rebuild(G)
compute_quotientgraph_mean_attribute_from_points(G, QG, attribute='quotient_graph_node')
compute_quotientgraph_mean_attribute_from_points(G, QG, attribute='viterbi_class')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_merge_stem_class", data_on_nodes='viterbi_class_mean', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_merge_stem_nodesnumber", data_on_nodes='quotient_graph_node_mean', data=True, attributekmeans4clusters = False)

list_of_leaves4 = []
list_of_leaves4 = [x for x, y in QG.nodes(data=True) if y['viterbi_class_mean'] == 1]
stem = [x for x, y in QG.nodes(data=True) if y['viterbi_class_mean'] == 3]
nodes_on_paths = []
for l in list_of_leaves4:
    path = nx.bidirectional_shortest_path(QG, l, stem[0])
    nodes_on_paths += path
list_node_needed = list(set(nodes_on_paths))

list_node_to_merge = []
for n in QG.nodes:
    if n not in list_node_needed:
        list_node_to_merge.append(n)

dictcolor = dict()
for key in list_node_needed:
    dictcolor[key]='b'
for key in list_node_to_merge:
    dictcolor[key]='r'

figure = plt.figure(0)
figure.clf()
colormap = ['red' if node in list_node_to_merge else 'green' for node in QG]
graph_layout = nx.kamada_kawai_layout(QG)
nx.drawing.nx_pylab.draw_networkx(QG,
                                  ax=figure.gca(),
                                  pos=graph_layout,
                                  with_labels=False,
                                  node_size=20,
                                  node_color=colormap)

figure.subplots_adjust(wspace=0, hspace=0)
figure.tight_layout()
figure.savefig('detect_anomalies')

"""
