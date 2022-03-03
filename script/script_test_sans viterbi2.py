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

pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenos/cheno_virtuel.ply", format='ply')
r = 18
SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r)
G = PointCloudGraph(SimG)
G.pcd = pcdfinal
# LOWER POINT, USER INPUT
root_point_riemanian = 7365
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
if G.nodes[root_point_riemanian]["quotient_graph_node"] in list_leaves1:
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
if G.nodes[root_point_riemanian]["quotient_graph_node"] in list_leaves2:
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



calcul_quotite_feuilles(QG, list_leaves3, list_of_linear, root_point_riemanian)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='quotite_feuille_n', name='quotite_feuille_sur_cluster')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotite_feuilles_sur_noeuds", data_on_nodes='quotite_feuille_n', data=True, attributekmeans4clusters = False)

#Selection du plus long des plus courts chemins avec la pondération sur les edges en fonction de la quotité de feuilles
list_apex = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 4]
stem_detection_with_quotite_leaves(QG, list_apex, list_of_linear, root_point_riemanian)

QG = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[3])
QG.point_cloud_graph = G
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="QG_merge_tige", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)

list_leaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
list_notleaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] != 1]
calcul_quotite_feuilles(QG, list_leaves, list_notleaves, root_point_riemanian)
for n in list_leaves:
    QG.nodes[n]["quotite_feuille_n"] = -1





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
    for e in QG.edges():
        QG.edges[e]['useful_path_shortest'] = False
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
        for i in range(len(path) - 1):
            if QG.has_edge(path[i], path[i + 1]):
                e = (path[i], path[i + 1])
            else:
                e = (path[i + 1], path[i])
            QG.edges[e]['useful_path_shortest'] = True

maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5, class_apex=4,
                         weight_petiol_petiol=0, weight_petiol_apex=10000)

shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class', weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3, new_class_branch=6)

display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet_branches", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole_branches')



def merge_remaining_clusters(quotientgraph, remaining_clusters_class=0, class_attribute='viterbi_class'):
    list_clusters = [x for x, y in quotientgraph.nodes(data=True) if y[class_attribute] == remaining_clusters_class]
    print(list_clusters)
    G = quotientgraph.point_cloud_graph

    for u in list_clusters:
        count = quotientgraph.compute_quotientgraph_metadata_on_a_node_interclass(u)
        max_class_o = max(count, key=count.get)
        max_class = max_class_o
        # deal with possibility of max_class_ another class to merge. In that case, second best linked is chosent
        # if only linked to another class to merge, merge with this one.
        while max_class in list_clusters and bool(count):
            count.pop(max_class)
            max_class = max(count, key=count.get)
        #if bool(count) is False:
        #    max_class = max_class_o

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


QG = merge_remaining_clusters(quotientgraph=QG, remaining_clusters_class=0, class_attribute='viterbi_class')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet_branches_delete0", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole_branches_delete0')


def remove_edges_useful_paths(QG):
    for e in QG.edges():
        if QG.edges[e]['useful_path_shortest'] is False:
            QG.remove_edge(*e)

remove_edges_useful_paths(QG)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet_branches_remove", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole_branches_remove')

topology_control(QG, attribute_class_control ='viterbi_class',
                 class_stem=3, class_petiols=5, class_limb=1,
                 class_apex=4, error_norm=10, error_dir=11)

display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="topo_control", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='topo_control')
treat_topology_error(QG, attribute_class_control='viterbi_class', error = 10, way_to_treat='norm', number_of_cluster_tested=20)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='topo_reseg')
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_seg.txt")

def treat_topology_error2(QG, attribute_class_control='viterbi_class', error = 10, way_to_treat='norm_gradient', number_of_cluster_tested=20):
    list_nodes_QG_to_work = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == error]

    for node_QG in list_nodes_QG_to_work:
        neighb = QG[node_QG]
        edges_neighb = QG.edges(node_QG)
        SG_node = create_subgraphs_to_work(QG, list_quotient_node_to_work=[node_QG])

        resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=[node_QG], number_of_cluster_tested=number_of_cluster_tested,
                                          attribute=way_to_treat, number_attribute=1, standardization=False, numer=1)

        sub_QG = QuotientGraph()
        sub_QG.rebuild(SG_node)
        #sub-QG, rename the nodes to be sure they are not the same as QG.
        maxi = max(QG.nodes)
        H = nx.relabel_nodes(sub_QG, lambda x: x + maxi)
        # Link sub-QG to QG.
        # add neighbors to H ?
        # add adjacent edges ?




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


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
i = 0
number_attribute = 1
attribute ='norm_gradient'
sub = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=[i])
SG = sub
list_nodes = list(SG.nodes)
Xcoord = np.zeros((len(SG), 3))
for u in range(len(SG)):
    Xcoord[u] = SG.nodes[list(SG.nodes)[u]]['pos']
Xnorm = np.zeros((len(SG), number_attribute))
for u in range(len(SG)):
    Xnorm[u] = SG.nodes[list(SG.nodes)[u]][attribute]
sc = StandardScaler()
Xnorm = sc.fit_transform(Xnorm)
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 20), metric='distortion')
visualizer.fit(Xnorm)  # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
k_opt = visualizer.elbow_value_
if k_opt > 1:
    clustering = skc.KMeans(n_clusters=k_opt, init='k-means++', n_init=20, max_iter=300, tol=0.0001).fit(
        Xnorm)
    new_labels = np.zeros((len(SG.nodes), 4))
    new_labels[:, 0:3] = Xcoord
    for pt in range(len(list_nodes)):
        new_labels[pt, 3] = clustering.labels_[pt]
    np.savetxt('pcd_new_labels_' + str(i) + '.txt', new_labels, delimiter=",")
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_seg.txt")


"""