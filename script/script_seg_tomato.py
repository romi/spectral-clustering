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
import open3d as o3d

import spectral_clustering.point_cloud_graph as kpcg

import time
import copy
import os

from collections import Counter

from spectral_clustering.point_cloud_graph import *
from spectral_clustering.quotient_graph import *
from spectral_clustering.topological_energy import *
from spectral_clustering.segmentation_algorithms import *
from spectral_clustering.split_and_merge import *
from spectral_clustering.display_and_export import *
from spectral_clustering.quotientgraph_semantics import *
#from spectral_clustering.Viterbi import *
from spectral_clustering.dijkstra_segmentation import *
from spectral_clustering.quotientgraph_operations import *
from spectral_clustering.evaluation import *
import argparse

from importlib import reload

parser = argparse.ArgumentParser()
parser.add_argument("InputFileName")
args = parser.parse_args()
model_filename = args.InputFileName
begin = time.time()
#model_filename = '/Users/katiamirande/Documents/Données/Donnees_tests/Pheno4D/Tomato01/T01_0307_a.ply'
filenamemod = os.path.splitext(os.path.basename(model_filename))[0]
pcd = o3d.io.read_point_cloud(model_filename, format='ply')
r = 18
SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r)
G = PointCloudGraph(SimG)
G.pcd = pcdfinal
# LOWER POINT, USER INPUT
#root_point_riemanian = 7104
# LOWER POINT, AUTO in Z
def detect_lower_point(G):
    a = np.amin(G.nodes_coords[:,2], axis= 0)
    result = np.where(G.nodes_coords[:,2] == a)
    return result[0][0]
root_point_riemanian = detect_lower_point(G)
#print(root_point_riemanian)
# In cloudcompare : Edit > ScalarFields > Add point indexes on SF

G.compute_graph_eigenvectors()

#export_fiedler_vector_on_pointcloud(G, filename="pcd_vp2.txt")
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



resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = list_of_linear, number_of_cluster_tested = 20, attribute='direction_gradient', number_attribute=3, standardization=False)
QG.rebuild(QG.point_cloud_graph)

export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="reseg_linear.txt")
#delete_small_clusters(QG)
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
if G.nodes[root_point_riemanian]["quotient_graph_node"] in list_leaves2:
    list_leaves2.remove(G.nodes[root_point_riemanian]["quotient_graph_node"])
list_of_linear.append(G.nodes[root_point_riemanian]["quotient_graph_node"])
G.nodes[root_point_riemanian]["viterbi_class"] = 0

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm2')

QG.compute_direction_info(list_leaves=list_leaves2)
opti_energy_dot_product(quotientgraph=QG, subgraph_riemannian=G, angle_to_stop=30, export_iter=False, list_leaves=list_leaves2)
QG.rebuild(QG.point_cloud_graph)
G = QG.point_cloud_graph
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

########################## End GEOMETRY ONLY

# arbre couvrant sur QG.
# Calcul des centroîdes, stockage dans un attribut 'centroide_coordinates'
QG.ponderate_with_coordinates_of_points()

list_leaves3 = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
list_of_linear = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 0]


#G.find_local_extremum_of_Fiedler()
#export_some_graph_attributes_on_point_cloud(G,
#                                            graph_attribute="extrem_local_Fiedler",
#                                            filename="pcd_extrem_local.txt")
#QG.count_local_extremum_of_Fiedler()
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='number_of_local_Fiedler_extremum', name='number_extrema_QG_cluster')

differenciate_apex_limb(QG, attribute_class = 'viterbi_class', number_leaves_limb = 1, new_apex_class = 4)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb')
display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name="apex_classlqg2",
                                             data_on_nodes='viterbi_class', data=True, attributekmeans4clusters=False)
#export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
#                                            graph_attribute="quotient_graph_node",
#                                            filename="pcd_seg.txt")



#calcul_quotite_feuilles(QG, list_leaves3, list_of_linear, root_point_riemanian)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='leaf_quotient_n', name='quotite_feuille_sur_cluster')
#display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotite_feuilles_sur_noeuds", data_on_nodes='leaf_quotient_n', data=True, attributekmeans4clusters = False)
#
#Selection du plus long des plus courts chemins avec la pondération sur les edges en fonction de la quotité de feuilles
list_apex = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 4]
stem_detection_with_quotite_leaves(QG, list_leaves3, list_apex, list_of_linear, root_point_riemanian, new_class_stem=3)


QG = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[3])
QG.point_cloud_graph = G
#display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="QG_merge_tige", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)

list_leaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
list_notleaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] != 1]
calculate_leaf_quotients(QG, list_leaves, list_notleaves, root_point_riemanian)
for n in list_leaves:
    QG.nodes[n]["leaf_quotient_n"] = -1


weight_with_semantic(QG=QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_linear=0, class_apex=4,
                         weight_limb_limb=10000, weight_apex_apex=10000, weight_apex_mainstem=10000,
                         weight_limb_apex=10000, weight_limb_mainstem=10000, weight_linear_mainstem=0, weight_linear_linear=0,
                         weight_linear_limb=0, weight_linear_apex=0)

shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class', weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3, new_class_branch=6)

#display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet_branches", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole_branches')



maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5, class_apex=4, class_branch=6,
                         weight_petiol_petiol=0, weight_petiol_apex=10000, weight_branch_limb=10000)


shortest_paths_from_limbs_det_petiol(QG, root_point_riemanian, class_attribute='viterbi_class', weight='weight_sem_paths', class_limb=1, class_linear=0, class_mainstem=3, new_class_petiol=5)

#display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole')


QG = merge_remaining_clusters(quotientgraph=QG, remaining_clusters_class=0, class_attribute='viterbi_class')
#display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet_branches_delete0", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole_branches_delete0')

remove_edges_useful_paths(QG)
#display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="apex_class_pet_branches_remove", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_apex_limb_petiole_branches_remove')

display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name=filenamemod + "_semantic_1",
                                             data_on_nodes='viterbi_class', data=True, attributekmeans4clusters=False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name=filenamemod+'_semantic_1')
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename=filenamemod+"_instance_1.txt")

topology_control(QG, attribute_class_control ='viterbi_class',
                 class_stem=3, class_petiols=5, class_limb=1,
                 class_apex=4, error_norm=10, error_dir=11)

#display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="topo_control", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='topo_control')
#treat_topology_error(QG, attribute_class_control='viterbi_class', error = 10, way_to_treat='norm', number_of_cluster_tested=20)
#export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='topo_reseg')
#export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
#                                          graph_attribute="quotient_graph_node",
#                                            filename="pcd_seg.txt")



transfer_quotientgraph_infos_on_riemanian_graph(QG, info='viterbi_class')

def treat_topology_error2(QG, attribute_class_control='viterbi_class',number_attribute=1, class_limb = 1, class_linear = 0, error = 10, way_to_treat='norm_gradient', number_of_cluster_tested=20):
    list_nodes_QG_to_work = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == error]
    G = QG.point_cloud_graph
    for node_QG in list_nodes_QG_to_work:
        neighb = QG[node_QG]
        edges_neighb = QG.edges(node_QG)
        SG_node = create_subgraphs_to_work(QG, list_quotient_node_to_work=[node_QG])

        clusters_centers = resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=[node_QG], number_of_cluster_tested=number_of_cluster_tested,
                                          attribute=way_to_treat, number_attribute=number_attribute, standardization=False, numer=1)

        if way_to_treat == 'norm_gradient':
            print(clusters_centers)
            label_leaves = select_minimum_centroid_class(clusters_centers)
            print(label_leaves)
            list_true_labels = list(set(nx.get_node_attributes(SG_node, 'quotient_graph_node').values()))
            list_true_labels.sort()
            print(list_true_labels)
            points_leaves = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == list_true_labels[label_leaves]]
            print(points_leaves)
            for n in SG_node:
                if n in points_leaves:
                    G.nodes[n][attribute_class_control] = class_limb
                else:
                    G.nodes[n][attribute_class_control] = class_linear

        if way_to_treat == 'direction_gradient':
            for n in SG_node:
                G.nodes[n][attribute_class_control] = class_linear


        #sub_QG = QuotientGraph()
        #sub_QG.rebuild(SG_node)
        #sub-QG, rename the nodes to be sure they are not the same as QG.
        #maxi = max(QG.nodes)
        #H = nx.relabel_nodes(sub_QG, lambda x: x + maxi)
        # Link sub-QG to QG.
        # add neighbors to H ?
        # add adjacent edges ?

treat_topology_error2(QG=QG,attribute_class_control='viterbi_class',number_attribute=1, class_limb = 1, class_linear = 0, error = 10, way_to_treat='norm_gradient', number_of_cluster_tested=20)

export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                          graph_attribute="quotient_graph_node",
                                            filename="pcd_seg_norm.txt")
treat_topology_error2(QG=QG,attribute_class_control='viterbi_class', number_attribute=3, class_limb = 1, class_linear = 0, error = 11, way_to_treat='direction_gradient', number_of_cluster_tested=20)
#export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
#                                          graph_attribute="quotient_graph_node",
#                                            filename="pcd_seg_dir.txt")

#export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
#                                            graph_attribute="quotient_graph_node",
#                                            filename="pcd_reseg.txt")
#export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
#                                            graph_attribute="viterbi_class",
#                                            filename="pcd_reseg_semantic.txt")

QG.rebuild(G)
label_leaves = 1.0
list_leaves3 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves, attribute='viterbi_class')
list_apex3 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, 4.0, attribute='viterbi_class')
for n in QG.nodes:
    if n in list_leaves3:
        QG.nodes[n]["viterbi_class"] = 1
    elif n in list_apex3:
        QG.nodes[n]["viterbi_class"] = 1
    else:
        list_of_linear.append(n)
        QG.nodes[n]["viterbi_class"] = 0

def obtain_tree_with_botanical(QG):
    G = QG.point_cloud_graph
    differenciate_apex_limb(QG, attribute_class='viterbi_class', number_leaves_limb=1, new_apex_class=4)
    # Selection du plus long des plus courts chemins avec la pondération sur les edges en fonction de la quotité de feuilles
    list_apex = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 4]
    list_of_linear = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 0]
    stem_detection_with_quotite_leaves(QG, list_leaves3, list_apex, list_of_linear, root_point_riemanian, new_class_stem=3)
    QG = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[3])
    QG.point_cloud_graph = G
    weight_with_semantic(QG=QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_linear=0,
                         class_apex=4,
                         weight_limb_limb=10000, weight_apex_apex=10000, weight_apex_mainstem=10000,
                         weight_limb_apex=10000, weight_limb_mainstem=10000, weight_linear_mainstem=0,
                         weight_linear_linear=0,
                         weight_linear_limb=0, weight_linear_apex=0)
    shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class',
                                        weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3,
                                        new_class_branch=6)

    maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5, class_apex=4, class_branch=6,
                         weight_petiol_petiol=0, weight_petiol_apex=10000, weight_branch_limb=10000)

    shortest_paths_from_limbs_det_petiol(QG, root_point_riemanian, class_attribute='viterbi_class',
                                         weight='weight_sem_paths', class_limb=1, class_linear=0, class_mainstem=3,
                                         new_class_petiol=5)

    QG = merge_remaining_clusters(quotientgraph=QG, remaining_clusters_class=0, class_attribute='viterbi_class')

    remove_edges_useful_paths(QG)

    return QG

QG=obtain_tree_with_botanical(QG)

display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name=filenamemod + "_semantic_final",
                                             data_on_nodes='viterbi_class', data=True, attributekmeans4clusters=False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name=filenamemod+'_semantic_final')
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename=filenamemod+"_instance_final.txt")

time2 = time.time()

print(time2-time1)
#def create_tree_with_semantic_weights_and_shortest_paths()


#def detect_anomalies()

"""
# Plus courts chemins avec quotite feuille, remove les points par lesquels ça passe pas.
selected_edges = [(u,v) for u,v,e in QG.edges(data=True) if e['useful_path'] == 50]
QG.remove_edges_from(selected_edges)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="span-path", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="span-path_intra", data_on_nodes='intra_class_node_number', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="span-path_quotite", data_on_nodes='leaf_quotient_n', data=True, attributekmeans4clusters = False)

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
