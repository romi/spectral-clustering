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

from importlib import reload

begin = time.time()

pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenos/cheno_virtuel.ply", format='ply')
r = 18
SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r)
G = PointCloudGraph(SimG)
G.pcd = pcdfinal


G.compute_graph_eigenvectors()
"""
for i in range(50):
    sgk.export_eigenvectors_on_pointcloud(pcd = pcdfinal, keigenvec = G.keigenvec, k=i, filename='vecteurproprecol'+ str(i)+'.txt')
"""
G.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')

clusters_centers = G.clustering_by_kmeans_using_gradient_norm(export_in_labeled_point_cloud=True, number_of_clusters=4)


QG = QuotientGraph()
QG.build_from_pointcloudgraph(G, G.kmeans_labels_gradient)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_attribute.txt")
time1 = time.time()
# resegment leaves according to norm
label_leaves = select_minimum_centroid_class(clusters_centers)

list_leaves = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves)
resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = list_leaves, number_of_cluster_tested = 10, attribute='norm_gradient', number_attribute=1, standardization=False)
QG.rebuild(QG.point_cloud_graph)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_attribute_after_resegment_leaves_norm.txt")

#resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = list(QG.nodes), number_of_cluster_tested = 10, attribute='direction_gradient')
#QG.rebuild(QG.point_cloud_graph)
#export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
#                                            graph_attribute="quotient_graph_node",
#                                            filename="pcd_attribute_after_resegment_all_directions.txt")

time2 = time.time()
end = time2-time1
print(end)


#draw_quotientgraph_cellcomplex(pcd, QG_t, G, color_attribute='quotient_graph_node', filename="graph_and_quotientgraph_3d.png")


define_and_optimize_topological_energy(quotient_graph=QG,
                                       point_cloud_graph=G,
                                       exports=True,
                                       formulae='improved',
                                       number_of_iteration=10000,
                                       choice_of_node_to_change='max_energy')

QG.rebuild(QG.point_cloud_graph)
list_leaves = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves)
"""
segment_each_cluster_by_optics(quotientgraph=QG,list_leaves=list_leaves, leaves_out=True)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="optics_seg_directions.txt")
"""
list_of_linear = []
for n in QG.nodes:
    if n not in list_leaves:
        list_of_linear.append(n)
"""
sub_lin=create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=list_of_linear)
oversegment_part(quotientgraph=QG, subgraph_riemannian=sub_lin, average_size_cluster=20)
"""
resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = list_of_linear, number_of_cluster_tested = 10, attribute='direction_gradient', number_attribute=3, standardization=False)
QG.rebuild(QG.point_cloud_graph)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="reseg_linear.txt")

define_and_optimize_topological_energy(quotient_graph=QG,
                                       point_cloud_graph=G,
                                       exports=True,
                                       formulae='improved',
                                       number_of_iteration=5000,
                                       choice_of_node_to_change='max_energy')
QG.rebuild(QG.point_cloud_graph)

list_leaves = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves)
QG.compute_direction_info(list_leaves=list_leaves)
opti_energy_dot_product(quotientgraph=QG, subgraph_riemannian=G, angle_to_stop=30, export_iter=True, list_leaves=list_leaves)


QG.compute_nodes_coordinates()
QG.compute_local_descriptors()
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="planarity", data_on_nodes='planarity', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="linearity", data_on_nodes='linearity', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="scattering", data_on_nodes='scattering', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, 'planarity2')
export_quotient_graph_attribute_on_point_cloud(QG, 'linearity')
export_quotient_graph_attribute_on_point_cloud(QG, 'scattering')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_final", data_on_nodes='intra_class_node_number', data=False, attributekmeans4clusters = False)

QG_t2 = minimum_spanning_tree_quotientgraph_semantics(QG)
#QG_t2 = nx.minimum_spanning_tree(QG, algorithm='kruskal', weight='inverse_inter_class_edge_weight')

#display_and_export_quotient_graph_matplotlib(quotient_graph=QG_t2, node_sizes=20, filename="quotient_graph_matplotlib", data_on_nodes='quotient_graph_node', data=True, attributekmeans4clusters = False)

end = time.time()

timefinal = end-begin

compute_quotientgraph_mean_attribute_from_points(G, QG, attribute='norm_gradient')

list_attribute = []
for i in QG.nodes:
    list_attribute.append(QG.nodes[i]['intra_class_node_number'])
e = np.percentile(list_attribute, 50)
export_quotient_graph_attribute_on_point_cloud(QG, 'intra_class_node_number')


######################### VITERBI ################################################"
st_tree = read_pointcloudgraph_into_treex(pointcloudgraph=QG_t2)
# manual choice of root : QG node number of the lower stem cluster
rt = 8
t = build_spanning_tree(st_tree, rt, list_att=['planarity2', 'linearity', 'intra_class_node_number'])
create_observation_list(t, list_obs=['planarity2', 'linearity'], name='observations')

initial_distribution = [1, 0]
transition_matrix = [[0.2, 0.8], [0, 1]]
parameterstot = [[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]]
def gen_emission(k, parameters):  # Gaussian emission
    return random.gauss(parameters[k][0], parameters[k][1])
def pdf_emission_dim1(x, moy, sd):  # Gaussian emission
    return 1.0 / (sd * math.sqrt(2 * math.pi)) * math.exp(
        -1.0 / (2 * sd ** 2) * (x - moy) ** 2)
def pdf_emission_dimn(x, k, parameterstot):
    p = 1
    if len(parameterstot) == 1:
        p = pdf_emission_dim1(x, moy=parameterstot[0][k][0], sd=parameterstot[0][k][1])
        return p
    else:
        for i in range(len(parameterstot[0])):
            p *= pdf_emission_dim1(x[i], moy=parameterstot[k][i][0], sd=parameterstot[k][i][1])
        return p

viterbi(t, 'observations', initial_distribution, transition_matrix, pdf_emission_dimn, parameterstot)

add_viterbi_results_to_quotient_graph(QG_t2, t, list_semantics=['leaf', 'stem', 'NSP'])
add_viterbi_results_to_quotient_graph(QG, t, list_semantics=['leaf', 'stem', 'NSP'])
display_and_export_quotient_graph_matplotlib(quotient_graph=QG_t2, node_sizes=20, filename="quotient_graph_viterbi", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class')

#######################

# Fusion/divison des parties tiges lorsque c'est possible en excluant les feuilles
# selection des feuilles
list_of_leaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
list_of_linear = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 0]
def transfer_quotientgraph_infos_on_riemanian_graph(QG=QG,info='viterbi_class'):
    G = QG.point_cloud_graph
    for n in G.nodes:
        G.nodes[n][info] = QG.nodes[G.nodes[n]['quotient_graph_node']][info]
transfer_quotientgraph_infos_on_riemanian_graph(QG=QG,info='viterbi_class')
"""
resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = list_of_linear, number_of_cluster_tested = 10, attribute='direction_gradient', number_attribute=3, standardization=False)
QG.rebuild(QG.point_cloud_graph)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="reseg_tiges.txt")
QG.compute_direction_info(list_leaves=list_of_leaves)
opti_energy_dot_product(quotientgraph=QG,
                        subgraph_riemannian=G,
                        angle_to_stop=50,
                        export_iter=True,
                        list_leaves=list_of_leaves)
"""
# Détermination tige et attribution d'un numéro
def determination_main_stem(QG = QG, list_of_linear_QG_nodes=list_of_linear):
    sub_riemanian = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=list_of_linear_QG_nodes)
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

list_of_stem = determination_main_stem(QG=QG, list_of_linear_QG_nodes=list_of_linear)

#compute_quotientgraph_mean_attribute_from_points(G, QG, attribute='viterbi_class')

export_some_graph_attributes_on_point_cloud(pointcloudgraph=QG.point_cloud_graph,
                                            graph_attribute='viterbi_class',
                                            filename='add_stem_class.txt')
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
                                            filename="add_stem_class", data_on_nodes='viterbi_class',
                                            data=True, attributekmeans4clusters=False)

# Fusion des clusters




# Contrôle enchainement tige, pétiole, feuille