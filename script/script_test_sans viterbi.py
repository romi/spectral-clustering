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

from importlib import reload

begin = time.time()

pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenos/cheno_A_2021_04_19.ply", format='ply')
r = 18
SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r)
G = PointCloudGraph(SimG)
G.pcd = pcdfinal
# LOWER POINT, USER INPUT
root_point_riemanian = 33512
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

list_leaves = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves)
list_leaves_point = []
for qnode in list_leaves:
    list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
    list_leaves_point += list_of_nodes_each

list_of_linear = []
for n in QG.nodes:
    if n not in list_leaves:
        list_of_linear.append(n)

resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = list_of_linear, number_of_cluster_tested = 10, attribute='direction_gradient', number_attribute=3, standardization=False)
QG.rebuild(QG.point_cloud_graph)

export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="reseg_linear.txt")

QG.rebuild(QG.point_cloud_graph)

#delete_small_clusters(quotientgraph=QG, min_number_of_element_in_a_quotient_node=50)

QG.compute_direction_info(list_leaves=list_leaves)
opti_energy_dot_product(quotientgraph=QG, subgraph_riemannian=G, angle_to_stop=30, export_iter=True, list_leaves=list_leaves)


list_of_linear = []
for n in QG.nodes:
    QG.nodes[n]["viterbi_class"] = np.nan
for n in QG.nodes:
    if n in list_leaves:
        QG.nodes[n]["viterbi_class"] = 1
    if n not in list_leaves:
        list_of_linear.append(n)
        QG.nodes[n]["viterbi_class"] = 0

list_leaves.remove(G.nodes[root_point_riemanian]["quotient_graph_node"])
list_of_linear.append(G.nodes[root_point_riemanian]["quotient_graph_node"])
G.nodes[root_point_riemanian]["viterbi_class"] = 0
delete_small_clusters(quotientgraph=QG, min_number_of_element_in_a_quotient_node=50)

determination_main_stem_shortest_paths_improved(QG=QG, ptsource=root_point_riemanian, list_of_linear_QG_nodes=list_of_linear, angle_to_stop=45, minimumpoint=5)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='stem_detect')