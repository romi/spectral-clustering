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


from importlib import reload

begin = time.time()

pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/chenos/cheno_B_2021_04_19.ply", format='ply')
r = 18
SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r)
G = PointCloudGraph(SimG)
G.pcd = pcdfinal


G.compute_graph_eigenvectors()
G.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')

G.clustering_by_kmeans_using_gradient_norm(export_in_labeled_point_cloud=True, number_of_clusters=4)


QG = QuotientGraph()
QG.build_from_pointcloudgraph(G, G.kmeans_labels_gradient)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_attribute.txt")

#### test essai split branches et tiges
lqg = [x for x, y in QG.nodes(data=True) if y['kmeans_labels'] != 0]
sub = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=lqg)
oversegment_part(quotientgraph=QG, subgraph_riemannian=sub, average_size_cluster=150)



#draw_quotientgraph_cellcomplex(pcd, QG_t, G, color_attribute='quotient_graph_node', filename="graph_and_quotientgraph_3d.png")


define_and_optimize_topological_energy(quotient_graph=QG,
                                       point_cloud_graph=G,
                                       exports=True,
                                       formulae='improved',
                                       number_of_iteration=10000,
                                       choice_of_node_to_change='max_energy')

QG.rebuild(QG.point_cloud_graph)

segment_each_cluster_by_optics(quotientgraph=QG, leaves_out=True)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="optics_seg_directions.txt")

define_and_optimize_topological_energy(quotient_graph=QG,
                                       point_cloud_graph=G,
                                       exports=True,
                                       formulae='improved',
                                       number_of_iteration=1000,
                                       choice_of_node_to_change='max_energy')
QG.rebuild(QG.point_cloud_graph)


QG.compute_direction_info(list_leaves=[])
opti_energy_dot_product(quotientgraph=QG, subgraph_riemannian=G, angle_to_stop=30,export_iter=True)


QG.compute_nodes_coordinates()
QG.compute_local_descriptors()
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="planarity", data_on_nodes='planarity', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="linearity", data_on_nodes='linearity', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="scattering", data_on_nodes='scattering', data=True, attributekmeans4clusters = False)
export_quotient_graph_attribute_on_point_cloud(QG, 'planarity')
export_quotient_graph_attribute_on_point_cloud(QG, 'linearity')
export_quotient_graph_attribute_on_point_cloud(QG, 'scattering')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_final", data_on_nodes='intra_class_node_number', data=False, attributekmeans4clusters = False)

QG_t2 = minimum_spanning_tree_quotientgraph_semantics(QG)
#QG_t2 = nx.minimum_spanning_tree(QG, algorithm='kruskal', weight='inverse_inter_class_edge_weight')

#display_and_export_quotient_graph_matplotlib(quotient_graph=QG_t2, node_sizes=20, filename="quotient_graph_matplotlib", data_on_nodes='quotient_graph_node', data=True, attributekmeans4clusters = False)

end = time.time()

timefinal = end-begin

compute_quotientgraph_mean_attribute_from_points(G, QG, attribute='norm_gradient')
export_quotient_graph_attribute_on_point_cloud(QG, attribute='lambda2')