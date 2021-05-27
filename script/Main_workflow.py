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


from importlib import reload

pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/rawPoints_Arabido_0029.ply", format='ply')
r = 18
SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r)
G = PointCloudGraph(SimG)

def add_label_from_pcd_file(pointcloudgraph, file="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/peuplier_seg.txt"):
    label = np.loadtxt(file, delimiter=',')
    l = label[:, 3].astype('int32')
    pointcloudgraph.clustering_labels = l.reshape((len(l), 1))

def add_label_from_seperate_file(pointcloudgraph, file="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/rawLabels_Arabido_0029.txt"):
    label = np.loadtxt(file, delimiter=',')
    l = label.astype('int32')
    pointcloudgraph.clustering_labels = l.reshape((len(l), 1))
    j = 0
    for n in pointcloudgraph.nodes:
        pointcloudgraph.nodes[n]['clustering_labels'] = l[j]
        j += 1


add_label_from_seperate_file(G)

G.compute_graph_eigenvectors()
G.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
G.clustering_by_kmeans_in_four_clusters_using_gradient_norm(export_in_labeled_point_cloud=True)
#G.clustering_by_fiedler_and_optics(criteria=np.multiply(G.direction_gradient_on_fiedler_scaled, G.gradient_on_fiedler))

QG = QuotientGraph()
#QG.build_from_pointcloudgraph(G, G.kmeans_labels_gradient)
QG.build_from_pointcloudgraph(G, G.clustering_labels)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_attribute.txt")

def compute_quotientgraph_attribute_from_points(G, QG, attribute='clustering_labels'):
    for n in QG.nodes:
        moy = 0
        list_of_nodes = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        for e in list_of_nodes:
            moy += G.nodes[e][attribute]
        QG.nodes[n][attribute] = moy/len(list_of_nodes)


QG.compute_local_descriptors()
QG.compute_nodes_coordinates()
QG_t = nx.minimum_spanning_tree(QG, algorithm='kruskal', weight='inverse_inter_class_edge_weight')

display_and_export_quotient_graph_matplotlib(quotient_graph=QG_t, node_sizes=20, filename="quotient_graph_matplotlib", data_on_nodes='intra_class_node_number', data=True, attributekmeans4clusters = False)

draw_quotientgraph_cellcomplex(pcd, QG_t, G, color_attribute='quotient_graph_node', filename="graph_and_quotientgraph_3d.png")

"""
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

list_of_nodes_to_work = oversegment_part(quotientgraph=QG, list_quotient_node_to_work=[10], average_size_cluster=50)
QG.compute_direction_info(list_leaves=[])
opti_energy_dot_product(quotientgraph=QG, energy_to_stop=0.29, leaves_out=False, list_graph_node_to_work=list_of_nodes_to_work)
"""