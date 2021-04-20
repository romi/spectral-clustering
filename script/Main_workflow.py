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

pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/Older_Cheno.ply", format='ply')
r = 18
G = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r)
G = PointCloudGraph(G)

G.compute_graph_eigenvectors()
G.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
G.clustering_by_kmeans_in_four_clusters_using_gradient_norm(export_in_labeled_point_cloud=True)
#G.clustering_by_fiedler_and_optics(criteria=np.multiply(G.direction_gradient_on_fiedler_scaled, G.gradient_on_fiedler))

QG = QuotientGraph()
QG.build_from_pointcloudgraph(G, G.kmeans_labels_gradient)
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="quotient_graph_node",
                                            filename="pcd_attribute.txt")


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
