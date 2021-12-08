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
root_point_riemanian = 34586
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
# resegment leaves according to norm
label_leaves = select_minimum_centroid_class(clusters_centers)

list_leaves = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(G, QG, label_leaves)
list_leaves_point = []
for qnode in list_leaves:
    list_of_nodes_each = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qnode]
    list_leaves_point += list_of_nodes_each

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
list_leaves = collect_quotient_graph_nodes_from_pointcloudpoints_majority(quotient_graph=QG, list_of_points=list_leaves_point)

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
#resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework = [G.nodes[root_point_riemanian]['quotient_graph_node']], number_of_cluster_tested = 5, attribute='direction_gradient', number_attribute=3, standardization=False)
#QG.rebuild(QG.point_cloud_graph)
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
#delete_small_clusters(quotientgraph=QG, min_number_of_element_in_a_quotient_node=50)
list_leaves = collect_quotient_graph_nodes_from_pointcloudpoints_majority(quotient_graph=QG, list_of_points=list_leaves_point)
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

display_and_export_quotient_graph_matplotlib(quotient_graph=QG_t2, node_sizes=20, filename="quotient_graph_matplotlib", data_on_nodes='quotient_graph_node', data=True, attributekmeans4clusters = False)

end = time.time()

timefinal = end-begin
rt = G.nodes[root_point_riemanian]['quotient_graph_node']

######################### VITERBI ################################################
viterbi_workflow(minimum_spanning_tree=QG_t2,
                 quotient_graph=QG,
                 root=rt,
                 observation_list_import=['planarity2', 'linearity'],
                 initial_distribution=[1, 0],
                 transition_matrix=[[0.2, 0.8], [0, 1]],
                 parameters_emission=[[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]])
#######################
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='first_viterbi')

# Fusion/divison des parties tiges lorsque c'est possible en excluant les feuilles
# selection des feuilles
list_of_leaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
list_of_linear = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 0]
def transfer_quotientgraph_infos_on_riemanian_graph(QG=QG,info='viterbi_class'):
    G = QG.point_cloud_graph
    for n in G.nodes:
        G.nodes[n][info] = QG.nodes[G.nodes[n]['quotient_graph_node']][info]

transfer_quotientgraph_infos_on_riemanian_graph(QG=QG,info='viterbi_class')

# Détermination tige et attribution d'un numéro
def determination_main_stem_shortest_paths(QG = QG, list_of_linear_QG_nodes=list_of_linear):
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

def determination_main_stem_shortest_paths_improved(QG = QG, ptsource=0, list_of_linear_QG_nodes=list_of_linear, angle_to_stop=45, minimumpoint=5):
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
          QG.nodes[node]['viterbi_class'] = -1

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
        QG.nodes[n]['viterbi_class'] = 3

    transfer_quotientgraph_infos_on_riemanian_graph(QG=QG, info='viterbi_class')

    return final_list_stem_clean

def determination_main_stem(QG=QG, list_of_linear_QG_nodes=list_of_linear, stemroot = rt, list_leaves=list_of_leaves, angle_to_stop=45, new_viterbi_class_number=3):
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



list_of_stem = determination_main_stem_shortest_paths_improved(QG=QG, ptsource=root_point_riemanian, list_of_linear_QG_nodes=list_of_linear, angle_to_stop=45)
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='stem_detect')

export_quotient_graph_attribute_on_point_cloud(QG, attribute="dir_gradient_stdv")
export_quotient_graph_attribute_on_point_cloud(QG, attribute="dir_gradient_angle_mean")

display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_viterbi_stem", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20, filename="quotient_graph_matplotlib_nodesnumber", data_on_nodes='quotient_graph_node_mean', data=True, attributekmeans4clusters = False)


### Merge only the leaves
QG_merge = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass = [1, 3])
export_some_graph_attributes_on_point_cloud(pointcloudgraph=QG_merge.point_cloud_graph,
                                                    graph_attribute='quotient_graph_node',
                                                    filename='Merge_leaves_after_viterbi.txt')


export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG_merge, node_sizes=20, filename="quotient_graph_matplotlib_viterbi_stemmerge", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)
display_and_export_quotient_graph_matplotlib(quotient_graph=QG_merge, node_sizes=20, filename="quotient_graph_matplotlib_nodesnumbermerge", data_on_nodes='quotient_graph_node_mean', data=True, attributekmeans4clusters = False)



####
"""
from treex import *
from treex.simulation import *
from treex.utils.save import *
QG_t2 = minimum_spanning_tree_quotientgraph_semantics(QG_merge)
st_tree = read_pointcloudgraph_into_treex(pointcloudgraph=QG_t2)
rt = QG_merge.point_cloud_graph.nodes[root_point_riemanian]['quotient_graph_node']
t = build_spanning_tree(st_tree, rt, list_att=['planarity2'])
subtree = t.subtrees_sorted_by('height')

for h in sorted(subtree.keys()):
    for elt in subtree[h]:
        if h == 0:
            elt.add_attribute_to_id("seg", 0)
        elif h == 1:
            #ce sont des feuilles
            if len(elt.my_children)>1:
                elt.add_attribute_to_id("seg", 0)
            else:
                elt.add_attribute_to_id("seg", 1)
        else:
            # ce sont des tiges à "h-1"
            i = max([child.get_attribute("seg") for child in elt.my_children])
            elt.add_attribute_to_id("seg", i+1)

dict = t.dict_of_ids()
for n in t.list_of_ids():
    qg_node = dict[n]['attributes']['nx_label']
    QG_merge.nodes[qg_node]['seg'] = dict[n]['attributes']['seg']

export_quotient_graph_attribute_on_point_cloud(QG_merge, attribute='seg')



determination_main_stem(QG=QG, list_of_linear_QG_nodes=list_of_linear, stemroot = rt, list_leaves=list_of_leaves, angle_to_stop=45)


export_some_graph_attributes_on_point_cloud(pointcloudgraph=QG.point_cloud_graph,
                                            graph_attribute='viterbi_class',
                                            filename='add_stem_class.txt')
export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='afteraddstem')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG, node_sizes=20,
                                            filename="add_stem_class", data_on_nodes='viterbi_class',
                                            data=True, attributekmeans4clusters=False)

# Fusion des clusters
QG_merge = deepcopy(QG)
QG_merge = merge_similar_class_QG_nodes(QG_merge, attribute='viterbi_class', export=True)
export_quotient_graph_attribute_on_point_cloud(QG_merge, attribute='viterbi_class', name='merge_class')

# Contrôle enchainement tige, pétiole, feuille

def topology_control(quotient_graph, viterbi_class_stem = 3, viterbi_class_petiols = 0 , viterbi_class_leaves = 1, viterbi_error_norm = 5, viterbi_error_dir=6):
    QG = quotient_graph
    # check leaves connected to stem
    stem = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == viterbi_class_stem]
    for n in QG[stem[0]]:
        if QG.nodes[n]['viterbi_class'] == viterbi_class_leaves:
            QG.nodes[n]['viterbi_class'] = viterbi_error_norm

    end_nodes = [x for x in QG.nodes() if QG.degree(x) == 1]
    for n in end_nodes:
        if QG.nodes[n]['viterbi_class'] != viterbi_class_leaves:
            QG.nodes[n]['viterbi_class'] = viterbi_error_norm

    petiols = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == viterbi_class_petiols]
    for n in petiols:
        if QG.degree(n) >= 3:
            QG.nodes[n]['viterbi_class'] = viterbi_error_dir

topology_control(quotient_graph=QG_merge)
export_quotient_graph_attribute_on_point_cloud(QG_merge, attribute='viterbi_class', name='QG_detection_erreurs_topologiques')
display_and_export_quotient_graph_matplotlib(quotient_graph=QG_merge, node_sizes=20,
                                            filename="detection_erreurs", data_on_nodes='viterbi_class',
                                            data=True, attributekmeans4clusters=False)

def treat_topology_error(quotient_graph, viterbi_error = 5, way_to_treat='direction'):
    QG = quotient_graph
    nodes_to_treat = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == viterbi_error]

    if way_to_treat == 'direction':
        resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=nodes_to_treat, number_of_cluster_tested=10,
                                          attribute='direction_gradient', number_attribute=3, standardization=False)
    elif way_to_treat == 'norm':
        resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=nodes_to_treat, number_of_cluster_tested=10,
                                          attribute='norm_gradient', number_attribute=1, standardization=False, numer = 1000)

treat_topology_error(quotient_graph=QG_merge, viterbi_error=5, way_to_treat='norm')
treat_topology_error(quotient_graph=QG_merge, viterbi_error=6, way_to_treat='dir')
QG_merge.rebuild(G=QG_merge.point_cloud_graph)
export_some_graph_attributes_on_point_cloud(pointcloudgraph=QG_merge.point_cloud_graph,
                                            graph_attribute='quotient_graph_node',
                                            filename='G_après_correction_erreurs_topo.txt')



delete_small_clusters(quotientgraph=QG_merge, min_number_of_element_in_a_quotient_node=50)
QG_merge.compute_local_descriptors()
QG_t2 = minimum_spanning_tree_quotientgraph_semantics(QG_merge)
rt = QG_merge.point_cloud_graph.nodes[root_point_riemanian]['quotient_graph_node']
######################### VITERBI ################################################"
viterbi_workflow(minimum_spanning_tree=QG_t2,
                 quotient_graph=QG_merge,
                 root=rt,
                 observation_list_import=['planarity2', 'linearity'],
                 initial_distribution=[1, 0, 0],
                 transition_matrix=[[0, 1, 0], [0, 0.2, 0.8], [0, 0, 1]],
                 parameters_emission=[[[0.4, 0.4], [0.8, 0.2]], [[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]])

#######################


export_quotient_graph_attribute_on_point_cloud(QG_merge, attribute='viterbi_class', name='QG_after_corr_topo')

topology_control(quotient_graph=QG_merge, viterbi_class_stem = 0, viterbi_class_petiols = 1 , viterbi_class_leaves = 2, viterbi_error_norm = 5, viterbi_error_dir=6)


"""