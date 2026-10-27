#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single CLI for both the chenopodium and tomato segmentation pipelines.

Runs either pipeline on a 3D plant point cloud. Previously the two workflows
lived in two scripts (`Main_workflow.py` for chenopodium, `script_seg_tomato.py`
for tomato) with their accuracy knobs hard-coded. This module merges them behind
a ``--pipeline`` flag and exposes the knobs as options with unchanged defaults.
"""

import json
from pathlib import Path

import click
import networkx as nx
import numpy as np
import open3d as o3d

import spectral_clustering.similarity_graph as sgk
from spectral_clustering.Viterbi import viterbi_workflow
from spectral_clustering.display_and_export import display_and_export_quotient_graph_matplotlib
from spectral_clustering.display_and_export import export_fiedler_vector_on_pointcloud
from spectral_clustering.display_and_export import export_quotient_graph_attribute_on_point_cloud
from spectral_clustering.display_and_export import export_some_graph_attributes_on_point_cloud
from spectral_clustering.point_cloud_graph import PointCloudGraph
from spectral_clustering.quotient_graph import QuotientGraph
from spectral_clustering.quotientgraph_operations import calculate_leaf_quotients
from spectral_clustering.quotientgraph_operations import collect_quotient_graph_nodes_from_pointcloudpoints_majority
from spectral_clustering.quotientgraph_operations import merge_one_class_QG_nodes
from spectral_clustering.quotientgraph_operations import transfer_quotientgraph_infos_on_riemanian_graph
from spectral_clustering.quotientgraph_semantics import determination_main_stem_shortest_paths_improved
from spectral_clustering.quotientgraph_semantics import differenciate_apex_limb
from spectral_clustering.quotientgraph_semantics import maj_weight_semantics
from spectral_clustering.quotientgraph_semantics import merge_remaining_clusters
from spectral_clustering.quotientgraph_semantics import minimum_spanning_tree_quotientgraph_semantics
from spectral_clustering.quotientgraph_semantics import remove_edges_useful_paths
from spectral_clustering.quotientgraph_semantics import shortest_paths_from_apex_det_branch
from spectral_clustering.quotientgraph_semantics import shortest_paths_from_limbs_det_petiol
from spectral_clustering.quotientgraph_semantics import stem_detection_with_quotite_leaves
from spectral_clustering.quotientgraph_semantics import topology_control
from spectral_clustering.quotientgraph_semantics import weight_with_semantic
from spectral_clustering.split_and_merge import create_subgraphs_to_work
from spectral_clustering.split_and_merge import opti_energy_dot_product
from spectral_clustering.split_and_merge import resegment_nodes_with_elbow_method
from spectral_clustering.split_and_merge import select_all_quotientgraph_nodes_from_pointcloudgraph_cluster
from spectral_clustering.split_and_merge import select_minimum_centroid_class
from spectral_clustering.topological_energy import define_and_optimize_topological_energy


def _detect_lower_point(G):
    """Return the node index with the lowest Z coordinate (stem base)."""
    a = np.amin(G.nodes_coords[:, 2], axis=0)
    result = np.where(G.nodes_coords[:, 2] == a)
    return result[0][0]


def _load_point_cloud(pcd_path):
    """Read a PLY point cloud, returning (pcd, output_directory)."""
    pcd = o3d.io.read_point_cloud(Path(pcd_path), format='ply')
    return pcd, Path(pcd_path).parent


def _build_graph(pcd, radius):
    """Shared graph construction + spectral clustering of the gradient norm."""
    SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=radius)
    pcd_g = PointCloudGraph(SimG)
    pcd_g.pcd = pcdfinal
    return pcd_g


def pipeline_chenopodium(pcd_path, radius=18, root_point_riemanian=34586, n_clusters=4,
                         elbow_max=10, angle_merge=30, angle_stem=45, energy_iterations=10000,
                         formulae='improved', observations=None, initial_distribution=None,
                         transition_matrix=None, emission_params=None):
    """Semantic + instance segmentation for chenopodium (Viterbi HMM workflow).

    Mirrors the original ``Main_workflow.py`` body with the accuracy knobs
    exposed as parameters (defaults unchanged).
    """
    pcd, root_pcd_path = _load_point_cloud(pcd_path)

    pcd_g = _build_graph(pcd, radius)

    pcd_g.compute_graph_eigenvectors()
    export_fiedler_vector_on_pointcloud(pcd_g, filename=root_pcd_path / "pcd_vp2.txt")

    pcd_g.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
    clusters_centers = pcd_g.clustering_by_kmeans_using_gradient_norm(export_in_labeled_point_cloud=True,
                                                                      number_of_clusters=n_clusters)

    qg = QuotientGraph()
    qg.build_from_pointcloudgraph(pcd_g, pcd_g.kmeans_labels_gradient)
    export_some_graph_attributes_on_point_cloud(qg.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "pcd_attribute.txt")

    label_leaves = select_minimum_centroid_class(clusters_centers)
    list_leaves = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(pcd_g, qg, label_leaves)
    list_leaves_point = []
    for qnode in list_leaves:
        list_of_nodes_each = [x for x, y in pcd_g.nodes(data=True) if y['quotient_graph_node'] == qnode]
        list_leaves_point += list_of_nodes_each

    resegment_nodes_with_elbow_method(qg, QG_nodes_to_rework=list_leaves, number_of_cluster_tested=elbow_max,
                                      attribute='norm_gradient', number_attribute=1, standardization=False)
    qg.rebuild(qg.point_cloud_graph)
    export_some_graph_attributes_on_point_cloud(qg.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "pcd_attribute_after_resegment_leaves_norm.txt")

    define_and_optimize_topological_energy(quotient_graph=qg, point_cloud_graph=pcd_g, exports=True,
                                           formulae=formulae, number_of_iteration=energy_iterations,
                                           choice_of_node_to_change='max_energy')
    qg.rebuild(qg.point_cloud_graph)
    list_leaves = collect_quotient_graph_nodes_from_pointcloudpoints_majority(quotient_graph=qg,
                                                                              list_of_points=list_leaves_point)

    list_of_linear = [n for n in qg.nodes if n not in list_leaves]
    resegment_nodes_with_elbow_method(qg, QG_nodes_to_rework=list_of_linear, number_of_cluster_tested=elbow_max,
                                      attribute='direction_gradient', number_attribute=3, standardization=False)
    qg.rebuild(qg.point_cloud_graph)
    export_some_graph_attributes_on_point_cloud(qg.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "reseg_linear.txt")

    define_and_optimize_topological_energy(quotient_graph=qg, point_cloud_graph=pcd_g, exports=True,
                                           formulae=formulae, number_of_iteration=5000,
                                           choice_of_node_to_change='max_energy')
    qg.rebuild(qg.point_cloud_graph)
    list_leaves = collect_quotient_graph_nodes_from_pointcloudpoints_majority(quotient_graph=qg,
                                                                              list_of_points=list_leaves_point)
    qg.compute_direction_info(list_leaves=list_leaves)
    opti_energy_dot_product(quotientgraph=qg, subgraph_riemannian=pcd_g, angle_to_stop=angle_merge, export_iter=True,
                            list_leaves=list_leaves)
    qg.compute_nodes_coordinates()
    qg.compute_local_descriptors()

    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="planarity", data_on_nodes='planarity',
                                                 data=True, attributekmeans4clusters=False, directory=root_pcd_path)
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="linearity", data_on_nodes='linearity',
                                                 data=True, attributekmeans4clusters=False, directory=root_pcd_path)
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="scattering", data_on_nodes='scattering',
                                                 data=True, attributekmeans4clusters=False, directory=root_pcd_path)
    export_quotient_graph_attribute_on_point_cloud(qg, 'planarity2')
    export_quotient_graph_attribute_on_point_cloud(qg, 'linearity')
    export_quotient_graph_attribute_on_point_cloud(qg, 'scattering')

    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="quotient_graph_matplotlib_final",
                                                 data_on_nodes='intra_class_node_number', data=False,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    QG_t2 = minimum_spanning_tree_quotientgraph_semantics(qg)
    display_and_export_quotient_graph_matplotlib(qg=QG_t2, node_sizes=20, name="quotient_graph_matplotlib",
                                                 data_on_nodes='quotient_graph_node', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    rt = pcd_g.nodes[root_point_riemanian]['quotient_graph_node']
    viterbi_workflow(minimum_spanning_tree=QG_t2, quotient_graph=qg, root=rt,
                     observation_list_import=observations, initial_distribution=initial_distribution,
                     transition_matrix=transition_matrix, parameters_emission=emission_params)
    export_quotient_graph_attribute_on_point_cloud(qg, attribute='viterbi_class', name='first_viterbi')

    list_of_leaves = [x for x, y in qg.nodes(data=True) if y['viterbi_class'] == 1]
    list_of_linear = [x for x, y in qg.nodes(data=True) if y['viterbi_class'] == 0]
    transfer_quotientgraph_infos_on_riemanian_graph(QG=qg, info='viterbi_class')

    list_of_stem = determination_main_stem_shortest_paths_improved(QG=qg, ptsource=root_point_riemanian,
                                                                   list_of_linear_QG_nodes=list_of_linear,
                                                                   angle_to_stop=angle_stem)
    export_quotient_graph_attribute_on_point_cloud(qg, attribute='viterbi_class', name='stem_detect')
    export_quotient_graph_attribute_on_point_cloud(qg, attribute="dir_gradient_stdv")
    export_quotient_graph_attribute_on_point_cloud(qg, attribute="dir_gradient_angle_mean")

    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="quotient_graph_matplotlib_viterbi_stem",
                                                 data_on_nodes='viterbi_class', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="quotient_graph_matplotlib_nodesnumber",
                                                 data_on_nodes='quotient_graph_node_mean', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    QG_merge = merge_one_class_QG_nodes(qg, attribute='viterbi_class', viterbiclass=[1, 3])
    export_some_graph_attributes_on_point_cloud(pcd_g=QG_merge.point_cloud_graph,
                                                graph_attribute='quotient_graph_node',
                                                filename='Merge_leaves_after_viterbi.txt')
    export_quotient_graph_attribute_on_point_cloud(qg, attribute='viterbi_class')

    display_and_export_quotient_graph_matplotlib(qg=QG_merge, node_sizes=20, name="quotient_graph_matplotlib_viterbi_stemmerge",
                                                 data_on_nodes='viterbi_class', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)
    display_and_export_quotient_graph_matplotlib(qg=QG_merge, node_sizes=20, name="quotient_graph_matplotlib_nodesnumbermerge",
                                                 data_on_nodes='quotient_graph_node_mean', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    return QG_merge


def pipeline_tomato(pcd_path, radius=18, root_point_riemanian=None, n_clusters=4, elbow_max=20,
                    angle_merge=30, formulae='improved'):
    """Semantic + instance segmentation for tomato (geometric semantic workflow).

    Mirrors the original ``script_seg_tomato.py`` body with the accuracy knobs
    exposed as parameters (defaults unchanged). The root point is auto-detected
    as the lowest-Z node unless an explicit index is given.
    """
    pcd, root_pcd_path = _load_point_cloud(pcd_path)

    pcd_g = _build_graph(pcd, radius)

    if root_point_riemanian is None:
        root_point_riemanian = _detect_lower_point(pcd_g)

    pcd_g.compute_graph_eigenvectors()
    pcd_g.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
    clusters_centers = pcd_g.clustering_by_kmeans_using_gradient_norm(export_in_labeled_point_cloud=True,
                                                                      number_of_clusters=n_clusters)
    QG = QuotientGraph()
    QG.build_from_pointcloudgraph(pcd_g, pcd_g.kmeans_labels_gradient)
    export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "pcd_attribute.txt")

    label_leaves = select_minimum_centroid_class(clusters_centers)
    list_leaves1 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(pcd_g, QG, label_leaves)

    for n in QG.nodes:
        QG.nodes[n]["viterbi_class"] = np.nan
    list_of_linear = []
    for n in QG.nodes:
        if n in list_leaves1:
            QG.nodes[n]["viterbi_class"] = 1
        else:
            list_of_linear.append(n)
            QG.nodes[n]["viterbi_class"] = 0
    if pcd_g.nodes[root_point_riemanian]["quotient_graph_node"] in list_leaves1:
        list_leaves1.remove(pcd_g.nodes[root_point_riemanian]["quotient_graph_node"])
    list_of_linear.append(pcd_g.nodes[root_point_riemanian]["quotient_graph_node"])
    pcd_g.nodes[root_point_riemanian]["viterbi_class"] = 0

    export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm1')

    list_leaves_point = []
    for qnode in list_leaves1:
        list_of_nodes_each = [x for x, y in pcd_g.nodes(data=True) if y['quotient_graph_node'] == qnode]
        list_leaves_point += list_of_nodes_each

    list_of_linear = [n for n in QG.nodes if n not in list_leaves1]
    resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=list_of_linear, number_of_cluster_tested=elbow_max,
                                      attribute='direction_gradient', number_attribute=3, standardization=False)
    QG.rebuild(QG.point_cloud_graph)
    export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "reseg_linear.txt")

    list_leaves2 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(pcd_g, QG, label_leaves)
    list_of_linear = []
    for n in QG.nodes:
        QG.nodes[n]["viterbi_class"] = np.nan
        if n in list_leaves2:
            QG.nodes[n]["viterbi_class"] = 1
        else:
            list_of_linear.append(n)
            QG.nodes[n]["viterbi_class"] = 0
    if pcd_g.nodes[root_point_riemanian]["quotient_graph_node"] in list_leaves2:
        list_leaves2.remove(pcd_g.nodes[root_point_riemanian]["quotient_graph_node"])
    list_of_linear.append(pcd_g.nodes[root_point_riemanian]["quotient_graph_node"])
    pcd_g.nodes[root_point_riemanian]["viterbi_class"] = 0
    export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm2')

    QG.compute_direction_info(list_leaves=list_leaves2)
    opti_energy_dot_product(quotientgraph=QG, subgraph_riemannian=pcd_g, angle_to_stop=angle_merge, export_iter=False,
                            list_leaves=list_leaves2)
    QG.rebuild(QG.point_cloud_graph)
    pcd_g = QG.point_cloud_graph
    list_leaves3 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(pcd_g, QG, label_leaves)
    list_of_linear = []
    for n in QG.nodes:
        QG.nodes[n]["viterbi_class"] = np.nan
        if n in list_leaves3:
            QG.nodes[n]["viterbi_class"] = 1
        else:
            list_of_linear.append(n)
            QG.nodes[n]["viterbi_class"] = 0
    if pcd_g.nodes[root_point_riemanian]["quotient_graph_node"] in list_leaves3:
        list_leaves3.remove(pcd_g.nodes[root_point_riemanian]["quotient_graph_node"])
    list_of_linear.append(pcd_g.nodes[root_point_riemanian]["quotient_graph_node"])
    pcd_g.nodes[root_point_riemanian]["viterbi_class"] = 0
    export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_from_norm3')

    QG.ponderate_with_coordinates_of_points()
    list_leaves3 = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
    list_of_linear = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 0]

    differenciate_apex_limb(QG, attribute_class='viterbi_class', number_leaves_limb=1, new_apex_class=4)
    display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name="apex_classlqg2",
                                                 data_on_nodes='viterbi_class', data=True, attributekmeans4clusters=False)

    list_apex = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 4]
    stem_detection_with_quotite_leaves(QG, list_leaves3, list_apex, list_of_linear, root_point_riemanian, new_class_stem=3)

    QG = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[3])
    QG.point_cloud_graph = pcd_g

    list_leaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 1]
    list_notleaves = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] != 1]
    calculate_leaf_quotients(QG, list_leaves, list_notleaves, root_point_riemanian)
    for n in list_leaves:
        QG.nodes[n]["leaf_quotient_n"] = -1

    weight_with_semantic(QG=QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_linear=0,
                         class_apex=4, weight_limb_limb=10000, weight_apex_apex=10000, weight_apex_mainstem=10000,
                         weight_limb_apex=10000, weight_limb_mainstem=10000, weight_linear_mainstem=0,
                         weight_linear_linear=0, weight_linear_limb=0, weight_linear_apex=0)
    shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class',
                                        weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3,
                                        new_class_branch=6)
    maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5,
                         class_apex=4, class_branch=6, weight_petiol_petiol=0, weight_petiol_apex=10000,
                         weight_branch_limb=10000)
    shortest_paths_from_limbs_det_petiol(QG, root_point_riemanian, class_attribute='viterbi_class',
                                         weight='weight_sem_paths', class_limb=1, class_linear=0, class_mainstem=3,
                                         new_class_petiol=5)
    QG = merge_remaining_clusters(quotientgraph=QG, remaining_clusters_class=0, class_attribute='viterbi_class')
    remove_edges_useful_paths(QG)

    display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name="semantic_1",
                                                 data_on_nodes='viterbi_class', data=True, attributekmeans4clusters=False)
    export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_1')
    export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "instance_1.txt")

    topology_control(QG, attribute_class_control='viterbi_class', class_stem=3, class_petiols=5, class_limb=1,
                     class_apex=4, error_norm=10, error_dir=11)

    transfer_quotientgraph_infos_on_riemanian_graph(QG, info='viterbi_class')

    def treat_topology_error2(QG, attribute_class_control='viterbi_class', number_attribute=1, class_limb=1,
                              class_linear=0, error=10, way_to_treat='norm_gradient', number_of_cluster_tested=20):
        list_nodes_QG_to_work = [x for x, y in QG.nodes(data=True) if y[attribute_class_control] == error]
        G = QG.point_cloud_graph
        for node_QG in list_nodes_QG_to_work:
            SG_node = create_subgraphs_to_work(QG, list_quotient_node_to_work=[node_QG])
            clusters_centers = resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=[node_QG],
                                                                 number_of_cluster_tested=number_of_cluster_tested,
                                                                 attribute=way_to_treat, number_attribute=number_attribute,
                                                                 standardization=False, numer=1)
            if way_to_treat == 'norm_gradient':
                label_leaves = select_minimum_centroid_class(clusters_centers)
                list_true_labels = list(set(nx.get_node_attributes(SG_node, 'quotient_graph_node').values()))
                list_true_labels.sort()
                points_leaves = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == list_true_labels[label_leaves]]
                for n in SG_node:
                    if n in points_leaves:
                        G.nodes[n][attribute_class_control] = class_limb
                    else:
                        G.nodes[n][attribute_class_control] = class_linear
            if way_to_treat == 'direction_gradient':
                for n in SG_node:
                    G.nodes[n][attribute_class_control] = class_linear

    treat_topology_error2(QG=QG, attribute_class_control='viterbi_class', number_attribute=1, class_limb=1,
                          class_linear=0, error=10, way_to_treat='norm_gradient', number_of_cluster_tested=elbow_max)
    export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "pcd_seg_norm.txt")
    treat_topology_error2(QG=QG, attribute_class_control='viterbi_class', number_attribute=3, class_limb=1,
                          class_linear=0, error=11, way_to_treat='direction_gradient', number_of_cluster_tested=elbow_max)

    QG.rebuild(pcd_g)
    label_leaves = 1.0
    list_leaves3 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(pcd_g, QG, label_leaves, attribute='viterbi_class')
    list_apex3 = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(pcd_g, QG, 4.0, attribute='viterbi_class')
    for n in QG.nodes:
        if n in list_leaves3:
            QG.nodes[n]["viterbi_class"] = 1
        elif n in list_apex3:
            QG.nodes[n]["viterbi_class"] = 1
        else:
            list_of_linear.append(n)
            QG.nodes[n]["viterbi_class"] = 0

    QG = _obtain_tree_with_botanical(QG, root_point_riemanian, list_leaves3)

    display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name="semantic_final",
                                                 data_on_nodes='viterbi_class', data=True, attributekmeans4clusters=False)
    export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class', name='semantic_final')
    export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "instance_final.txt")

    return QG


def _obtain_tree_with_botanical(QG, root_point_riemanian, list_leaves3):
    G = QG.point_cloud_graph
    differenciate_apex_limb(QG, attribute_class='viterbi_class', number_leaves_limb=1, new_apex_class=4)
    list_apex = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 4]
    list_of_linear = [x for x, y in QG.nodes(data=True) if y['viterbi_class'] == 0]
    stem_detection_with_quotite_leaves(QG, list_leaves3, list_apex, list_of_linear, root_point_riemanian, new_class_stem=3)
    QG = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[3])
    QG.point_cloud_graph = G
    weight_with_semantic(QG=QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_linear=0,
                         class_apex=4, weight_limb_limb=10000, weight_apex_apex=10000, weight_apex_mainstem=10000,
                         weight_limb_apex=10000, weight_limb_mainstem=10000, weight_linear_mainstem=0,
                         weight_linear_linear=0, weight_linear_limb=0, weight_linear_apex=0)
    shortest_paths_from_apex_det_branch(QG, root_point_riemanian, class_attribute='viterbi_class',
                                        weight='weight_sem_paths', class_apex=4, class_branch=0, class_mainstem=3,
                                        new_class_branch=6)
    maj_weight_semantics(QG, class_attribute='viterbi_class', class_limb=1, class_mainstem=3, class_petiol=5,
                         class_apex=4, class_branch=6, weight_petiol_petiol=0, weight_petiol_apex=10000,
                         weight_branch_limb=10000)
    shortest_paths_from_limbs_det_petiol(QG, root_point_riemanian, class_attribute='viterbi_class',
                                         weight='weight_sem_paths', class_limb=1, class_linear=0, class_mainstem=3,
                                         new_class_petiol=5)
    QG = merge_remaining_clusters(quotientgraph=QG, remaining_clusters_class=0, class_attribute='viterbi_class')
    remove_edges_useful_paths(QG)
    return QG


@click.command(help="Perform semantic and instance segmentation of a 3D plant point cloud.")
@click.argument("pcd-path", type=click.Path(exists=True))
@click.option("--pipeline", type=click.Choice(["chenopodium", "tomato"]), default="chenopodium",
              help="Segmentation workflow to run (default: chenopodium).")
@click.option("-r", "--radius", type=int, default=18,
              help="Nearest-neighbor count for Riemannian graph creation.")
@click.option("-p", "--root-point", "root_point", type=int, default=None,
              help="Root point index in the point cloud (base of stem). "
                   "Chenopodium default 34586; tomato auto-detects the lowest-Z node.")
@click.option("--root-auto", is_flag=True, default=False,
              help="Auto-detect the root as the lowest-Z node (tomato behaviour).")
@click.option("--n-clusters", type=int, default=4,
              help="Number of clusters for the initial gradient-norm k-means.")
@click.option("--elbow-max", type=int, default=10,
              help="Max clusters tested by the elbow method during re-segmentation "
                   "(chenopodium 10, tomato 20).")
@click.option("--angle-merge", type=float, default=30,
              help="Angle threshold (deg) for merging clusters in energy dot-product optimisation.")
@click.option("--angle-stem", type=float, default=45,
              help="Angle threshold (deg) for stem continuation (chenopodium).")
@click.option("--energy-iterations", type=int, default=10000,
              help="Iterations for the first topological-energy optimisation (chenopodium).")
@click.option("--formulae", type=click.Choice(["improved", "classical"]), default="improved",
              help="Topological-energy formula (chenopodium).")
@click.option("--observations", type=str, default='["planarity2", "linearity"]',
              help="JSON list of Viterbi observation attributes (chenopodium).")
@click.option("--initial-distribution", type=str, default="[1, 0]",
              help="JSON list, Viterbi initial distribution (chenopodium).")
@click.option("--transition-matrix", type=str, default="[[0.2, 0.8], [0, 1]]",
              help="JSON matrix, Viterbi transition matrix (chenopodium).")
@click.option("--emission-params", type=str,
              default="[[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]]",
              help="JSON list, Viterbi emission parameters (chenopodium).")
def main(pcd_path, pipeline, radius, root_point, root_auto, n_clusters, elbow_max, angle_merge,
         angle_stem, energy_iterations, formulae, observations, initial_distribution, transition_matrix,
         emission_params):
    """Perform semantic and instance segmentation of a 3D plant point cloud.

    Selects the workflow with --pipeline:

    - chenopodium (default): graph-based semantic + instance segmentation of
      Chenopodium album plants. Root defaults to index 34586 unless --root-auto
      or -p/--root-point is given. See docs/usage.md for the Viterbi knobs.
    - tomato: semantic + instance segmentation of tomato plants. The root
      (stem base) is auto-detected as the lowest-Z node unless -p/--root-point
      is given.

    Outputs (labelled point clouds, graph attributes, figures) are written to
    the directory containing the input cloud.
    """
    if pipeline == "tomato":
        pipeline_tomato(pcd_path=pcd_path, radius=radius, root_point_riemanian=root_point,
                        n_clusters=n_clusters, elbow_max=elbow_max, angle_merge=angle_merge)
    else:
        root_riemanian = root_point if root_point is not None else 34586
        if root_auto:
            SimG, pcdfinal = sgk.create_connected_riemannian_graph(
                point_cloud=o3d.io.read_point_cloud(Path(pcd_path), format='ply'),
                method='knn', nearest_neighbors=radius)
            root_riemanian = _detect_lower_point(PointCloudGraph(SimG))
        pipeline_chenopodium(pcd_path=pcd_path, radius=radius, root_point_riemanian=root_riemanian,
                             n_clusters=n_clusters, elbow_max=elbow_max, angle_merge=angle_merge,
                             angle_stem=angle_stem, energy_iterations=energy_iterations, formulae=formulae,
                             observations=json.loads(observations),
                             initial_distribution=json.loads(initial_distribution),
                             transition_matrix=json.loads(transition_matrix),
                             emission_params=json.loads(emission_params))


if __name__ == "__main__":
    main()
