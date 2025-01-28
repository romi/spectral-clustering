#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from pathlib import Path

import click
import open3d as o3d

import spectral_clustering.similarity_graph as sgk
from spectral_clustering.Viterbi import viterbi_workflow
from spectral_clustering.display_and_export import display_and_export_quotient_graph_matplotlib
from spectral_clustering.display_and_export import export_fiedler_vector_on_pointcloud
from spectral_clustering.display_and_export import export_quotient_graph_attribute_on_point_cloud
from spectral_clustering.display_and_export import export_some_graph_attributes_on_point_cloud
from spectral_clustering.point_cloud_graph import PointCloudGraph
from spectral_clustering.quotient_graph import QuotientGraph
from spectral_clustering.quotientgraph_operations import collect_quotient_graph_nodes_from_pointcloudpoints_majority
from spectral_clustering.quotientgraph_operations import merge_one_class_QG_nodes
from spectral_clustering.quotientgraph_operations import transfer_quotientgraph_infos_on_riemanian_graph
from spectral_clustering.quotientgraph_semantics import determination_main_stem_shortest_paths_improved
from spectral_clustering.quotientgraph_semantics import minimum_spanning_tree_quotientgraph_semantics
from spectral_clustering.split_and_merge import opti_energy_dot_product
from spectral_clustering.split_and_merge import resegment_nodes_with_elbow_method
from spectral_clustering.split_and_merge import select_all_quotientgraph_nodes_from_pointcloudgraph_cluster
from spectral_clustering.split_and_merge import select_minimum_centroid_class
from spectral_clustering.topological_energy import define_and_optimize_topological_energy


@click.command(help="Perform semantic and instance segmentation of 3D plant point cloud.")
@click.argument("pcd-path", type=click.Path(exists=True))
@click.option("-r", "--radius", type=int, default=18,
              help="Parameter for nearest neighbors in graph creation.")
@click.option("-p", "--root_point_riemanian", type=int, default=34586,
              help="A root point index in the point cloud, a starting point in the tree structure.")
def main(pcd_path, radius=18, root_point_riemanian=34586):
    """Perform semantic and instance segmentation of 3D plant point cloud.

    Parameters
    ----------
    pcd_path : str or pathlib.Path
        A path to a point cloud to process.
    radius : int, optional
        Parameter for nearest neighbors in graph creation.
    root_point_riemanian : int, optional
        A root point index in the point cloud, a starting point in the tree structure.
    """
    begin = time.time()  # Start timer to measure total execution time

    # Load the input point cloud:
    # "Data/chenos/cheno_A_2021_04_19.ply"
    try:
        pcd = o3d.io.read_point_cloud(Path(pcd_path), format='ply')
    except Exception as e:
        click.echo(f"Error reading point cloud file at {pcd_path}: {e}")
        return
    root_pcd_path = Path(pcd_path).parent

    # -----------------------------------------------------------------------------
    # Step 1: Build a Riemannian similarity graph based on the input point cloud
    # -----------------------------------------------------------------------------
    SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=radius)

    # Wrap the similarity graph in a PointCloudGraph object for easier manipulation
    pcd_g = PointCloudGraph(SimG)
    pcd_g.pcd = pcdfinal  # Store the processed point cloud in the graph for later use

    # Compute graph eigenvalues and eigenvectors to extract spectral information
    pcd_g.compute_graph_eigenvectors()
    # Export the Fiedler vector (representing low-frequency spectral properties) to a file
    export_fiedler_vector_on_pointcloud(pcd_g, filename=root_pcd_path / "pcd_vp2.txt")

    # Calculate the gradient of the Fiedler vector to highlight changes in spectral information
    pcd_g.compute_gradient_of_fiedler_vector(method='by_fiedler_weight')
    # Cluster the nodes in the graph using k-means based on Fiedler vector gradient norm
    clusters_centers = pcd_g.clustering_by_kmeans_using_gradient_norm(export_in_labeled_point_cloud=True,
                                                                      number_of_clusters=4)

    # Create a QuotientGraph (abstract representation of clustered nodes)
    qg = QuotientGraph()
    qg.build_from_pointcloudgraph(pcd_g, pcd_g.kmeans_labels_gradient)

    # Export quotient graph node attribute to a file for visualization or further analysis
    export_some_graph_attributes_on_point_cloud(qg.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "pcd_attribute.txt")

    time1 = time.time()  # Measure time taken for the initial graph processing steps

    # -----------------------------------------------------------------------------
    # Step 2: Identify and process leaf clusters in the graph
    # -----------------------------------------------------------------------------
    # Identify the label corresponding to the cluster with the minimal centroid (usually a "leaf" cluster)
    label_leaves = select_minimum_centroid_class(clusters_centers)
    # Collect the corresponding quotient graph nodes associated with the identified label
    list_leaves = select_all_quotientgraph_nodes_from_pointcloudgraph_cluster(pcd_g, qg, label_leaves)
    # List of individual points associated with the leaf nodes (non-clustered graph level)
    list_leaves_point = []
    for qnode in list_leaves:
        list_of_nodes_each = [x for x, y in pcd_g.nodes(data=True) if y['quotient_graph_node'] == qnode]
        list_leaves_point += list_of_nodes_each

    # Perform re-segmentation of the leaf nodes using the elbow method (optimal number of clusters is determined)
    resegment_nodes_with_elbow_method(qg, QG_nodes_to_rework=list_leaves, number_of_cluster_tested=10,
                                      attribute='norm_gradient', number_attribute=1, standardization=False)

    # Rebuild the quotient graph to reflect the updated segmentation
    qg.rebuild(qg.point_cloud_graph)
    # Export updated node attributes after re-segmentation
    export_some_graph_attributes_on_point_cloud(qg.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "pcd_attribute_after_resegment_leaves_norm.txt")

    time2 = time.time()  # Measure time for segmentation operations
    end = time2 - time1
    print(end)

    # -----------------------------------------------------------------------------
    # Step 3: Optimize topological energy for refined clustering
    # -----------------------------------------------------------------------------
    # Define and optimize the topological energy function for better clustering results
    define_and_optimize_topological_energy(quotient_graph=qg,
                                           point_cloud_graph=pcd_g,
                                           exports=True,
                                           formulae='improved',
                                           number_of_iteration=10000,
                                           choice_of_node_to_change='max_energy')

    # Rebuild the quotient graph after energy optimization
    qg.rebuild(qg.point_cloud_graph)
    # Collect updated leaf nodes after energy optimization
    list_leaves = collect_quotient_graph_nodes_from_pointcloudpoints_majority(quotient_graph=qg,
                                                                              list_of_points=list_leaves_point)

    # -----------------------------------------------------------------------------
    # Step 4: Re-process linear nodes (non-leaf clusters) in the graph
    # -----------------------------------------------------------------------------
    # Create a list of non-leaf nodes to re-segment using direction gradient
    list_of_linear = []
    for n in qg.nodes:
        if n not in list_leaves:
            list_of_linear.append(n)

    # Re-segment linear (non-leaf) nodes using the elbow method based on direction gradient
    resegment_nodes_with_elbow_method(qg, QG_nodes_to_rework=list_of_linear, number_of_cluster_tested=10,
                                      attribute='direction_gradient', number_attribute=3, standardization=False)
    # Rebuild the quotient graph after re-segmentation
    qg.rebuild(qg.point_cloud_graph)

    # Export attributes of the linear nodes after re-segmentation
    export_some_graph_attributes_on_point_cloud(qg.point_cloud_graph,
                                                graph_attribute="quotient_graph_node",
                                                filename=root_pcd_path / "reseg_linear.txt")

    # -----------------------------------------------------------------------------
    # Step 5: Additional topological energy optimization
    # -----------------------------------------------------------------------------
    # Perform another round of topological energy optimization
    define_and_optimize_topological_energy(quotient_graph=qg,
                                           point_cloud_graph=pcd_g,
                                           exports=True,
                                           formulae='improved',
                                           number_of_iteration=5000,
                                           choice_of_node_to_change='max_energy')

    # Rebuild the quotient graph again
    qg.rebuild(qg.point_cloud_graph)

    # Collect updated leaf nodes
    list_leaves = collect_quotient_graph_nodes_from_pointcloudpoints_majority(quotient_graph=qg,
                                                                              list_of_points=list_leaves_point)
    # Add direction information to the quotient graph nodes
    qg.compute_direction_info(list_leaves=list_leaves)

    # Optimize energy dot product considering angular constraints
    opti_energy_dot_product(quotientgraph=qg, subgraph_riemannian=pcd_g, angle_to_stop=30, export_iter=True,
                            list_leaves=list_leaves)

    # Compute coordinates and local descriptors for the quotient graph nodes
    qg.compute_nodes_coordinates()
    qg.compute_local_descriptors()

    # Export quotient graph visualizations for different attributes
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="planarity",
                                                 data_on_nodes='planarity', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="linearity",
                                                 data_on_nodes='linearity', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20, name="scattering",
                                                 data_on_nodes='scattering', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    # Export node attributes back to the original point cloud
    export_quotient_graph_attribute_on_point_cloud(qg, 'planarity2')
    export_quotient_graph_attribute_on_point_cloud(qg, 'linearity')
    export_quotient_graph_attribute_on_point_cloud(qg, 'scattering')

    # Generate final visualization of the complete quotient graph
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20,
                                                 name="quotient_graph_matplotlib_final",
                                                 data_on_nodes='intra_class_node_number', data=False,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    # Create a minimum spanning tree of the quotient graph for further analysis
    QG_t2 = minimum_spanning_tree_quotientgraph_semantics(qg)

    # Visualize the quotient graph with nodes labeled by quotient graph attributes
    display_and_export_quotient_graph_matplotlib(qg=QG_t2, node_sizes=20, name="quotient_graph_matplotlib",
                                                 data_on_nodes='quotient_graph_node', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    end = time.time()  # End total execution timer

    timefinal = end - begin  # Total runtime

    # -----------------------------------------------------------------------------
    # Step 6: Perform Viterbi workflow for semantic analysis of quotient graph
    # -----------------------------------------------------------------------------
    # Retrieve the root node's quotient graph label
    rt = pcd_g.nodes[root_point_riemanian]['quotient_graph_node']
    # Perform Viterbi workflow for semantic analysis and classification of graph nodes
    viterbi_workflow(minimum_spanning_tree=QG_t2,
                     quotient_graph=qg,
                     root=rt,
                     observation_list_import=['planarity2', 'linearity'],
                     initial_distribution=[1, 0],
                     transition_matrix=[[0.2, 0.8], [0, 1]],
                     parameters_emission=[[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]]
                     )

    # Export the Viterbi classification result back to the point cloud
    export_quotient_graph_attribute_on_point_cloud(qg, attribute='viterbi_class', name='first_viterbi')

    # Identify leaves and linear parts based on Viterbi classification
    list_of_leaves = [x for x, y in qg.nodes(data=True) if y['viterbi_class'] == 1]
    list_of_linear = [x for x, y in qg.nodes(data=True) if y['viterbi_class'] == 0]

    # Transfer Viterbi information back to the Riemannian (original) graph
    transfer_quotientgraph_infos_on_riemanian_graph(QG=qg, info='viterbi_class')

    # Identify stems in the quotient graph based on shortest paths starting from the root point
    list_of_stem = determination_main_stem_shortest_paths_improved(
        QG=qg,
        ptsource=root_point_riemanian,  # Root point for calculation
        list_of_linear_QG_nodes=list_of_linear,  # Use linear parts as reference
        angle_to_stop=45  # Threshold angle for determining stem continuation
    )

    # Export stem detection information back to the point cloud
    export_quotient_graph_attribute_on_point_cloud(qg, attribute='viterbi_class', name='stem_detect')

    # Export additional attributes related to gradient direction
    export_quotient_graph_attribute_on_point_cloud(qg, attribute="dir_gradient_stdv")
    export_quotient_graph_attribute_on_point_cloud(qg, attribute="dir_gradient_angle_mean")

    # Visualize the quotient graph with Viterbi classification results
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20,
                                                 name="quotient_graph_matplotlib_viterbi_stem",
                                                 data_on_nodes='viterbi_class', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    # Another visualization: Graph nodes labeled by mean values of node attributes
    display_and_export_quotient_graph_matplotlib(qg, node_sizes=20,
                                                 name="quotient_graph_matplotlib_nodesnumber",
                                                 data_on_nodes='quotient_graph_node_mean', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    # Merge specific classes of nodes based on their Viterbi classification
    # Here, only 'leaves' and any custom classes (like class 3) are merged.QG_merge = merge_one_class_QG_nodes(QG, attribute='viterbi_class', viterbiclass=[1, 3])
    QG_merge = merge_one_class_QG_nodes(qg, attribute='viterbi_class', viterbiclass=[1, 3])

    # Export merged attributes back to the point cloud for inspection
    export_some_graph_attributes_on_point_cloud(pcd_g=QG_merge.point_cloud_graph,
                                                graph_attribute='quotient_graph_node',
                                                filename='Merge_leaves_after_viterbi.txt')

    # Export the Viterbi classification result post-merge
    export_quotient_graph_attribute_on_point_cloud(qg, attribute='viterbi_class')

    # Re-visualize the quotient graph after merging leaves and additional classes
    display_and_export_quotient_graph_matplotlib(qg=QG_merge, node_sizes=20,
                                                 name="quotient_graph_matplotlib_viterbi_stemmerge",
                                                 data_on_nodes='viterbi_class', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)

    # Final visualization: Quotient graph nodes with mean values of updated attributes
    display_and_export_quotient_graph_matplotlib(qg=QG_merge, node_sizes=20,
                                                 name="quotient_graph_matplotlib_nodesnumbermerge",
                                                 data_on_nodes='quotient_graph_node_mean', data=True,
                                                 attributekmeans4clusters=False, directory=root_pcd_path)


if __name__ == "__main__":
    main()
