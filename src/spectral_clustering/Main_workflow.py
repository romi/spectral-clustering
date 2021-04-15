





pcd = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Data/Older_Cheno.ply", format='ply')
r = 18
G = kpcg.PointCloudGraph()
G.PointCloudGraph_init_with_pcd(point_cloud=pcd, method='knn', nearest_neighbors=r)
print(nx.is_connected(G))
if nx.is_connected(G) is False:
    largest_cc = max(nx.connected_components(G), key=len)
    # creating the new pcd point clouds
    coords = np.zeros((len(largest_cc), 3))
    i = 0
    for node in largest_cc:
        coords[i, :] = G.nodes[node]['pos']
        i += 1
    np.savetxt('New_pcd_connected.txt', coords, delimiter=' ', fmt='%f')
    pcd2 = open3d.read_point_cloud("/Users/katiamirande/PycharmProjects/Spectral_clustering_0/Src/spectral_clustering/New_pcd_connected.txt", format='xyz')
    r = 18
    G = kpcg.PointCloudGraph()
    G.PointCloudGraph_init_with_pcd(point_cloud=pcd2, method='knn', nearest_neighbors=r)


G.compute_graph_eigenvectors()
G.compute_gradient_of_Fiedler_vector(method='by_Fiedler_weight')
G.clustering_by_kmeans_in_four_clusters_using_gradient_norm(export_in_labeled_point_cloud=True)
#G.clustering_by_fiedler_and_optics(criteria=np.multiply(G.direction_gradient_on_Fiedler_scaled, G.gradient_on_Fiedler))

QG = QuotientGraph()
QG.build_QuotientGraph_from_PointCloudGraph(G, G.kmeans_labels_gradient)

QG.init_topo_scores(QG.point_cloud_graph, exports=True, formulae='improved')

QG.optimization_topo_scores(G=QG.point_cloud_graph, exports=True, number_of_iteration=10000,
                            choice_of_node_to_change='max_energy', formulae='improved')

QG.rebuild_quotient_graph(QG.point_cloud_graph)

QG.segment_each_cluster_by_optics_using_directions()

QG.init_topo_scores(G=QG.point_cloud_graph, exports=True, formulae='improved')
QG.optimization_topo_scores(G=QG.point_cloud_graph, exports=True, number_of_iteration=1000,
                            choice_of_node_to_change='max_energy', formulae='improved')
QG.rebuild_quotient_graph(QG.point_cloud_graph)

list_of_nodes_to_work = QG.oversegment_part(list_quotient_node_to_work=[10], average_size_cluster=50)
QG.compute_direction_info(list_leaves=[])
QG.opti_energy_dot_product(energy_to_stop=0.29, leaves_out=False, list_graph_node_to_work=list_of_nodes_to_work)
