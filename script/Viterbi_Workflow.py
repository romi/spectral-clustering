########### Imports


from treex import *
from treex.simulation import *
from treex.simulation.galton_watson import __discrete_distribution  # only for generating discrete observations
from spectral_clustering.Original_TreeX import *
from spectral_clustering.display_and_export import *

import time

import spectral_clustering.similarity_graph as sgk
from spectral_clustering.Viterbi import *
from spectral_clustering.point_cloud_graph import *
from spectral_clustering.quotient_graph import *
from spectral_clustering.topological_energy import *
from spectral_clustering.segmentation_algorithms import *
from spectral_clustering.split_and_merge import *
from spectral_clustering.display_and_export import *
from spectral_clustering.quotientgraph_semantics import *


a = time.time()
file1 = "/Users/katiamirande/Documents/Tests/Viterbi/chénopode/cheno_C_2021_04_07_small/seg/New_pcd_connected.ply"
pcd = o3d.io.read_point_cloud(file1, format='ply')
r = 18
SimG, pcdfinal = sgk.create_connected_riemannian_graph(point_cloud=pcd, method='knn', nearest_neighbors=r, radius=r)
G = PointCloudGraph(SimG)
G.pcd = pcdfinal
file2 = "/Users/katiamirande/Documents/Tests/Viterbi/chénopode/cheno_C_2021_04_07_small/seg/final_split_and_merge.txt"
sgk.add_label_from_pcd_file(G, file=file2)
b = time.time()
print(b-a)
QG = QuotientGraph()
QG.build_from_pointcloudgraph(G, G.clustering_labels)
c = time.time()
print(c-b)
QG.compute_nodes_coordinates()
QG.compute_local_descriptors(method='all_qg_cluster', data='coords', neighborhood='pointcloudgraph', scale = 10)
d = time.time()
print(d-c)








export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="planarity",
                                            filename="G_planarity.txt")
export_some_graph_attributes_on_point_cloud(QG.point_cloud_graph,
                                            graph_attribute="planarity",
                                            filename="G_linearity.txt")

display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name="planarity",
                                             data_on_nodes='planarity', data=True, attributekmeans4clusters=False)
display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name="linearity",
                                             data_on_nodes='linearity', data=True, attributekmeans4clusters=False)
display_and_export_quotient_graph_matplotlib(qg=QG, node_sizes=20, name="scattering",
                                             data_on_nodes='scattering', data=True, attributekmeans4clusters=False)
export_quotient_graph_attribute_on_point_cloud(QG, 'planarity')
export_quotient_graph_attribute_on_point_cloud(QG, 'linearity')
export_quotient_graph_attribute_on_point_cloud(QG, 'scattering')
QG_t2 = minimum_spanning_tree_quotientgraph_semantics(QG)

st_tree = read_pointcloudgraph_into_treex(pointcloudgraph=QG_t2)
rt = 8
t = build_spanning_tree(st_tree, rt, list_att=['planarity', 'linearity', 'scattering'])

create_observation_list(t, list_obs=['planarity', 'linearity', 'scattering'])

#########################################################################

initial_distribution = [1, 0, 0]
transition_matrix = [[0.2, 0, 0.8], [0, 0.8, 0.2], [0, 0.8, 0.2]]
# transition_matrix = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
continuous_obs = True
if continuous_obs:  # observations are Gaussian
    parameterstot = [[[0.1, 0.9], [0.8, 0.9], [0.01, 0.3]], [[0.8, 0.9], [0.5, 0.9], [0.01, 0.3]], [[0.1, 0.9], [0.9, 0.9], [0.1, 0.3]]]
    # parameterstot = [[[0.73, 0.1], [0.25, 0.1], [0.03, 0.05]], [[0.66, 0.04], [0.33, 0.05], [0.0001, 0.1]],
    #                 [[0.10, 0.20], [0.6, 0.35], [0.20, 0.1]]]
    # parameterstot = [[[0.8, 0.6], [0.2, 0.3], [0.8, 0.2]]]
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
display_and_export_quotient_graph_matplotlib(qg=QG_t2, node_sizes=20, name="quotient_graph_viterbi",
                                             data_on_nodes='viterbi_class', data=True, attributekmeans4clusters=False)

export_quotient_graph_attribute_on_point_cloud(QG, attribute='viterbi_class')
