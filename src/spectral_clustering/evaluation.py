import operator
from spectral_clustering.display_and_export import *
from spectral_clustering.split_and_merge import *
from random import choice
from spectral_clustering.dijkstra_segmentation import *

#Evaluations functions

def count_number_limbs_apex_etc(QG, class_apex=4, class_limb=1, attribute='viterbi_class'):
    list_of_limb = [x for x, y in QG.nodes(data=True) if y[attribute] == class_limb]
    list_of_apex = [x for x, y in QG.nodes(data=True) if y[attribute] == class_apex]

    return (len(list_of_apex, len(list_of_limb)))

def length_main_stem(QG, class_main_stem=3, attribute='viterbi_class'):
    G = QG.point_cloud_graph
    list_QG_nodes = [x for x, y in QG.nodes(data=True) if y[attribute] == class_main_stem]
    # get all points from stem
    list_G_nodes = list()
    for qgn in list_QG_nodes:
        add_list = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qgn]
        list_G_nodes += add_list
    #make subgraph
    print(list_G_nodes)
    substem = G.subgraph(list_G_nodes)

    #find longest of shortest paths
    random_node = choice(list(substem.nodes))
    dict = nx.single_source_dijkstra_path_length(substem, random_node, weight='distance')
    ptsource = max(dict.items(), key=operator.itemgetter(1))[0]
    # Stockage de tous les poids des plus courts chemin entre le précédent point et l'ensemble des points du graphe
    dict = nx.single_source_dijkstra_path_length(substem, ptsource, weight='distance')
    # Isolation du point le plus éloigné
    ptarrivee = max(dict.items(), key=operator.itemgetter(1))[0]
    # Obtention du chemin entre le point source et le point d'arrivée finaux
    segmsource = nx.dijkstra_path(substem, ptsource, ptarrivee, weight='distance')

    Gaffichage = nx.Graph()
    pts = np.array(G.pcd.points)
    N = len(G.pcd.points)
    for i in range(N):
        Gaffichage.add_node(i, pos=pts[i])
    nx.add_path(Gaffichage, segmsource)
    edgelist = Gaffichage.edges
    graph = open3d.geometry.LineSet()
    graph.points = open3d.Vector3dVector(pts)
    graph.lines = open3d.Vector2iVector(edgelist)
    open3d.draw_geometries([graph, pcd])
    #compute the length
    l = 0
    for i in range(len(segmsource)-1):
        l += G.edges[segmsource[i], segmsource[i+1]]["distance"]

    return l

def export_each_element_point_cloud(QG, class_to_export=1, attribute='viterbi_class', name='limb_piece'):
    G = QG.point_cloud_graph
    list = [x for x, y in QG.nodes(data=True) if y[attribute] == class_to_export]
    for n in list:
        list_points = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        H = G.subgraph(list_points)
        labels_from_qg = np.zeros((len(H.nodes), 4))
        i = 0
        for n in H.nodes:
            labels_from_qg[i, 0:3] = G.nodes[n]['pos']
            labels_from_qg[i, 3] = G.nodes[n]['quotient_graph_node']
            i += 1
        np.savetxt('pcd_' + name + str(n) +'_'+ str(i) + '.txt', labels_from_qg, delimiter=",")


def resegment_apex_for_eval_and_export(QG, class_apex=4, attribute='viterbi_class', name='apex_piece'):
    list_of_apex = [x for x, y in QG.nodes(data=True) if y[attribute] == class_apex]

    #resegment with direction infos + elbow method
    for n in list_of_apex:
        resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=[n],
                                          number_of_cluster_tested=20,
                                          attribute='norm_gradient',
                                          number_attribute=1,
                                          standardization=False, numer=1, G_mod=False)

