import operator
from spectral_clustering.display_and_export import *
from spectral_clustering.split_and_merge import *
from random import choice
from spectral_clustering.dijkstra_segmentation import *
import copy

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
    open3d.draw_geometries([graph, G.pcd])
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
        labels_from_qg = np.zeros((len(H.nodes), 3))
        i = 0
        if len(H.nodes) > 50:
            for n in H.nodes:
                labels_from_qg[i, 0:3] = G.nodes[n]['pos']
                i += 1
            np.savetxt('pcd_' + name + str(n) +'_'+ str(i) + '.txt', labels_from_qg, delimiter=" ")


def resegment_apex_for_eval_and_export(QG, class_apex=4, attribute='viterbi_class', name='apex_piece', lim = 30):
    list_of_apex = [x for x, y in QG.nodes(data=True) if y[attribute] == class_apex]
    attribute_seg = 'direction_gradient'
    #resegment with direction infos + elbow method
    #for n in list_of_apex:
        #resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=[n],
        #                                  number_of_cluster_tested=20,
        #                                  attribute='direction_gradient',
        #                                  number_attribute=3,
        #                                  standardization=False, numer=1, G_mod=False, export_div=True)

    # Check if maximas has been computed or recompute them
    QG.count_local_extremum_of_Fiedler()
    # Resegment each apex with n = number of extremum and k-means ?
    for n in list_of_apex:
        number_clust = QG.nodes[n]['number_of_local_Fiedler_extremum']
        sub = create_subgraphs_to_work(quotientgraph=QG, list_quotient_node_to_work=[n])
        SG = sub
        list_nodes = list(SG.nodes)
        # creation of matrices to work with in the elbow_method package
        if len(SG.nodes) > number_clust and len(SG.nodes) > lim:
            Xcoord = np.zeros((len(SG), 3))
            for u in range(len(SG)):
                Xcoord[u] = SG.nodes[list(SG.nodes)[u]]['pos']
            Xnorm = np.zeros((len(SG), 3))
            for u in range(len(SG)):
                Xnorm[u] = SG.nodes[list(SG.nodes)[u]][attribute_seg]

            kmeans = skc.KMeans(n_clusters=number_clust, init='k-means++', n_init=20, max_iter=300,
                                tol=0.0001).fit(Xnorm)
            kmeans_labels = kmeans.labels_[:, np.newaxis]

            new_labels = np.zeros((len(SG.nodes), 4))
            new_labels[:, 0:3] = Xcoord
            for pt in range(len(list_nodes)):
                new_labels[pt, 3] = kmeans_labels[pt]
            np.savetxt('pcd_new_labels_' + str(n) + '.txt', new_labels, delimiter=",")
            indices = np.argsort(new_labels[:, 3])
            arr_temp = new_labels[indices]
            arr_split = np.array_split(arr_temp, np.where(np.diff(arr_temp[:, 3]) != 0)[0] + 1)
            v = 0
            for arr in arr_split:
                np.savetxt('pcd_new_labels_apex' + str(n) +'_'+ str(v) + '.txt', arr, delimiter=",")
                v += 1

    # exports


# Fonctions MIoU, etc etc


def compute_recall_precision_IoU(file_semantic_results="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/script/pcd_viterbi_classsemantic_final.txt",
                                 file_ground_truth_coord="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/script/cheno_virtuel_coordinates.txt",
                                 file_ground_truth_labels="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/script/cheno_virtuel_labels.txt",
                                 limb_gt=2, cotyledon_gt=3, main_stem_gt=1, petiole_gt=4,
                                 class_limb=1, class_mainstem=3, class_petiol=5, class_branch=6, class_apex=4):
    xc, yc, zc, labelsc = np.loadtxt(fname = file_semantic_results, delimiter=',', unpack=True)
    xt, yt, zt = np.loadtxt(fname = file_ground_truth_coord, delimiter=',', unpack=True)
    labelst = np.loadtxt(fname = file_ground_truth_labels, delimiter=',', unpack=True)
    exp = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelst[:, np.newaxis]), axis=1)
    np.savetxt('Ground_truth_virtual.txt', exp, delimiter=' ', fmt='%f')

    #creation d'une liste ground truth correspondant aux labels.
    label_gt_end = copy.deepcopy(labelst)
    for i in range(len(label_gt_end)):
        if label_gt_end[i] == limb_gt:
            new = class_limb
        elif label_gt_end[i] == main_stem_gt:
            new = class_mainstem
        elif label_gt_end[i] == petiole_gt:
            new = class_petiol
        elif label_gt_end[i] == cotyledon_gt:
            new = class_limb
        label_gt_end[i] = new

    #ici j'ai trop de différents labels par rapport à la vérité terrain, a enlever si c'est ok entre les deux

    for i in range(len(labelsc)):
        if labelsc[i] == class_branch:
            new = class_petiol
            labelsc[i] = new
        elif labelsc[i] == class_apex:
            new = class_limb
            labelsc[i] = new



    #faire une vérification que les coordoonnées correspondent ?
    TP = dict()
    FN = dict()
    FP = dict()
    for i in set(label_gt_end):
        TP[i] = 0
        FN[i] = 0
        FP[i] = 0


    for i in range(len(labelsc)):
        labelgiven = labelsc[i]
        labeltruth = label_gt_end[i]
        if labelsc[i] == label_gt_end[i]:
            TP[labelgiven] += 1
        else:
            FP[labelgiven] += 1
            FN[labeltruth] += 1

    Re = dict()
    Pr = dict()
    IoU = dict()
    TPtot = 0
    FNtot = 0
    FPtot = 0
    MIoU = 0
    for i in set(label_gt_end):
        Re[i] = (TP[i]) / (TP[i]+FN[i])
        Pr[i] = (TP[i]) / (TP[i]+FP[i])
        IoU[i] = (TP[i]) / (TP[i]+FN[i]+FP[i])
        TPtot += TP[i]
        FNtot += FN[i]
        FPtot += FP[i]
        MIoU += IoU[i]

    MIoU /= len(set(label_gt_end))
    totalacc = TPtot / (TPtot + FNtot + FPtot)






