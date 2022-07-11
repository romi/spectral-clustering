import operator
from sklearn import *
from spectral_clustering.display_and_export import *
from spectral_clustering.split_and_merge import *
from random import choice
from spectral_clustering.dijkstra_segmentation import *
import copy
from sklearn.metrics.cluster import rand_score
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
                                 name_model='name_model',
                                 limb_gt=2, cotyledon_gt=3, main_stem_gt=1, petiole_gt=4,
                                 class_limb=1, class_mainstem=3, class_petiol=5, class_branch=6, class_apex=4):
    xc, yc, zc, labelsc = np.loadtxt(fname = file_semantic_results, delimiter=',', unpack=True)
    xt, yt, zt = np.loadtxt(fname = file_ground_truth_coord, delimiter=',', unpack=True)
    labelst = np.loadtxt(fname = file_ground_truth_labels, delimiter=',', unpack=True)
    exp = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelst[:, np.newaxis]), axis=1)
    np.savetxt('Ground_truth_virtual'+name_model+'.txt', exp, delimiter=' ', fmt='%f')

    #creation d'une liste ground truth correspondant aux labels.
    label_gt_end = copy.deepcopy(labelst)
    for i in range(len(label_gt_end)):
        if label_gt_end[i] == limb_gt:
            new = class_limb
        elif label_gt_end[i] == main_stem_gt:
            new = class_mainstem
        elif label_gt_end[i] == petiole_gt:
            new = class_petiol
        #elif label_gt_end[i] == petiole_gt:
        #    new = class_mainstem
        elif label_gt_end[i] == cotyledon_gt:
            new = class_limb
        label_gt_end[i] = new

    gt_final = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], label_gt_end[:, np.newaxis]), axis=1)
    np.savetxt('Ground_truth_final' + name_model + '.txt', gt_final, delimiter=' ', fmt='%f')
    #ici j'ai trop de différents labels par rapport à la vérité terrain, a enlever si c'est ok entre les deux

    for i in range(len(labelsc)):
        if labelsc[i] == class_branch:
            new = class_mainstem
            labelsc[i] = new
        elif labelsc[i] == class_apex:
            new = class_limb
            labelsc[i] = new
        #elif labelsc[i] == class_petiol:
        #    new = class_mainstem
        #    labelsc[i] = new

    label_final = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelsc[:, np.newaxis]),axis=1)
    np.savetxt('Label_final' + name_model + '.txt', label_final, delimiter=' ', fmt='%f')

    mres = np.zeros((len(set(label_gt_end)) + 4, 8))
    #mres[0, 1] = 'TP'
    #mres[0, 2] = 'FN'
    #mres[0, 3] = 'FP'
    #mres[0, 4] = 'Re'
    #mres[0, 5] = 'Pr'
    #mres[0, 6] = 'IoU'
    #faire une vérification que les coordoonnées correspondent ?
    TP = dict()
    FN = dict()
    FP = dict()
    TN = dict()
    a = 1
    for i in set(label_gt_end):
        TP[i] = 0
        FN[i] = 0
        FP[i] = 0
        TN[i] = 0
        mres[a, 0] = i
        a += 1



    for i in range(len(labelsc)):
        labelgiven = labelsc[i]
        labeltruth = label_gt_end[i]
        if labelsc[i] == label_gt_end[i]:
            TP[labelgiven] += 1
        else:
            FP[labelgiven] += 1
            FN[labeltruth] += 1
        for c in set(label_gt_end):
            if c != labelsc[i] and c != label_gt_end[i]:
                TN[c] += 1


    Re = dict()
    Pr = dict()
    IoU = dict()
    TPtot = 0
    FNtot = 0
    FPtot = 0
    TNtot = 0
    MIoU = 0
    for i in set(label_gt_end):
        Re[i] = (TP[i]) / (TP[i]+FN[i])
        Pr[i] = (TP[i]) / (TP[i]+FP[i])
        IoU[i] = (TP[i]) / (TP[i]+FN[i]+FP[i])
        TPtot += TP[i]
        FNtot += FN[i]
        FPtot += FP[i]
        TNtot += TN[i]
        MIoU += IoU[i]

    MIoU /= len(set(label_gt_end))
    totalacc = (TPtot+TNtot) / (TPtot +TNtot + FPtot +FNtot)
    f1_score = TPtot / (TPtot + 0.5*(FNtot + FPtot))

    a = 1
    for i in set(label_gt_end):
        mres[a, 1] = TP[i]
        mres[a, 2] = FN[i]
        mres[a, 3] = FP[i]
        mres[a, 4] = TN[i]
        mres[a, 5] = Re[i]
        mres[a, 6] = Pr[i]
        mres[a, 7] = IoU[i]
        a += 1
    mres[len(set(label_gt_end)) + 1, 1] = TPtot
    mres[len(set(label_gt_end)) + 1, 2] = FNtot
    mres[len(set(label_gt_end)) + 1, 3] = FPtot
    mres[len(set(label_gt_end)) + 1, 4] = TNtot
    mres[len(set(label_gt_end)) + 1, 7] = MIoU
    mres[len(set(label_gt_end)) + 2, 1] = totalacc
    mres[len(set(label_gt_end)) + 3, 1] = f1_score

    cm = metrics.confusion_matrix(label_gt_end, labelsc)
    np.savetxt(name_model+'scikit_cm', cm, fmt='%.4e')
    np.savetxt(name_model + 'eval.txt', mres, fmt='%.4e')


def change_labels(file_semantic_results="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/script/pcd_viterbi_classsemantic_final.txt",
                    name_model='name_model',
                    class_limb=1, class_mainstem=3, class_petiol=5, class_branch=6, class_apex=4):

    xc, yc, zc, labelsc = np.loadtxt(fname=file_semantic_results, delimiter=',', unpack=True)
    for i in range(len(labelsc)):
        if labelsc[i] == class_branch:
            new = class_mainstem
            labelsc[i] = new
        elif labelsc[i] == class_apex:
            new = class_limb
            labelsc[i] = new
        elif labelsc[i] == class_petiol:
            new = class_mainstem
            labelsc[i] = new

    label_final = np.concatenate((xc[:, np.newaxis], yc[:, np.newaxis], zc[:, np.newaxis], labelsc[:, np.newaxis]),axis=1)
    np.savetxt('Label_final' + name_model + '.txt', label_final, delimiter=' ', fmt='%f')


def downsample_pcd(file_pcd="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/script/pcd_viterbi_classsemantic_final.txt",
                    name_model='name_model'):

    pcd = open3d.read_point_cloud(file_pcd, format='ply')
    downpcd = open3d.voxel_down_sample_and_trace(input=pcd, voxel_size=1.0, approximate_class=True)
    open3d.write_point_cloud(name_model+"down_sample.ply", downpcd)

def compute_recall_precision_IoU_real_plants(file_semantic_results="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/script/pcd_viterbi_classsemantic_final.txt",
                                             file_instance_results="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/script/pcd_viterbi_classsemantic_final.txt",
                                 file_ground_truth="/Users/katiamirande/PycharmProjects/Spectral_clustering_0/script/cheno_virtuel_coordinates.txt",
                                 name_model= "name",
                                 class_limb=1, class_mainstem=3, class_petiol=5, class_branch=6, class_apex=4):
    xc, yc, zc, labelsc = np.loadtxt(fname = file_semantic_results, delimiter=',', unpack=True)
    xc, yc, zc, labelinstance = np.loadtxt(fname = file_instance_results, delimiter=',', unpack=True)
    xt, yt, zt, labelst, np2,  c1,c2,c3 = np.loadtxt(fname = file_ground_truth, delimiter=',', unpack=True)
    exp = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelst[:, np.newaxis]), axis=1)
    np.savetxt('Ground_truth_'+name_model+'.txt', exp, delimiter=' ', fmt='%f')

    rand = metrics.cluster.rand_score(labelst, labelinstance)
    print("rand")
    print(rand)
    rand_adj = metrics.cluster.adjusted_rand_score(labelst, labelinstance)
    print("rand_adjusted")
    print(rand_adj)
    mutual= metrics.adjusted_mutual_info_score(labelst, labelinstance)
    print("adjusted_mutual_info_score")
    print(mutual)
    comp = metrics.completeness_score(labelst, labelinstance)
    print("completeness_score")
    print(comp)
    fowlkes = metrics.fowlkes_mallows_score(labelst, labelinstance)
    print("fowlkes_mallows_score")
    print(fowlkes)
    homogeneity = metrics.homogeneity_score(labelst, labelinstance)
    print("homogeneity_score")
    print(homogeneity)

    #creation d'une liste ground truth correspondant aux labels.
    label_gt_end = copy.deepcopy(labelst)
    for i in range(len(label_gt_end)):
        if 399 < label_gt_end[i] < 500:
            new = class_limb
        elif label_gt_end[i] == 0:
            new = class_mainstem
        elif 199 < label_gt_end[i] < 300:
            new = class_petiol
        elif 299 < label_gt_end[i] < 399:
            new = class_apex
        elif 99 < label_gt_end[i] < 199:
            new = class_branch
        elif 999 < label_gt_end[i]:
            new = class_limb
        label_gt_end[i] = new

    gt_final = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], label_gt_end[:, np.newaxis]), axis=1)
    np.savetxt('Ground_truth_final' + name_model + '.txt', gt_final, delimiter=' ', fmt='%f')
    #ici j'ai trop de différents labels par rapport à la vérité terrain, a enlever si c'est ok entre les deux
    """
    for i in range(len(labelsc)):
        if labelsc[i] == class_branch:
            new = class_mainstem
            labelsc[i] = new
        elif labelsc[i] == class_apex:
            new = class_limb
            labelsc[i] = new
        #elif labelsc[i] == class_petiol:
        #    new = class_mainstem
        #    labelsc[i] = new
    """
    if set(label_gt_end) > set(labelsc):
        list = set(label_gt_end)
    else:
        list = set(labelsc)
    label_final = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelsc[:, np.newaxis]),axis=1)
    np.savetxt('Label_final' + name_model + '.txt', label_final, delimiter=' ', fmt='%f')

    mres = np.zeros(((len(list)) + 4, 8))
    #mres[0, 1] = 'TP'
    #mres[0, 2] = 'FN'
    #mres[0, 3] = 'FP'
    #mres[0, 4] = 'Re'
    #mres[0, 5] = 'Pr'
    #mres[0, 6] = 'IoU'
    #faire une vérification que les coordoonnées correspondent ?
    TP = dict()
    FN = dict()
    FP = dict()
    TN = dict()
    a = 1

    for i in list:
        TP[i] = 0
        FN[i] = 0
        FP[i] = 0
        TN[i] = 0
        mres[a, 0] = i
        a += 1


    for i in range(len(labelsc)):
        labelgiven = labelsc[i]
        labeltruth = label_gt_end[i]
        if labelsc[i] == label_gt_end[i]:
            TP[labelgiven] += 1
        else:
            FP[labelgiven] += 1
            FN[labeltruth] += 1
        for c in set(label_gt_end):
            if c != labelsc[i] and c != label_gt_end[i]:
                TN[c] += 1


    Re = dict()
    Pr = dict()
    IoU = dict()
    TPtot = 0
    FNtot = 0
    FPtot = 0
    TNtot = 0
    MIoU = 0
    for i in set(label_gt_end):
        Re[i] = (TP[i]) / (TP[i]+FN[i])
        Pr[i] = (TP[i]) / (TP[i]+FP[i])
        IoU[i] = (TP[i]) / (TP[i]+FN[i]+FP[i])
        TPtot += TP[i]
        FNtot += FN[i]
        FPtot += FP[i]
        TNtot += TN[i]
        MIoU += IoU[i]

    MIoU /= len(set(label_gt_end))
    totalacc = (TPtot+TNtot) / (TPtot +TNtot + FPtot +FNtot)
    f1_score = TPtot / (TPtot + 0.5*(FNtot + FPtot))

    a = 1
    for i in set(label_gt_end):
        mres[a, 1] = TP[i]
        mres[a, 2] = FN[i]
        mres[a, 3] = FP[i]
        mres[a, 4] = TN[i]
        mres[a, 5] = Re[i]
        mres[a, 6] = Pr[i]
        mres[a, 7] = IoU[i]
        a += 1
    mres[len(list) + 1, 1] = TPtot
    mres[len(list) + 1, 2] = FNtot
    mres[len(list) + 1, 3] = FPtot
    mres[len(list) + 1, 4] = TNtot
    mres[len(list) + 1, 7] = MIoU
    mres[len(list) + 2, 1] = totalacc
    mres[len(list) + 3, 1] = f1_score

    cm = metrics.confusion_matrix(label_gt_end, labelsc)
    np.savetxt(name_model+'scikit_cm', cm, fmt='%.4e')
    np.savetxt(name_model + 'eval.txt', mres, fmt='%.4e')

