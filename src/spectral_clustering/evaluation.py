#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import operator
from random import choice

import networkx as nx
import numpy as np
import open3d as o3d
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score

from spectral_clustering.split_and_merge import create_subgraphs_to_work


# Evaluations functions

def count_number_limbs_apex_etc(qg, class_apex=4, class_limb=1, attribute='viterbi_class'):
    """
    Counts the number of nodes classified as 'limb' and 'apex' based on a given attribute
    in a graph structure.

    This function iterates over the nodes of a given graph, evaluates their data based
    on a specified attribute, and counts the nodes that match the classification for
    'apex' and 'limb'. Returns a tuple of integers representing the count of 'apex'
    nodes and 'limb' nodes respectively.

    Parameters
    ----------
    qg : spectral_clustering.quotient_graph.QuotientGraph
        A graph object where each node contains data as a dictionary, accessed with
        the `attribute` parameter. The nodes are analyzed to determine if the provided
        `attribute` matches the classifications for 'apex' or 'limb'.
    class_apex : int, optional
        The value of the attribute that classifies a node as an 'apex' node. Defaults to 4.
    class_limb : int, optional
        The value of the attribute that classifies a node as a 'limb' node. Defaults to 1.
    attribute : str, optional
        The key name of the attribute in the node data dictionaries used to categorize
        nodes as 'apex' or 'limb'. Defaults to 'viterbi_class'.

    Returns
    -------
    tuple of int
        A tuple containing two integers:
        - The number of nodes classified as 'apex'.
        - The number of nodes classified as 'limb'.
    """
    list_of_limb = [x for x, y in qg.nodes(data=True) if y[attribute] == class_limb]
    list_of_apex = [x for x, y in qg.nodes(data=True) if y[attribute] == class_apex]

    return (len(list_of_apex, len(list_of_limb)))


def length_main_stem(qg, class_main_stem=3, attribute='viterbi_class'):
    """Computes the length of the main stem in a graph.

    This function identifies the main stem of a given graph by analyzing node
    properties and relationships in a quotient graph and its subgraph. It finds
    the longest of shortest paths in the subgraph, computes its length, and visualizes
    the path. It leverages Dijkstra's algorithm to determine the longest path distance.

    Parameters
    ----------
    qg : spectral_clustering.quotient_graph.QuotientGraph
        The quotient graph that contains nodes and their attributes used to identify
        the main stem.
    class_main_stem : int, optional
        The class used to filter nodes in the quotient graph. Only nodes from
        `gq` with an attribute value matching this class are considered for the
        analysis. Defaults to ``3``.
    attribute : str, optional
        The name of the node attribute in `gq` used to categorize nodes for
        identifying the main stem. Defaults to ``'viterbi_class'``.

    Returns
    -------
    float
        The length of the main stem, calculated as the sum of distances along
        the longest of shortest paths in the subgraph.
    """
    G = qg.point_cloud_graph
    list_QG_nodes = [x for x, y in qg.nodes(data=True) if y[attribute] == class_main_stem]
    # get all points from stem
    list_G_nodes = list()
    for qgn in list_QG_nodes:
        add_list = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == qgn]
        list_G_nodes += add_list
    # make subgraph
    print(list_G_nodes)
    substem = G.subgraph(list_G_nodes)

    # find longest of shortest paths
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

    graph = o3d.geometry.LineSet()
    graph.points = o3d.utility.Vector3dVector(pts)
    graph.lines = o3d.utility.Vector2iVector(edgelist)
    o3d.visualization.draw_geometries([graph, G.pcd])
    # compute the length
    l = 0
    for i in range(len(segmsource) - 1):
        l += G.edges[segmsource[i], segmsource[i + 1]]["distance"]

    return l


def export_each_element_point_cloud(qg, class_to_export=1, attribute='viterbi_class', name='limb_piece'):
    """
    Exports elements of a point cloud graph based on a specified class and attribute. For each matching element,
    the method identifies corresponding nodes, creates a subgraph, and saves positional data
    to a text file if the subgraph contains more than 50 nodes.

    Parameters
    ----------
    qg : spectral_clustering.quotient_graph.QuotientGraph
        The quotient graph containing nodes and their associated attributes.
    class_to_export : int, optional
        The specific class value in the `attribute` of the nodes of `gq` to export.
        Defaults to ``1``.
    attribute : str, optional
        The attribute of the nodes in `gq` to match against `class_to_export`
        for filtering. Defaults to ``'viterbi_class'``.
    name : str, optional
        The name used for the output file prefix. Defaults to ``'limb_piece'``.
    """
    G = qg.point_cloud_graph
    list = [x for x, y in qg.nodes(data=True) if y[attribute] == class_to_export]
    for n in list:
        list_points = [x for x, y in G.nodes(data=True) if y['quotient_graph_node'] == n]
        H = G.subgraph(list_points)
        labels_from_qg = np.zeros((len(H.nodes), 3))
        i = 0
        if len(H.nodes) > 50:
            for n in H.nodes:
                labels_from_qg[i, 0:3] = G.nodes[n]['pos']
                i += 1
            np.savetxt('pcd_' + name + str(n) + '_' + str(i) + '.txt', labels_from_qg, delimiter=" ")


def resegment_apex_for_eval_and_export(qg, class_apex=4, attribute='viterbi_class', name='apex_piece', lim=30):
    """Resegments specific nodes in a quotient graph for evaluation and export purposes.

    This function operates on a quotient graph (QG) to identify specific nodes
    based on their attributes and class values. It performs resegmentation for
    nodes matching specified criteria using k-means clustering. Furthermore, it
    leverages elbow methods, computes maxima if required, and exports the
    resegmented data.

    Parameters
    ----------
    qg : spectral_clustering.quotient_graph.QuotientGraph
        A quotient graph on which resegmentation and transformations will be
        performed.
    class_apex : int, optional
        The value of the `attribute` to identify nodes in the quotient graph for
        resegmentation. Default is ``4``.
    attribute : str, optional
        Node attribute to match for identifying nodes for resegmentation.
        Default is ``'viterbi_class'``.
    name : str, optional
        Base name used for exporting file outputs of resegmented labels.
        Default is ``'apex_piece'``.
    lim : int, optional
        Threshold to limit the number of nodes resegmented. If the number of
        nodes in the subgraph exceeds this value, resegmentation is performed.
        Default is ``30``.

    Notes
    -----
    - This function assumes that the quotient graph (QG) has been preprocessed
      and contains required information such as local Fiedler extrema counts
      for nodes.
    - Node positions and direction gradients are used internally during the
      k-means clustering process.
    - File exports are automatically handled for each resegmented apex. Resulting
      files contain label data produced by k-means clustering.

    """
    list_of_apex = [x for x, y in qg.nodes(data=True) if y[attribute] == class_apex]
    attribute_seg = 'direction_gradient'
    # resegment with direction infos + elbow method
    # for n in list_of_apex:
    # resegment_nodes_with_elbow_method(QG, QG_nodes_to_rework=[n],
    #                                  number_of_cluster_tested=20,
    #                                  attribute='direction_gradient',
    #                                  number_attribute=3,
    #                                  standardization=False, numer=1, G_mod=False, export_div=True)

    # Check if maximas has been computed or recompute them
    qg.count_local_extremum_of_Fiedler()
    # Resegment each apex with n = number of extremum and k-means ?
    for n in list_of_apex:
        number_clust = qg.nodes[n]['number_of_local_Fiedler_extremum']
        sub = create_subgraphs_to_work(quotientgraph=qg, list_quotient_node_to_work=[n])
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

            kmeans = KMeans(n_clusters=number_clust, init='k-means++', n_init=20, max_iter=300,
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
                np.savetxt('pcd_new_labels_apex' + str(n) + '_' + str(v) + '.txt', arr, delimiter=",")
                v += 1

    # exports


# Fonctions MIoU, etc etc


def compute_recall_precision_IoU(file_semantic_results="script/pcd_viterbi_classsemantic_final.txt",
                                 file_ground_truth_coord="script/cheno_virtuel_coordinates.txt",
                                 file_ground_truth_labels="script/cheno_virtuel_labels.txt",
                                 name_model='name_model',
                                 limb_gt=2, cotyledon_gt=3, main_stem_gt=1, petiole_gt=4,
                                 class_limb=1, class_mainstem=3, class_petiol=5, class_branch=6, class_apex=4):
    """
    Compute recall, precision, and IoU (Intersection over Union) metrics for evaluating the
    semantic segmentation model's performance. It involves comparing semantic segmentation
    results produced by the model with ground truth labels, adjusts the label mappings, and
    computes various performance metrics like recall, precision, IoU, overall accuracy,
    mean IoU, and F1-score. Additionally, confusion matrices and evaluation results
    are saved to output text files.

    Parameters
    ----------
    file_semantic_results : str, default="script/pcd_viterbi_classsemantic_final.txt"
        Path to the file containing semantic segmentation results, with x, y, z
        coordinates and predicted labels.
    file_ground_truth_coord : str, default="script/cheno_virtuel_coordinates.txt"
        Path to the file containing ground truth 3D coordinates x, y, and z.
    file_ground_truth_labels : str, default="script/cheno_virtuel_labels.txt"
        Path to the file containing ground truth labels corresponding to the coordinates.
    name_model : str, default='name_model'
        Base name for output files to store results.
    limb_gt : int, default=2
        Ground truth label value for the limb.
    cotyledon_gt : int, default=3
        Ground truth label value for the cotyledon.
    main_stem_gt : int, default=1
        Ground truth label value for the main stem.
    petiole_gt : int, default=4
        Ground truth label value for the petiole.
    class_limb : int, default=1
        Prediction label value for the limb.
    class_mainstem : int, default=3
        Prediction label value for the main stem.
    class_petiol : int, default=5
        Prediction label value for the petiole.
    class_branch : int, default=6
        Prediction label value for the branch.
    class_apex : int, default=4
        Prediction label value for the apex.

    Raises
    ------
    FileNotFoundError
        If any of the input file paths does not exist.

    ValueError
        If the data in the provided files is not in the expected format.

    Notes
    -----
    The procedure saves the following output files:
        - 'Ground_truth_virtual`name_model`.txt': Adjusted ground truth data.
        - 'Ground_truth_final`name_model`.txt': Final ground truth labels adjusted.
        - 'Label_final`name_model`.txt': Predicted labels adjusted.
        - '`name_model`scikit_cm': Confusion matrix.
        - '`name_model`eval.txt': Evaluation metrics (TP, FP, FN, Recall, Precision, IoU, etc.).
    - The function ensures that predicted labels are remapped to align with ground truth
      labels before metric computation.
    - Mean IoU is calculated as the average IoU across all unique ground truth labels.
    - Accuracy and F1-score are computed using standard metrics for binary/multi-class
      classification.
    - The confusion matrix is generated using the sklearn `confusion_matrix` function,
      and saved to a file.
    """
    xc, yc, zc, labelsc = np.loadtxt(fname=file_semantic_results, delimiter=',', unpack=True)
    xt, yt, zt = np.loadtxt(fname=file_ground_truth_coord, delimiter=',', unpack=True)
    labelst = np.loadtxt(fname=file_ground_truth_labels, delimiter=',', unpack=True)
    exp = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelst[:, np.newaxis]), axis=1)
    np.savetxt('Ground_truth_virtual' + name_model + '.txt', exp, delimiter=' ', fmt='%f')

    # creation d'une liste ground truth correspondant aux labels.
    label_gt_end = copy.deepcopy(labelst)
    for i in range(len(label_gt_end)):
        if label_gt_end[i] == limb_gt:
            new = class_limb
        elif label_gt_end[i] == main_stem_gt:
            new = class_mainstem
        elif label_gt_end[i] == petiole_gt:
            new = class_petiol
        # elif label_gt_end[i] == petiole_gt:
        #    new = class_mainstem
        elif label_gt_end[i] == cotyledon_gt:
            new = class_limb
        label_gt_end[i] = new

    gt_final = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], label_gt_end[:, np.newaxis]),
                              axis=1)
    np.savetxt('Ground_truth_final' + name_model + '.txt', gt_final, delimiter=' ', fmt='%f')
    # ici j'ai trop de différents labels par rapport à la vérité terrain, a enlever si c'est ok entre les deux

    for i in range(len(labelsc)):
        if labelsc[i] == class_branch:
            new = class_mainstem
            labelsc[i] = new
        elif labelsc[i] == class_apex:
            new = class_limb
            labelsc[i] = new
        # elif labelsc[i] == class_petiol:
        #    new = class_mainstem
        #    labelsc[i] = new

    label_final = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelsc[:, np.newaxis]),
                                 axis=1)
    np.savetxt('Label_final' + name_model + '.txt', label_final, delimiter=' ', fmt='%f')

    mres = np.zeros((len(set(label_gt_end)) + 4, 8))
    # mres[0, 1] = 'TP'
    # mres[0, 2] = 'FN'
    # mres[0, 3] = 'FP'
    # mres[0, 4] = 'Re'
    # mres[0, 5] = 'Pr'
    # mres[0, 6] = 'IoU'
    # faire une vérification que les coordoonnées correspondent ?
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
        Re[i] = (TP[i]) / (TP[i] + FN[i])
        Pr[i] = (TP[i]) / (TP[i] + FP[i])
        IoU[i] = (TP[i]) / (TP[i] + FN[i] + FP[i])
        TPtot += TP[i]
        FNtot += FN[i]
        FPtot += FP[i]
        TNtot += TN[i]
        MIoU += IoU[i]

    MIoU /= len(set(label_gt_end))
    totalacc = (TPtot + TNtot) / (TPtot + TNtot + FPtot + FNtot)
    f1_score = TPtot / (TPtot + 0.5 * (FNtot + FPtot))

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
    np.savetxt(name_model + 'scikit_cm', cm, fmt='%.4e')
    np.savetxt(name_model + 'eval.txt', mres, fmt='%.4e')


def change_labels(file_semantic_results="script/pcd_viterbi_classsemantic_final.txt",
                  name_model='name_model',
                  class_limb=1, class_mainstem=3, class_petiol=5, class_branch=6, class_apex=4):
    """
    Modifies the semantic labels of a 3D point cloud dataset based on predefined class mappings and saves the updated
    labels to a new file.

    This function loads a set of 3D coordinates along with their semantic class labels from a text file. It updates the
    labels based on a mapping defined by the parameters and writes the modified labels into a new output file corresponding
    to the provided model name.

    Parameters
    ----------
    file_semantic_results : str
        Path to the input file containing the 3D point cloud coordinates and semantic labels. The file should have
        comma-separated values with four columns: x-coordinates, y-coordinates, z-coordinates, and semantic labels.
    name_model : str
        A string used for naming the output file containing the modified labels.
    class_limb : int
        The numeric class label representing "limb" in the dataset.
    class_mainstem : int
        The numeric class label representing "mainstem" in the dataset.
    class_petiol : int
        The numeric class label representing "petiol" in the dataset.
    class_branch : int
        The numeric class label representing "branch" in the dataset.
    class_apex : int
        The numeric class label representing "apex" in the dataset.
    """
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

    label_final = np.concatenate((xc[:, np.newaxis], yc[:, np.newaxis], zc[:, np.newaxis], labelsc[:, np.newaxis]),
                                 axis=1)
    np.savetxt('Label_final' + name_model + '.txt', label_final, delimiter=' ', fmt='%f')


def downsample_pcd(file_pcd="script/pcd_viterbi_classsemantic_final.txt",
                   name_model='name_model'):
    """
    Reduces the resolution of a point cloud by voxel down-sampling and saves the
    resulting down-sampled point cloud to a file.

    This function uses voxel grid down-sampling to reduce the density of points
    in the given point cloud file. Additionally, it traces approximate classes to
    maintain semantic information during the down-sampling process. The resulting
    down-sampled point cloud is saved in PLY format with a specified output file name.

    Parameters
    ----------
    file_pcd : str, optional
        The file path to the input point cloud (in PLY format). Default is
        "script/pcd_viterbi_classsemantic_final.txt".
    name_model : str, optional
        The base name for the output down-sampled point cloud file. The output
        file will have the name "<name_model>down_sample.ply".
    """
    pcd = o3d.io.read_point_cloud(file_pcd, format='ply')
    downpcd = o3d.geometry.voxel_down_sample_and_trace(input=pcd, voxel_size=1.0, approximate_class=True)
    o3d.io.write_point_cloud(name_model + "down_sample.ply", downpcd)


def compute_recall_precision_IoU_real_plants(file_semantic_results="script/pcd_viterbi_classsemantic_final.txt",
                                             file_instance_results="script/pcd_viterbi_classsemantic_final.txt",
                                             file_ground_truth="script/cheno_virtuel_coordinates.txt",
                                             name_model="name",
                                             class_limb=1, class_mainstem=3, class_petiol=5, class_branch=6,
                                             class_apex=4):
    """
    Computes various evaluation metrics such as recall, precision, IoU (Intersection over Union),
    and other classification, clustering, and region-based measures, for a 3D plant dataset that is
    segmented using semantic and instance labels compared against ground truth data.

    This function takes file paths of input data (semantic results, instance results, and ground truth),
    model names, and specific classification category IDs, and performs the following tasks:
    - Reads the input files and processes them into semantic and instance labels.
    - Adjusts and relabels certain class labels based on predefined thresholds.
    - Computes various metrics including TP, FP, FN, TN, precision, recall, IoU, and adjusts confusion matrices.
    - Outputs standardized data formats for the ground truth, predicted labels, and summary evaluation
      metrics to files.

    Parameters
    ----------
    file_semantic_results : str, optional
        Path to the input file containing semantic classification results.
        Defaults to "script/pcd_viterbi_classsemantic_final.txt".
    file_instance_results : str, optional
        Path to the input file containing instance classification results.
        Defaults to "script/pcd_viterbi_classsemantic_final.txt".
    file_ground_truth : str, optional
        Path to the input file containing ground truth coordinate data.
        Defaults to "script/cheno_virtuel_coordinates.txt".
    name_model : str, optional
        Name of the model being processed and evaluated. Outputs will include this name in their filenames.
        Defaults to "name".
    class_limb : int, optional
        ID value representing the limb class (used for relabeling). Defaults to 1.
    class_mainstem : int, optional
        ID value representing the main stem class (used for relabeling). Defaults to 3.
    class_petiol : int, optional
        ID value representing the petiol class (used for relabeling). Defaults to 5.
    class_branch : int, optional
        ID value representing the branch class (used for relabeling). Defaults to 6.
    class_apex : int, optional
        ID value representing the apex class (used for relabeling). Defaults to 4.

    Returns
    -------
    None

    Notes
    -----
    - The function outputs several files containing processed data and results:
      - Ground truth in standardized format with predicted classifications relabeled.
      - Metrics related to clustering and classification performance, including precision, recall,
        IoU, confusion matrix, and an F1 score.
      - Summary results are written to evaluation files using the provided `name_model`.

    - All thresholds for relabeling are predefined and hardcoded within the function for specific
      plant segmentation studies. Adjust these thresholds carefully if applying the function to
      datasets with different formats or labeling conventions.

    - The function leverages scikit-learn's metrics module for the computation of clustering and
      classification scores.

    - Outputs are saved in plaintext or numerical formats (e.g., `.txt`), making them suitable for
      importing into statistical or visualization tools for downstream analysis.
    """
    xc, yc, zc, labelsc = np.loadtxt(fname=file_semantic_results, delimiter=',', unpack=True)
    xc, yc, zc, labelinstance = np.loadtxt(fname=file_instance_results, delimiter=',', unpack=True)
    xt, yt, zt, labelst, np2, c1, c2, c3 = np.loadtxt(fname=file_ground_truth, delimiter=',', unpack=True)
    exp = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelst[:, np.newaxis]), axis=1)
    np.savetxt('Ground_truth_' + name_model + '.txt', exp, delimiter=' ', fmt='%f')

    rand = metrics.cluster.rand_score(labelst, labelinstance)
    print("rand")
    print(rand)
    rand_adj = metrics.cluster.adjusted_rand_score(labelst, labelinstance)
    print("rand_adjusted")
    print(rand_adj)
    mutual = metrics.adjusted_mutual_info_score(labelst, labelinstance)
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

    # creation d'une liste ground truth correspondant aux labels.
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

    gt_final = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], label_gt_end[:, np.newaxis]),
                              axis=1)
    np.savetxt('Ground_truth_final' + name_model + '.txt', gt_final, delimiter=' ', fmt='%f')
    # ici j'ai trop de différents labels par rapport à la vérité terrain, a enlever si c'est ok entre les deux
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
    label_final = np.concatenate((xt[:, np.newaxis], yt[:, np.newaxis], zt[:, np.newaxis], labelsc[:, np.newaxis]),
                                 axis=1)
    np.savetxt('Label_final' + name_model + '.txt', label_final, delimiter=' ', fmt='%f')

    mres = np.zeros(((len(list)) + 4, 8))
    # mres[0, 1] = 'TP'
    # mres[0, 2] = 'FN'
    # mres[0, 3] = 'FP'
    # mres[0, 4] = 'Re'
    # mres[0, 5] = 'Pr'
    # mres[0, 6] = 'IoU'
    # faire une vérification que les coordoonnées correspondent ?
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
        Re[i] = (TP[i]) / (TP[i] + FN[i])
        Pr[i] = (TP[i]) / (TP[i] + FP[i])
        IoU[i] = (TP[i]) / (TP[i] + FN[i] + FP[i])
        TPtot += TP[i]
        FNtot += FN[i]
        FPtot += FP[i]
        TNtot += TN[i]
        MIoU += IoU[i]

    MIoU /= len(set(label_gt_end))
    totalacc = (TPtot + TNtot) / (TPtot + TNtot + FPtot + FNtot)
    f1_score = TPtot / (TPtot + 0.5 * (FNtot + FPtot))

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
    np.savetxt(name_model + 'scikit_cm', cm, fmt='%.4e')
    np.savetxt(name_model + 'eval.txt', mres, fmt='%.4e')
