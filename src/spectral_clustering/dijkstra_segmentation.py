#!/usr/bin/env python
# -*- coding: utf-8 -*-

import operator
from random import choice

import networkx as nx
import numpy as np
import open3d as o3d
import scipy.sparse as spsp
import spectral_clustering.similarity_graph as SGk


# Méthode de segmentation telle qu'expliquée dans l'article
# Hétroy-Wheeler, Casella, and Boltcheva, “Segmentation of Tree Seedling Point Clouds into Elementary Units.”
########################## FONCTIONS

def actugraph(G):
    """
    Calculates the weight of graph edges based on the commute-time distance using eigenvalues
    and eigenvectors of the graph Laplacian matrix.

    This function computes the Laplacian matrix of the graph, calculates its eigenvalues
    and eigenvectors, and updates the weights of the edges in the graph using the
    commute-time distance. The commute-time distance is derived from the eigenvectors
    and eigenvalues of the Laplacian matrix.

    Parameters
    ----------
    G : networkx.Graph
        The input graph on which computations will be performed.
        The graph can have weighted or unweighted edges, and it will be updated
        with new weights based on the computed commute-time distances.

    Returns
    -------
    networkx.Graph
        A modified version of the input graph with updated edge weights. Each edge
        weight represents the commute-time distance between its endpoints.

    Raises
    ------
    TypeError
        If the input `G` is not a `networkx.Graph`.
    """
    Lcsr = nx.laplacian_matrix(G, weight='weight')
    Lcsr = spsp.csr_matrix.asfptype(Lcsr)
    # k nombre de vecteurs propres que l'on veut calculer
    k = 20
    keigenval, keigenvec = spsp.linalg.eigsh(Lcsr, k=k, sigma=0, which='LM')
    eigenval = keigenval.reshape(keigenval.shape[0], 1)
    # Actualisation du graphe pour obtenir les poids en fonction de la commute-time distance
    arcs = G.edges
    arcs = iter(arcs)
    arcs = tuple(arcs)
    nbe_arcs = G.number_of_edges()
    for t in range(nbe_arcs):
        pt1 = arcs[t][0]
        pt2 = arcs[t][1]
        Somme = 0
        for j in range(1, k):
            Somme = Somme + (pow(keigenvec[pt1, j] - keigenvec[pt2, j], 2) / eigenval[j])
        CommuteDist = np.sqrt(Somme)
        G[pt1][pt2]['weight'] = CommuteDist
    return G


def initdijkstralitt(G):
    """Initializes the first segment in a graph using Dijkstra's algorithm.

    This function selects a random node from the graph and computes a segment
    (based on shortest paths and their lengths) between two nodes. Specifically,
    it identifies the two most distant nodes from each other in terms of shortest
    path weights and determines a direct path connecting them.

    Parameters
    ----------
    G : networkx.Graph
        A graph represented as a NetworkX graph instance. The graph's edges
        must have a "weight" attribute to compute shortest path weights.

    Returns
    -------
    list
        A list of nodes representing the shortest path (based on edge weights)
        between two most distant nodes in the graph.

    """
    # initialisation du premier segment
    # Choix d'un point aléatoire dans le graphe
    random_node = choice(list(G.nodes))
    # Stockage de tous les poids des plus courts chemins vers tous les autres points du graphe
    dict = nx.single_source_dijkstra_path_length(G, random_node, weight='weight')
    # Isolation du point le plus éloigné au point choisi aléatoirement
    # Sélection du point associé au plus long chemin des plus courts chemins
    ptsource = max(dict.items(), key=operator.itemgetter(1))[0]
    # Stockage de tous les poids des plus courts chemin entre le précédent point et l'ensemble des points du graphe
    dict = nx.single_source_dijkstra_path_length(G, ptsource, weight='weight')
    # Isolation du point le plus éloigné
    ptarrivee = max(dict.items(), key=operator.itemgetter(1))[0]
    # Obtention du chemin entre le point source et le point d'arrivée finaux
    segmsource = nx.dijkstra_path(G, ptsource, ptarrivee, weight='weight')
    return segmsource


# c nombre de clusters voulus
def segmdijkstralitt(G, segmsource, c):
    """Compute segmented paths within a graph using Dijkstra's shortest path algorithm.

    This function iteratively calculates segmented paths within the given graph.
    At each step, it identifies a new segment based on the longest weighted path
    from a multi-source Dijkstra's search on the graph, starting from the combined
    set of nodes in already identified segments. The function also updates and
    reorganizes existing segments to account for the newly identified segment.

    Parameters
    ----------
    G : networkx.Graph
        The graph on which the segmented Dijkstra algorithm will be applied.
        The graph must have a 'weight' attribute defined for its edges.
    segmsource : list
        A list of nodes representing the source segment(s) within the graph.
        Initial segments for the segmentation process.
    c : int
        The number of iterations to perform, which defines the level of segmentation.

    Returns
    -------
    dict
        A dictionary where keys are segment ids (integers starting from 1) and
        values are lists of nodes representing the segments identified at each
        iteration.
    """
    # Itérateur
    i = 1
    # initialisation dictionnaire de segments/chemins
    segmentdict = {}
    segmentdict[i] = segmsource
    # Début boucle permettant de décrire l'ensemble du graphe via des chemins sur ce dernier
    while i < c:
        chemintot = []
        # Concaténation des chemins pour obtenir une seule liste utilisable par les fonctions nx dijkstra Segment(1..i)
        for Seg, chemin in segmentdict.items():
            chemintot = chemintot + chemin
        # Transformation de la liste obtenue en set
        setsegments = set(chemintot)
        # Calcul des chemins les plus courts vers les autres nodes du graphe
        dict = nx.multi_source_dijkstra_path_length(G, setsegments, weight='weight')
        # Sélection du point d'arrivee qui nécessite de prendre le chemin le plus lourd en pondération
        ptarrivee = max(dict.items(), key=operator.itemgetter(1))[0]
        # Obtention du chemin le plus long. Prise du point source, stockage.
        length, path = nx.multi_source_dijkstra(G, setsegments, ptarrivee)
        segmentdict[i + 1] = path
        p = path[0]
        # retrouver le segment auquel p appartient
        j = 1
        ind = -1
        while ind == -1 and j <= i:
            try:
                ind = segmentdict[j].index(p)
            except ValueError:
                ind = -1
            j = j + 1
        j = j - 1
        # Enlève tous les points successifs du Segment[j] de p à l'un des bouts et les ajoute au segment[i+2]
        indexp = segmentdict[j].index(p)
        segmentdict[i + 2] = segmentdict[j][:indexp]
        segmentdict[j] = segmentdict[j][indexp:]
        # Fin boucle, incrémentation i
        i = i + 2
    return segmentdict


# Segmentation via Dijkstra uniquement/ pas d'apport spectral

def initdijkstra(G):
    """Initializes the first segment of the graph using Dijkstra's algorithm.

    This function randomly selects a starting node from the graph, computes the shortest path
    distances from the selected node to all other nodes, and determines the farthest endpoint
    based on these distances. It then repeats the process starting from the farthest endpoint
    to determine another distant endpoint. Finally, it computes the shortest path between the
    two farthest endpoints and returns the path along with the source and destination nodes.

    Parameters
    ----------
    G : networkx.Graph
        A graph representation, where nodes are connected by weighted edges.

    Returns
    -------
    list
        The shortest path (as a sequence of nodes) between the final source and destination
        nodes with maximum distance calculated using Dijkstra's algorithm.
    int or str
        The node corresponding to the starting point (farthest from the randomly chosen node initially).
    int or str
        The node corresponding to the endpoint (farthest from the source node).
    """
    # initialisation du premier segment
    # Choix d'un point aléatoire dans le graphe
    random_node = choice(list(G.nodes))
    # Stockage de tous les poids des plus courts chemins vers tous les autres points du graphe
    dict = {}
    dict = nx.single_source_dijkstra_path_length(G, random_node, weight='weight')
    # Isolation du point le plus éloigné au point choisi aléatoirement
    # Sélection du point associé au plus long chemin des plus courts chemins
    ptsource = max(dict.items(), key=operator.itemgetter(1))[0]
    # Stockage de tous les poids des plus courts chemin entre le précédent point et l'ensemble des points du graphe
    dict = nx.single_source_dijkstra_path_length(G, ptsource, weight='weight')
    # Isolation du point le plus éloigné
    ptarrivee = max(dict.items(), key=operator.itemgetter(1))[0]
    # Obtention du chemin entre le point source et le point d'arrivée finaux
    segmsource = nx.dijkstra_path(G, ptsource, ptarrivee, weight='weight')
    return segmsource, ptsource, ptarrivee


# Densification du nombre de chemins définissant la tige.
# Ici, on densifie sur le même point d'arrivée et le même point source
# On pourrait envisager de l'effectuer sur des points extrêmes différents. Gain ?
# pathsupp est le nombre de chemins que l'on ajoute, il est lié au nombre r de k plus proches voisins crées pour le graphe
# On pourrait mettre un nombre en lien avec r, le nombre de plus proches voisins pour automatiser.
def densiftige(G, segmsource, ptsource, ptarrivee, pathsupp=5):
    """Densifies a segment of the given graph by adding alternate shorter paths between the source and target nodes.

    This method focuses on minimizing the error of misclassifying points belonging to the segment as those in branches,
    potentially reducing inconsistencies in the graph representation.

    Parameters
    ----------
    G : networkx.Graph
        The input graph representing the network structure.
    segmsource : list
        The main segment to be densified, represented as a list of nodes forming
        the segment in the graph.
    ptsource : any
        The starting node for the segment in the graph.
    ptarrivee : any
        The target node for the segment in the graph.
    pathsupp : int, optional
        The number of additional shorter paths to compute and densify the segment
        with, by default 5.

    Returns
    -------
    list
        The updated segment with added shorter paths forming a densified structure.
    """
    # L'objectif ici est de densifier le segment 1, de la tige en ajoutant d'autres plus courts chemins dans cette tige.
    # L'erreur du nombre de points considérés en branches alors qu'ils sont dans la tige devrait être diminuée.
    # Il faudra peut-être considérer le même processus pour les branches, à voir.
    i = 1
    Gdel = G.__class__()
    Gdel.add_nodes_from(G)
    Gdel.add_edges_from(G.edges)

    Gdel.remove_nodes_from(segmsource[1:len(segmsource) - 1])
    while i < pathsupp:
        segmsupp = nx.dijkstra_path(Gdel, ptsource, ptarrivee, weight='weight')
        segmsource = segmsource + segmsupp[1:len(segmsupp) - 1]
        print(segmsupp)
        print(ptarrivee)
        print(ptsource)
        Gdel.remove_nodes_from(segmsupp[1:len(segmsupp) - 1])
        i = i + 1
        print(i)
    return segmsource


# Itérations pour obtenir tous les segments, dans chacune des branches.
# Prop définit la distance à la tige à partir de laquelle on arrête de compter les branches.
# En retour : dictionnaire contenant tous les segments.
def segmdijkstra(G, segmsource, prop=0.25):
    """
    Calculates the segmentation of a network graph using an iterative approach
    based on Dijkstra's algorithm and evaluates the connectivity of graph nodes.
    The function returns the dictionary of path segments and the count of segments
    before the stopping condition is met.

    Parameters
    ----------
    G : networkx.Graph
        The input graph on which the segmentation is performed. The graph should
        be weighted, with weights specified via the 'weight' attribute on edges.
    segmsource : list
        A list specifying the source nodes from which the segmentation paths
        are initialized.
    prop : float, optional
        A propagation threshold value used to determine the stopping condition
        of the segmentation process. The default value is 0.25.

    Returns
    -------
    dict
        A dictionary where keys are the segment numbers, and values are lists of nodes that
        form the paths for each segment.
    int
        The total number of segments that were created before the stopping condition was satisfied.
    """
    # Itérateur
    i = 1
    longueur = True
    # initialisation dictionnaire de segments/chemins
    segmentdict = {}
    segmentdict[i] = segmsource
    # Début boucle permettant de décrire l'ensemble du graphe via des chemins sur ce dernier
    while longueur:
        chemintot = []
        # Concaténation des chemins pour obtenir une seule liste utilisable par les fonctions nx dijkstra Segment(1..i)
        for Seg, chemin in segmentdict.items():
            chemintot = chemintot + chemin
        # Transformation de la liste obtenue en set
        setsegments = set(chemintot)
        # Calcul des chemins les plus courts vers les autres nodes du graphe
        dict = nx.multi_source_dijkstra_path_length(G, setsegments, weight='weight')
        # Sélection du point d'arrivee qui nécessite de prendre le chemin le plus lourd en pondération
        ptarrivee = max(dict.items(), key=operator.itemgetter(1))[0]
        # Obtention du chemin le plus long. Prise du point source, stockage.
        length, path = nx.multi_source_dijkstra(G, setsegments, ptarrivee)
        print(length)
        if i == 1:
            ref = length
        if length < prop * ref:
            longueur = False
            c = i
        else:
            segmentdict[i + 1] = path
            p = path[0]
            # Fin boucle, incrémentation i
            i = i + 1
    return segmentdict, c


# c nombre de clusters obtenus
def affichesegm(pcd, segmentdict, c):
    """Visualizes intermediate results, specifically line segments.

    This function builds a graph representation of the segments and visualizes the graph
    along with the original point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The input point cloud to which the segments belong.
    segmentdict : dict
        A dictionary where keys are segment indices and values are lists of
        indices representing the line segments in the graph.
    c : int
        The number of distinct segments in the graph.

    Notes
    -----
    This function uses Open3D to render point clouds and geometry and uses
    NetworkX to build the graph based on the segment information. Line segments
    are visualized along with the point cloud based on the provided input.
    """
    # affichage du résultat intermédiaire, c'est-à-dire des segments.
    Gaffichage = nx.Graph()
    pts = np.array(pcd.points)
    N = len(pcd.points)
    for i in range(N):
        Gaffichage.add_node(i, pos=pts[i])
    for i in range(1, c):
        nx.add_path(Gaffichage, segmentdict[i])
    edgelist = Gaffichage.edges
    print(Gaffichage.edges)
    cloption = o3d.visualization.RenderOption()
    graph = o3d.geometry.LineSet()
    graph.points = o3d.utility.Vector3dVector(pts)
    graph.lines = o3d.utility.Vector2iVector(edgelist)
    o3d.visualization.draw_geometries([graph, pcd])


def sortienuagesegm(pcd, G, segmentdict, c):
    """
    Sorts points in a point cloud based on their shortest paths in a graph,
    assigns labels to them, and saves the results to a file.

    The function assigns a classification label to each point in a given
    point cloud based on the shortest path distances computed from a set
    of initial segments within a graph. Labels are determined by the segments
    the points are closest to. If a point cannot be classified, it is assigned
    a default label `c`. The resulting labeled point cloud is saved as a text
    file.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The input point cloud object, which contains the 3D points to be classified.
    G : networkx.Graph
        A graph where the shortest paths are computed. Nodes represent points, and
        edges have weights indicating distances.
    segmentdict : dict
        A dictionary representing segments. Keys are integers, and values
        are lists of integers representing node indices in the graph, which
        define segments considered for classification.
    c : int
        The default class label assigned to points that cannot be associated
        with any segment.

    Notes
    -----
    The function modifies the point labels by computing the shortest path from a set
    of segments in the graph to every point in the point cloud. For points that belong
    to more than one segment, it labels them based on their closest segment sequence
    in the search process.

    The labeled point cloud is saved as a comma-separated text file named
    'pcdclassifdijkstra3.txt'. The result includes the 3D coordinates of points
    and their corresponding labels.

    This function assumes that no duplicate edges with differing weights exist
    in the graph and that the provided point cloud is valid with a consistent
    structure.
    """
    label = []
    chemintot = []
    N = len(pcd.points)
    for Seg, chemin in segmentdict.items():
        chemintot = chemintot + chemin
    # Transformation de la liste obtenue en set
    setsegments = set(chemintot)
    length, path = nx.multi_source_dijkstra(G, setsegments, weight='weight')
    for p in range(N):
        if p in length:
            # Sélection du chemin qui concerne le point d'intérêt
            j = 1
            ind = -1
            if length[p] == 0:
                parrive = p
            else:
                parrive = path[p][0]
            while ind == -1 and j < c:
                try:
                    ind = segmentdict[j].index(parrive)
                except ValueError:
                    ind = -1
                j = j + 1
            j = j - 1
            label = label + [j]
        else:
            label = label + [c]

    label = np.asarray(label)
    label = np.asarray(label.reshape(np.asarray(pcd.points).shape[0], 1), dtype=np.float64)
    pcdtabclassif = np.concatenate([np.asarray(pcd.points), label], axis=1)
    np.savetxt('pcdclassifdijkstra3.txt', pcdtabclassif, delimiter=",")


############################### Corps

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("Data/impr3D_pcd.ply")
    r = 8
    G = SGk.create_riemannian_graph(pcd, method='knn', nearest_neighbors=r)
    SGk.draw_graph_open3d(pcd, G)

    method = 'k'

    if method == 'litt':
        G = actugraph(G)
        segmsource = initdijkstralitt(G)
        c = 13
        segmdict = segmdijkstralitt(G, segmsource, c)
    else:
        segmsource, ptsource, ptarrivee = initdijkstra(G)
        segmsource = densiftige(G, segmsource, ptsource, ptarrivee)
        segmdict, c = segmdijkstra(G, segmsource)
    affichesegm(pcd, segmdict, c)
    sortienuagesegm(pcd, G, segmdict, c)
