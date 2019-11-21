# Méthode de segmentation telle qu'expliquée dans l'article
# Hétroy-Wheeler, Casella, and Boltcheva, “Segmentation of Tree Seedling Point Clouds into Elementary Units.”

##################### IMPORTS
import time

import open3d as open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import SimilarityGraph as SGk
import operator # permet d'obtenir la clé dans un dictionnaire

from random import choice


##################### FONCTIONS

def actugraph(G):
    Lcsr = nx.laplacian_matrix(G, weight='weight')
    Lcsr = spsp.csr_matrix.asfptype(Lcsr)
    # k nombre de vecteurs propres que l'on veut calculer
    k = 20
    keigenval, keigenvec = spsp.linalg.eigsh(Lcsr,k=k,sigma=0, which='LM')
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
        for j in range(1,k):
            Somme = Somme + (pow(keigenvec[pt1, j] - keigenvec[pt2, j], 2)/eigenval[j])
        CommuteDist = np.sqrt(Somme)
        G[pt1][pt2]['weight'] = CommuteDist
    return G

def initdijkstralitt(G):
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
    return segmsource

# c nombre de clusters voulus
def segmdijkstralitt(G, segmsource, c):
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
            try :
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

def affichesegmlitt(pcd, G, segmentdict, c):
    Gaffichage = nx.Graph()
    pts = np.array(pcd.points)
    N = len(pcd.points)
    for i in range(N): Gaffichage.add_node(i, pos = pts[i])
    for i in range(1, c):
        Gaffichage.add_path(segmentdict[i])
    edgelist = Gaffichage.edges
    print(Gaffichage.edges)
    cloption = open3d.visualization.RenderOption()
    graph = open3d.geometry.LineSet()
    graph.points = open3d.Vector3dVector(pts)
    graph.lines = open3d.Vector2iVector(edgelist)
    open3d.draw_geometries([graph, pcd])

def sortienuagelitt(pcd, G, segmentdict):
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
            if length[p]==0:
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
    label = np.asarray(label.reshape(np.asarray(pcd.points).shape[0], 1), dtype= np.float64)
    pcdtabclassif = np.concatenate([np.asarray(pcd.points), label], axis = 1)
    np.savetxt('pcdclassifdijkstra2.txt', pcdtabclassif, delimiter = ",")


##################### MAIN
if __name__ == '__main__':
    pcd = open3d.read_point_cloud("Data/arabi_densep_clean_segm.ply")
    r = 8
    G = SGk.genGraph(pcd, r)
    #SGk.drawGraphO3D(pcd, G)
    G = actugraph(G)
    segmsource = initdijkstralitt(G)
    c = 13
    segmentdict = segmdijkstralitt(G, segmsource, c)
    affichesegmlitt(pcd, G, segmentdict, c)
    sortienuagelitt(pcd, G, segmentdict)