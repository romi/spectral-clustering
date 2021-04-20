########################## IMPORTS

import open3d as open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import spectral_clustering.similarity_graph as SGk
import operator  # permet d'obtenir la clé dans un dictionnaire

from random import choice

# Méthode de segmentation telle qu'expliquée dans l'article
# Hétroy-Wheeler, Casella, and Boltcheva, “Segmentation of Tree Seedling Point Clouds into Elementary Units.”
########################## FONCTIONS

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

# Segmentation via Dijkstra uniquement/ pas d'apport spectral

def initdijkstra(G):
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
def densiftige(G, segmsource, ptsource, ptarrivee, pathsupp = 5):
    # L'objectif ici est de densifier le segment 1, de la tige en ajoutant d'autres plus courts chemins dans cette tige.
    # L'erreur du nombre de points considérés en branches alors qu'ils sont dans la tige devrait être diminuée.
    # Il faudra peut-être considérer le même processus pour les branches, à voir.
    i = 1
    Gdel = G.__class__()
    Gdel.add_nodes_from(G)
    Gdel.add_edges_from(G.edges)

    Gdel.remove_nodes_from(segmsource[1:len(segmsource)-1])
    while i < pathsupp:
        segmsupp = nx.dijkstra_path(Gdel, ptsource, ptarrivee, weight='weight')
        segmsource = segmsource + segmsupp[1:len(segmsupp)-1]
        print(segmsupp)
        print(ptarrivee)
        print(ptsource)
        Gdel.remove_nodes_from(segmsupp[1:len(segmsupp)-1])
        i = i + 1
        print(i)
    return segmsource

# Itérations pour obtenir tous les segments, dans chacune des branches.
# Prop définit la distance à la tige à partir de laquelle on arrête de compter les branches.
# En retour : dictionnaire contenant tous les segments.
def segmdijkstra(G, segmsource, prop = 0.25):
    # Itérateur
    i = 1
    longueur = True
    # initialisation dictionnaire de segments/chemins
    segmentdict = {}
    segmentdict[i] = segmsource
    # Début boucle permettant de décrire l'ensemble du graphe via des chemins sur ce dernier
    while longueur == True:
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
        if length < prop*ref:
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
    # affichage du résultat intermédiaire, c'est-à-dire des segments.
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

def sortienuagesegm(pcd, G, segmentdict, c):
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
    np.savetxt('pcdclassifdijkstra3.txt', pcdtabclassif, delimiter = ",")

############################### Corps

if __name__ == '__main__':
    pcd = open3d.read_point_cloud("../../Data/impr3D_pcd.ply")
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