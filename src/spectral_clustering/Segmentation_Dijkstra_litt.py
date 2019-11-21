# Méthode de segmentation telle qu'expliquée dans l'article
# Hétroy-Wheeler, Casella, and Boltcheva, “Segmentation of Tree Seedling Point Clouds into Elementary Units.”

##################### IMPORTS

import open3d as open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import spectral_clustering.similarity_graph as SGk
import operator # permet d'obtenir la clé dans un dictionnaire

from random import choice

##################### FONCTIONS


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

def sortienuagelitt(pcd, G, segmentdict, c):
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
    G = SGk.gen_graph(pcd, method='knn', nearest_neighbors=r)
    #SGk.drawGraphO3D(pcd, G)
    G = actugraph(G)
    segmsource = initdijkstralitt(G)
    c = 13
    segmentdict = segmdijkstralitt(G, segmsource, c)
    affichesegmlitt(pcd, G, segmentdict, c)
    sortienuagelitt(pcd, G, segmentdict)