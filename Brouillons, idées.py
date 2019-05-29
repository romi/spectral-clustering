

# l'idée était ici de viusaliser autrement les vecteurs propres en réalisant des lignes les représentant.
# Les chemins sont stockés dans des graphes.
# Ce code est à conserver car permet éventuellement de
# def VisuEigenVectors(pcd, keigenvec, k):
from itertools import combinations
# élimination des vecteurs nuls
for j in range(keigenvec.shape[1]):
    if np.all( keigenvec[:,j] == 0):
        u = j
keigenvec = keigenvec[:,u:]
print(keigenvec)
# Création du dictionnaire de graphes
Gdic = {}
Gpath = {}
Distance = 0
for j in range(k):
    # Initialisation du Graphe correspondant au vecteur propre j
    Gdic[j] = nx.Graph()
    # Stockage de tous les points sur lesquels le vecteur propre a une composante en tant que node de Graphe
    # Le poids est stocké aussi.
    for i in range(keigenvec.shape[0]):
        if keigenvec[i,j] != 0.0:
            Gdic[j].add_node(i, weight = keigenvec[i, j])
    edges = combinations(Gdic[j].nodes, 2)
    for i in list(edges):
        Gdic[j].add_edge(i[0], i[1])
    # Ajout des poids sur tous les edges grâce aux coordonnées dans pcd
    arcs = Gdic[j]
    arcs = iter(arcs)
    arcs = tuple(arcs)
    nbe_arcs = Gdic[j].number_of_edges()
    for t in range(nbe_arcs):
        pt1 = arcs[t][0]
        pt2 = arcs[t][1]
        Distancenouv = np.sqrt(np.square(pcd[pt1][0] - pcd[pt2][0]) + np.square(pcd[pt1][1] - pcd[pt2][1]) + np.square(
            pcd[pt1][2] - pcd[pt2][2]))
        G[pt1][pt2]['weight'] = Distancenouv
            # Je cherche les deux points les plus éloignés l'un de l'autre.
            # Pour obtenir les deux extrémités du vecteur propre.
            # Pour par la suite calculer un plus court chemin entre ces deux points
            # et penser à supprimer l'edge qui les relie.
            #PlusGrandeDistance = max(Distancenouv, Distance)
            #if PlusGrandeDistance == Distancenouv :
                #CoupleEloigne = (pt1, pt2)
            #Distance = Distancenouv
        #Gdic[j].remove_edge(CoupleEloigne[0], CoupleEloigne[1])
        #Gpath[j] = dijkstra_path(Gdic[j], CoupleEloigne[0], CoupleEloigne[1], weight = 'weight')
    mst = nx.minimum_spanning_edges(Gdic[j], weight= 'weight', data= False)
    edgelist = list(mst)
# return edgelist

edgelist = VisuEigenVectors(pcd, keigenvec, 2)
graph = open3d.LineSet()
graph.points = pcd.points
graph.lines = open3d.Vector2iVector(edgelist)
open3d.draw_geometries([graph])

# Création graphe dans l'espace spectral

Lignes = keigenvec.shape[0]

Col = keigenvec.shape[1]

# Prise des points sous forme de tableau ndarray
keigenvec = np.array(keigenvec)