import open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
# from mayavi import mlab
import scipy.cluster.vq as vq
# from sklearn.cluster import DBSCAN

# p est le nuage de points pcd, r_nn le seuil pour la détermination des connections
# fonction permettant la création d'un graphe grâce à la librairie networkx
def genGraph(p, r_nn):
    N = len(p.points)
    # défintion d'un arbre KD contenant tous les points
    tree = open3d.KDTreeFlann(p)
    # Prise des points sous forme de tableau ndarray
    pts = np.array(p.points)

    # Déclaration d'un graph networkx
    G = nx.Graph()
    # On insère chaque point du nuage de points dans le graphe avec un numéro et le trio de coordonnées (pos) en attributs
    for i in range(N): G.add_node(i, pos = pts[i])

    # Construction des edges du graphe à partir d'un seuil
    # On part de la structure de nuage de points en KDTree
    # Cette structure dans open3d dispose de fonctions pour seuil, KNN, RKNN
    for i in range(N):
        #[k, idxs, _] = tree.search_radius_vector_3d(pts[i], r_nn)
        [k, idxs, _] = tree.search_knn_vector_3d(pts[i], r_nn)
        for idx in idxs:
            d = np.sqrt(np.square(pts[i][0] - pts[idx][0]) + np.square(pts[i][1] - pts[idx][1]) + np.square(
                pts[i][2] - pts[idx][2]))
            if d != 0:
                w = 1 / d
                G.add_edge(i, idx, weight=w)


    return G

# affichage via open3D
# En entrée : p nuage de points
def drawGraphO3D(p, G):
    graph = open3d.LineSet()
    graph.points = p.points
    graph.lines = open3d.Vector2iVector(G.edges)
    open3d.draw_geometries([graph])

# affichage du graphe via CellComplex
# en entrée : la matrice d'adjacence (matrice de similarité) et le nuage de points importé/lu via open3D
def drawGraphCC(pcd, simatrix):
    pcdtab = np.asarray(pcd.points)
    s, t = np.meshgrid(np.arange(len(pcdtab)), np.arange(len(pcdtab)))
    sources = s[simatrix > 0]
    targets = t[simatrix > 0]
    sources, targets = sources[sources < targets], targets[sources < targets]

    topomesh = edge_topomesh(np.transpose([sources, targets]), dict(zip(np.arange(len(pcdtab)), pcdtab)))

    from vplants.cellcomplex.property_topomesh.property_topomesh_visualization.vtk_actor_topomesh import VtkActorTopomesh
    from vplants.cellcomplex.property_topomesh.property_topomesh_visualization.vtk_tools import vtk_display_actor, \
       vtk_display_actors
    from vplants.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property

    compute_topomesh_property(topomesh, 'length', 1)

    edge_actor = VtkActorTopomesh()
    edge_actor.set_topomesh(topomesh, 1, property_name='length')
    edge_actor.line_glyph = 'line'
    edge_actor.update(colormap="cool")

    vertex_actor = VtkActorTopomesh()
    vertex_actor.set_topomesh(topomesh, 0)
    # vertex_actor.point_glyph = 'point'
    vertex_actor.point_glyph = 'sphere'
    vertex_actor.glyph_scale = 0.0001
    vertex_actor.update(colormap="Reds")

    vtk_display_actors([vertex_actor.actor, edge_actor.actor], background=(1, 1, 1))
