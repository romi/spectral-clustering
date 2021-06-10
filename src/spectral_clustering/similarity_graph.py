import open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import os
# from mayavi import mlab
import scipy.cluster.vq as vq
# from sklearn.cluster import DBSCAN

from spectral_clustering.point_cloud_graph import *

from spectral_clustering.utils.sparse import sparsity

# p est le nuage de points pcd, r_nn le seuil pour la détermination des connexions
# fonction permettant la création d'un graphe grâce à la librairie networkx
def create_riemannian_graph(pcd, method='knn', nearest_neighbors=1, radius=1.):
    """Generate a similarity graph from a point cloud.

    Parameters
    ----------
    p : open3d.open3d.geometry.PointCloud
        Point cloud on which to compute the similarity graph
    method : str
        Search method : 'knn' or 'radius'
    nearest_neighbors : int
        Number of nearest neighbors for the 'knn' method
    radius : float
        Radius for the 'radius' method

    Returns
    -------
    nx.Graph
        Similarity graph computed on the point cloud

    """
    N = len(pcd.points)
    # définition d'un arbre KD contenant tous les points
    tree = open3d.KDTreeFlann(pcd)

    open3d.geometry.estimate_normals(pcd)

    # Prise des points sous forme de tableau ndarray
    pts = np.array(pcd.points)
    normals = np.array(pcd.normals)

    # Déclaration d'un graph networkx
    G = nx.Graph()

    # On insère chaque point du nuage de points dans le graphe avec un numéro et le trio de coordonnées (pos) en attributs
    for i in range(N):
        G.add_node(i, pos=pts[i], normal=normals[i])

    # Construction des edges du graphe à partir d'un seuil
    # On part de la structure de nuage de points en KDTree
    # Cette structure dans open3d dispose de fonctions pour seuil, KNN, RKNN
    for i in range(N):
        if method == 'radius':
            [k, idxs, _] = tree.search_radius_vector_3d(pts[i], radius)
        elif method == 'knn':
            [k, idxs, _] = tree.search_knn_vector_3d(pts[i], nearest_neighbors)
        for idx in idxs:
            d = np.sqrt(np.square(pts[i][0] - pts[idx][0]) + np.square(pts[i][1] - pts[idx][1]) + np.square(
                pts[i][2] - pts[idx][2]))
            if d != 0:
                w = 1 / d
                G.add_edge(i, idx, weight = w)

    return G


def create_connected_riemannian_graph(point_cloud, method='knn', nearest_neighbors=1, radius=1.):
    G = create_riemannian_graph(pcd=point_cloud, method=method, nearest_neighbors=nearest_neighbors, radius=radius)
    pcd2=point_cloud
    if nx.is_connected(G) is False:
        print('not connected')
        largest_cc = max(nx.connected_components(G), key=len)
        # creating the new pcd point clouds
        coords = np.zeros((len(largest_cc), 3))
        i = 0
        for node in largest_cc:
            coords[i, :] = G.nodes[node]['pos']
            i += 1
        np.savetxt('New_pcd_connected.txt', coords, delimiter=' ', fmt='%f')
        path = os.getcwd()
        pcd2 = open3d.read_point_cloud(path+"/New_pcd_connected.txt", format='xyz')

        G = create_riemannian_graph(pcd2, method=method, nearest_neighbors=nearest_neighbors, radius=radius)
        pcd = pcd2

    return G, pcd2

# affichage via open3D
# En entrée : p nuage de points
def draw_graph_open3d(pcd, G):
    graph = open3d.LineSet()
    graph.points = pcd.points
    graph.lines = open3d.Vector2iVector(G.edges)
    open3d.draw_geometries([graph])


# affichage du graphe via CellComplex
# en entrée : la matrice d'adjacence (matrice de similarité) et le nuage de points importé/lu via open3D
def draw_graph_cellcomplex(pcd, G, pcd_to_superimpose):

    if type(pcd) is open3d.geometry.PointCloud:
        pcdtab = np.asarray(pcd.points)
    else:
        pcdtab = pcd
    # s, t = np.meshgrid(np.arange(len(pcdtab)), np.arange(len(pcdtab)))
    # sources = s[simatrix > 0]
    # targets = t[simatrix > 0]
    # sources, targets = sources[sources < targets], targets[sources < targets]

    from cellcomplex.property_topomesh.creation import edge_topomesh
    from cellcomplex.property_topomesh.visualization.vtk_actor_topomesh import VtkActorTopomesh
    from cellcomplex.property_topomesh.visualization.vtk_tools import vtk_display_actors
    from cellcomplex.property_topomesh.analysis import compute_topomesh_property
    from cellcomplex.property_topomesh.creation import vertex_topomesh

    topomesh = edge_topomesh(np.array([e for e in G.edges if e[0]!=e[1]]), dict(zip(np.asarray([n for n in G.nodes]), pcdtab)))

    compute_topomesh_property(topomesh, 'length', 1)

    edge_actor = VtkActorTopomesh()
    edge_actor.set_topomesh(topomesh, 1, property_name='length')
    edge_actor.line_glyph = 'tube'
    edge_actor.update(colormap="cool")

    vertex_actor = VtkActorTopomesh()
    vertex_actor.set_topomesh(topomesh, 0)
    # vertex_actor.point_glyph = 'point'
    vertex_actor.point_glyph = 'sphere'
    vertex_actor.glyph_scale = 2
    vertex_actor.update(colormap="Reds")

    point_cloud_actor = VtkActorTopomesh(vertex_topomesh(pcd_to_superimpose), 0)
    point_cloud_actor.point_glyph = 'point'
    point_cloud_actor.update(colormap="Blues")

    vtk_display_actors([vertex_actor.actor, edge_actor.actor, point_cloud_actor.actor], background=(0.9, 0.9, 0.9))


def export_eigenvectors_on_pointcloud(pcd, keigenvec, k, filename='vecteurproprecol.txt'):
    # Le facteur multiplicatif est présent uniquement pour pouvoir éventuellement mieux afficher les couleurs/poids dans CloudCompare
    #
    label = keigenvec[:, k]
    size = label.shape[0]
    label = np.asarray(label.reshape(size, 1), dtype=np.float64)
    pcd = np.array(pcd.points)
    pcdtabvecteurpropre = np.concatenate([pcd, label], axis=1)
    np.savetxt(filename, pcdtabvecteurpropre, delimiter=',')


def export_pointcloud_on_eigenvectors_3d(keigenvec, vec1, vec2, vec3, filename='espacespec.txt'):
    pts = keigenvec[:,[vec1, vec2, vec3]]
    pts = pts.reshape(keigenvec.shape[0], 3)
    np.savetxt(filename, pts, delimiter=',')
