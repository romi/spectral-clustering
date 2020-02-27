

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


"""
b = DBSCdAN(eps=.03, min_samples=10).fit(evec[:,1:10])
labels=db.labels_

colors = np.random.uniform(0,1,[k,3])

pcd.colors = open3d.Vector3dVector(colors[labels])

open3d.draw_geometries([pcd])
"""

# Partie actualisée dans une fonction dans quotient graph
min_number_of_element_in_a_quotient_node = 50
nodes_to_remove = []
for u in QG.nodes:
    if QG.nodes[u]['intra_class_node_number'] < min_number_of_element_in_a_quotient_node:
        adjacent_clusters = [n for n in QG[u]]
        max_number_of_nodes_in_adjacent_clusters = 0
        for i in range(len(adjacent_clusters)):
            if QG.nodes[adjacent_clusters[i]]['intra_class_node_number'] > max_number_of_nodes_in_adjacent_clusters:
                max_number_of_nodes_in_adjacent_clusters = QG.nodes[adjacent_clusters[i]]['intra_class_node_number']
                new_cluster = adjacent_clusters[i]
        # Opération de fusion du petit cluster avec son voisin le plus conséquent.
        # Mise à jour des attributs de la grande classe et suppression de la petite classe
        QG.nodes[new_cluster]['intra_class_node_number'] += QG.nodes[u]['intra_class_node_number']
        QG.nodes[new_cluster]['intra_class_edge_weight'] += (QG.nodes[u]['intra_class_edge_weight']
                                                             + QG.edges[new_cluster, u]['inter_class_edge_weight'])
        QG.nodes[new_cluster]['intra_class_edge_number'] += (QG.nodes[u]['intra_class_edge_number']
                                                             + QG.edges[new_cluster, u]['inter_class_edge_number'])

        # Mise à jour du lien avec le PointCloudGraph d'origine
        for v in G.nodes:
            if G.nodes[v]['quotient_graph_node'] == u:
                G.nodes[v]['quotient_graph_node'] = new_cluster

        # Mise à jour des edges
        for i in range(len(adjacent_clusters)):
            if QG.has_edge(new_cluster, adjacent_clusters[i]) is False:
                QG.add_edge(new_cluster, adjacent_clusters[i],
                            inter_class_edge_weight=QG.edges[u, adjacent_clusters[i]]['inter_class_edge_weight'],
                            inter_class_edge_number=QG.edges[u, adjacent_clusters[i]]['inter_class_edge_number'])
            elif QG.has_edge(new_cluster, adjacent_clusters[i]) and new_cluster != adjacent_clusters[i]:
                QG.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                    QG.edges[u, adjacent_clusters[i]]['inter_class_edge_weight']
                QG.edges[new_cluster, adjacent_clusters[i]]['inter_class_edge_weight'] += \
                    QG.edges[u, adjacent_clusters[i]]['inter_class_edge_number']
        nodes_to_remove.append(u)

QG.remove_nodes_from(nodes_to_remove)

display_and_export_quotient_graph_matplotlib(QG, node_sizes=20,
                                             filename="quotient_graph_matplotlib_without_small_clusters")

new_classif = np.asarray(list((dict(G.nodes(data='quotient_graph_node')).values())))
new_classif = new_classif[:, np.newaxis]
kpcg.export_anything_on_point_cloud(G, attribute=new_classif, filename='pcd_classif_without_small_clusters.txt')



# Travaux de region growing, transformé en fonction dans QuotientGraph
kmeans_labels = kmeans.labels_[:, np.newaxis]

connected_component_labels = np.zeros(kmeans_labels.shape, dtype=int)
seed_kmeans_labels = []
cluster_colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
seed_colors = []
next_label = 0

visited = np.zeros(kmeans_labels.shape, dtype=int)
queue = []

for i in range(kmeans_labels.shape[0]):

    # Find a seed to initialize the seed fill process.

    if visited[i] == 0:
        seed = i
        visited[i] = 1
        queue.append(i)
        seed_kmeans_labels.append(kmeans_labels[seed])
        seed_colors.append(cluster_colors[kmeans_labels[seed][0]])

        # Region growing from the specified seed.

        while len(queue) != 0:

            current = queue.pop(0)
            connected_component_labels[current] = next_label

            for n in G[current]:
                if kmeans_labels[n] == kmeans_labels[seed] and visited[n] == 0:
                    queue.append(n)
                    visited[n] = 1

        next_label += 1

export_anything_on_point_cloud(G, attribute=connected_component_labels)

quotient_graph_manual = nx.Graph()
quotient_graph_manual.add_nodes_from(range(next_label))
node_quotient_graph_labels = dict(zip(np.asarray(quotient_graph_manual.nodes()), range(next_label)))
node_kmeans_labels_values = dict(
    zip(np.asarray(quotient_graph_manual.nodes()), np.transpose(np.asarray(seed_kmeans_labels))[0]))
nx.set_node_attributes(quotient_graph_manual, node_kmeans_labels_values, 'kmeans_labels')

for (u, v) in G.edges:
    a = connected_component_labels[u, 0]
    b = connected_component_labels[v, 0]
    if a != b:
        if not quotient_graph_manual.has_edge(a, b):
            quotient_graph_manual.add_edge(a, b)

figure = plt.figure(0)
figure.clf()
graph_layout = nx.kamada_kawai_layout(quotient_graph_manual)
colormap = 'jet'
node_sizes = 20
node_color = [quotient_graph_manual.nodes[i]['kmeans_labels'] / 4 for i in quotient_graph_manual.nodes()]
nx.drawing.nx_pylab.draw_networkx(quotient_graph_manual,
                                  ax=figure.gca(),
                                  pos=graph_layout,
                                  with_labels=True,
                                  node_size=node_sizes,
                                  node_color=seed_colors,
                                  labels=node_quotient_graph_labels,
                                  cmap=plt.get_cmap(colormap))

figure.subplots_adjust(wspace=0, hspace=0)
figure.tight_layout()

"""
Gcopy = nx.MultiGraph()
Gcopy.add_nodes_from(G)
Gcopy.add_edges_from(G.edges)

node_clusters_values = dict(zip(G.nodes(), connected_component_labels))
same_attribute = lambda u, v: node_clusters_values[u] == node_clusters_values[v]

quotient_graph = nx.quotient_graph(Gcopy, partition=same_attribute, edge_relation=None,
                                   create_using=nx.MultiGraph)
print("quotient_graph_ok")

figure = plt.figure(0)
figure.clf()
graph_layout = nx.kamada_kawai_layout(G)
colormap = 'jet'
node_size = 10
nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                  ax=figure.gca(),
                                  pos=graph_layout,
                                  with_labels=False,
                                  node_size=node_sizes,
                                  cmap=plt.get_cmap(colormap))

figure.set_size_inches(10 * len(attribute_names), 10)
figure.subplots_adjust(wspace=0, hspace=0)
figure.tight_layout()



G.find_local_minimum_of_gradient_norm()
np.savetxt('min_loc.txt', G.nodes_coords[G.min_local], delimiter=",")

"""

"""
# Obtain the list of points associated with leaves
clusters = kmeans.labels_.tolist()
node_cluster0 = [i for i, x in enumerate(clusters) if x == 0]
clusters_array = kmeans.labels_[:,np.newaxis]
G.clusters_leaves = clusters_array
G.add_anything_as_attribute(anything=clusters_array, name_of_the_new_attribute='clusters_leaves')

# Creating Quotient Graph
clusters_leaves = nx.get_node_attributes(G, 'clusters_leaves')
node_clusters_values = dict(zip(G.nodes(), clusters_array))
same_attribute = lambda u, v: node_clusters_values[u] == node_clusters_values[v]

Gcopy = nx.MultiGraph()
Gcopy.add_nodes_from(G)
Gcopy.add_edges_from(G.edges)

quotient_graph = nx.quotient_graph(Gcopy, partition=same_attribute, edge_relation=None,
                                   create_using=nx.MultiGraph)
print("quotient_graph_ok")

figure = plt.figure(0)
figure.clf()
graph_layout = nx.kamada_kawai_layout(G)
colormap = 'jet'
node_size = 10
nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                  ax=figure.gca(),
                                  pos=graph_layout,
                                  with_labels=False,
                                  node_size=node_sizes,
                                  cmap=plt.get_cmap(colormap))

figure.set_size_inches(10 * len(attribute_names), 10)
figure.subplots_adjust(wspace=0, hspace=0)
figure.tight_layout()
"""