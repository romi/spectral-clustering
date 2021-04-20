import open3d as open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import scipy.signal as spsig
import scipy as sp
import scipy.cluster.vq as vq
import spectral_clustering.similarity_graph as SGk
import spectral_clustering.branching_graph as BGk
import sklearn.cluster as sk
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

## A partir de l'import d'un fichier texte
# Calcul des Kmeans sur un nuage de points en fichier .txt
# En prenant en compte les coordonnées et le label.
#pointcloud_with_label = np.loadtxt(fname="/Users/katiamirande/Documents/Tests/Tests graphes simples/Données_hypothèse_monotonie/Branches_t20_espacement_aleatoire/testchain1.txt"
#                                   , delimiter=',', usecols=range(4))

#pointcloud_with_label_sort = pointcloud_with_label[pointcloud_with_label[:,3].argsort()]
#vp2grad = np.gradient(pointcloud_with_label_sort[:,3])

"""
G = BGk.BranchingGraph(500)
G.add_branch(branch_size=100, linking_node=200)
G.add_branch(branch_size=20, linking_node=530)
G.add_branch(branch_size=10, linking_node=610)

G.add_branch(branch_size=100, linking_node=350)
G.add_branch(branch_size=20, linking_node=665)
G.add_branch(branch_size=10, linking_node=745)

G.compute_graph_eigenvectors()
#labels = G.keigenvec[:, 1]
#labels_sorts = labels[labels.argsort()]
#vp2grad = np.gradient(labels_sorts)
vp2grad = np.gradient(labels)
figureval = plt.figure(0)
figureval.clf()
figureval.gca().plot(range(len(np.transpose(vp2grad))), np.transpose(vp2grad), 'bo')
figureval.set_size_inches(20, 10)
figureval.subplots_adjust(wspace=0, hspace=0)
figureval.tight_layout()
figureval.savefig('/Users/katiamirande/Documents/Tests/Tests_gradient__sort_vp2')
"""

#c = 2
#means,labels = vq.kmeans2(pointcloud_with_label, c, minit='points', missing='warn')
#pcdtabclassif = sk.agglomerativeClustering().fit(pointcloud_with_label)

#labels = np.asarray(labels.reshape(pointcloud_with_label.shape[0], 1), dtype= np.float64)


#np.savetxt('Centroides.txt', means, delimiter= ',')
#np.savetxt('/Users/katiamirande/Documents/Tests/Tests graphes simples/Données/Chaine100Branchet25_n35/PCD_withvp2_segmented_with_agglomerativeclustering.txt', pcdtabclassif, delimiter = ",")

## Création du graph pour obtenir les données associées.
# Objectif : utilisation d'un algorithme sci kit : agglomerative clustering.

"""
#G = BGk.BranchingGraph(100)
#G.add_branch(branch_size=20, linking_node=35)

#G = BGk.BranchingGraph(100)
#G.add_branch(branch_size=20, linking_node=30)
#G.add_branch(branch_size=20, linking_node=35)
#G.add_branch(branch_size=20, linking_node=40)
#G.add_branch(branch_size=20, linking_node=47)
#G.add_branch(branch_size=20, linking_node=50)
#G.add_branch(branch_size=20, linking_node=55)
#G.add_branch(branch_size=20, linking_node=62)
#G.add_branch(branch_size=20, linking_node=63)
#G.add_branch(branch_size=20, linking_node=70)

G = BGk.BranchingGraph(500)
G.add_branch(branch_size=100, linking_node=200)
G.add_branch(branch_size=20, linking_node=530)
G.add_branch(branch_size=10, linking_node=610)

def clustering_by_fiedler_and_agglomerative(self):

    G.compute_graph_eigenvectors()
    G.add_eigenvector_value_as_attribute(k=2)

    A = nx.adjacency_matrix(G)
    vp2 = np.asarray(G.keigenvec[:,1])

    vp2_matrix = np.tile(vp2,(len(G),1))
    vp2_matrix[A.todense()==0] = np.nan
    vp2grad = (np.nanmax(vp2_matrix,axis=1) - np.nanmin(vp2_matrix,axis=1))/2.
    node_vp2grad_values = dict(zip(G.nodes(), vp2grad))
    nx.set_node_attributes(G, node_vp2grad_values, 'gradient_vp2')

    clustering = sk.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='single', n_clusters=6).fit(vp2grad[:, np.newaxis])

    node_clustering_label = dict(zip(G.nodes(), np.transpose(clustering.labels_)))
    nx.set_node_attributes(G, node_clustering_label, 'clustering_label')
    
BGk.save_graph_plot(G, ['gradient_vp2'], colormap='Reds',
                        filename='/Users/katiamirande/Documents/Tests/Tests_gradient_vp2/Tests_gradientminmax_agglomerativeclustering/valeurs_gradient_vp2')

BGk.save_graph_plot(G, ['clustering_label'], colormap='plasma', filename='/Users/katiamirande/Documents/Tests/Tests_gradient_vp2/Tests_gradientminmax_agglomerativeclustering/clusters_gradient_vp2')

"""

pcd = open3d.read_point_cloud("Data/arabi_densep_clean_segm.ply")
r = 8
G = SGk.create_riemannian_graph(pcd, method='knn', nearest_neighbors=r)

Lcsr = nx.laplacian_matrix(G, weight='weight')
Lcsr = spsp.csr_matrix.asfptype(Lcsr)
# k nombre de vecteurs propres que l'on veut calculer
k = 2
keigenval, keigenvec = spsp.linalg.eigsh(Lcsr,k=k,sigma=0, which='LM')
eigenval = keigenval.reshape(keigenval.shape[0], 1)

A = nx.adjacency_matrix(G)
vp2 = np.asarray(keigenvec[:, 1])

pcd_vp2 = np.concatenate([np.asarray(pcd.points), vp2[:,np.newaxis]], axis=1)
np.savetxt('pcd_vp2.txt', pcd_vp2, delimiter=",")
print("Export du nuage avec vecteur propre 2")

vp2_matrix = A.copy()
vp2_matrix[A.nonzero()] = vp2[A.nonzero()[1]]


#Second method fo gradient, just obtain the max and min value in the neighborhood.
node_neighbor_max_vp2 = np.array([vp2_matrix[node].data.max() for node in range(A.shape[0])])
node_neighbor_min_vp2 = np.array([vp2_matrix[node].data.min() for node in range(A.shape[0])])
vp2grad = node_neighbor_max_vp2 - node_neighbor_min_vp2

#First method not adapted to big clouds
#node_neighbor_max_vp2 = np.array([vp2[A[node].nonzero()[1]].max() for node in range(A.shape[0])])
#node_neighbor_min_vp2 = np.array([vp2[A[node].nonzero()[1]].min() for node in range(A.shape[0])])
"""
node_neighbor_max_vp2_node = np.array([np.argmax(vp2_matrix[node].data) for node in range(A.shape[0])])
node_neighbor_max_vp2 = np.array([vp2_matrix[node].data[neighbor_max_node] for node,neighbor_max_node in zip(range(A.shape[0]),node_neighbor_max_vp2_node)])

node_neighbor_min_vp2_node = np.array([np.argmin(vp2_matrix[node].data) for node in range(A.shape[0])])
node_neighbor_min_vp2 = np.array([vp2_matrix[node].data[neighbor_min_node] for node,neighbor_min_node in zip(range(A.shape[0]),node_neighbor_min_vp2_node)])

pcdtab = np.asarray(pcd.points)
vp2_max_min = (node_neighbor_max_vp2 - node_neighbor_min_vp2)

grad_weight = np.zeros(node_neighbor_min_vp2_node.shape[0])
for i in range(node_neighbor_min_vp2_node.shape[0]):
    grad_weight[i] = sp.spatial.distance.euclidean(pcdtab[node_neighbor_max_vp2_node[i]].reshape(1, 3), pcdtab[node_neighbor_min_vp2_node[i]].reshape(1, 3))

vp2grad = np.divide(vp2_max_min, grad_weight)
"""

pcd_vp2_grad = np.concatenate([np.asarray(pcd.points), vp2grad[:,np.newaxis]], axis=1)
np.savetxt('pcd_vp2_grad.txt', pcd_vp2_grad, delimiter=",")
print("Export du nuage avec gradient du vecteur propre 2")


pcd_vp2_grad_vp2 = np.concatenate([pcd_vp2_grad, vp2[:,np.newaxis]], axis=1)
pcd_vp2_grad_vp2_sort_by_vp2 = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:,4].argsort()]
figure = plt.figure(0)
figure.clf()
figure.gca().set_title("fiedler vector")
# figure.gca().plot(range(len(vec)),vec,color='blue')
figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_vp2[:, 4], color='blue')
figure.set_size_inches(10, 10)
figure.subplots_adjust(wspace=0, hspace=0)
figure.tight_layout()
figure.savefig("fiedler_vector")
print("Export du vecteur propre 2")


figure = plt.figure(1)
figure.clf()
figure.gca().set_title("Gradient of fiedler vector")
# figure.gca().plot(range(len(vec)),vec,color='blue')
figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_vp2[:, 3], color='blue')
figure.set_size_inches(10, 10)
figure.subplots_adjust(wspace=0, hspace=0)
figure.tight_layout()
figure.savefig("Gradient_of_fiedler_vector")
print("Export du gradient trié selon vecteur propre 2")

# vp2_matrix = np.tile(vp2, (len(G), 1))
# vp2_matrix[A.todense() == 0] = np.nan
# vp2grad = (np.nanmax(vp2_matrix, axis=1) - np.nanmin(vp2_matrix, axis=1)) / 2.

node_vp2grad_values = dict(zip(G.nodes(), vp2grad))

#mean_gradient = np.mean(vp2grad)

#maximas_grad = np.diff(np.sign(np.diff(vp2grad)))
#number_maximas = np.count_nonzero(maximas_grad)
#number_maximas = sum(1 for i in maximas_grad if i < 0)

#clustering via nombre de clusters
clustering = sk.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward', n_clusters=11).fit(vp2grad[:, np.newaxis])

#clustering via distance_threshold
#clustering = sk.AgglomerativeClustering(affinity='euclidean',
#                                        connectivity=A,
#                                        linkage='ward',
#                                        n_clusters=None,
#                                        compute_full_tree=True,
#                                        distance_threshold=mean_gradient).fit(vp2grad[:, np.newaxis])



label = np.asarray(clustering.labels_)
pcdtabclassif = np.concatenate([np.asarray(pcd.points), label[:,np.newaxis]], axis=1)
np.savetxt('pcdclassifagglo11.txt', pcdtabclassif, delimiter=",")
print("Export du nuage classifié")
