import open3d as open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import scipy.cluster.vq as vq
import spectral_clustering.similarity_graph as SGk
import spectral_clustering.branching_graph as BGk
import sklearn.cluster as sk
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

pcd = open3d.read_point_cloud("Data/arabi_densep_clean.ply")
r = 8
G = SGk.gen_graph(pcd, method='knn', nearest_neighbors=r)

Lcsr = nx.laplacian_matrix(G, weight='weight')
Lcsr = spsp.csr_matrix.asfptype(Lcsr)
# k nombre de vecteurs propres que l'on veut calculer
k = 2
keigenval, keigenvec = spsp.linalg.eigsh(Lcsr,k=k,sigma=0, which='LM')
eigenval = keigenval.reshape(keigenval.shape[0], 1)

A = nx.adjacency_matrix(G)
vp2 = np.asarray(keigenvec[:, 1])
vp2_matrix = np.tile(vp2, (len(G), 1))
vp2_matrix[A.todense() == 0] = np.nan
vp2grad = (np.nanmax(vp2_matrix, axis=1) - np.nanmin(vp2_matrix, axis=1)) / 2.
node_vp2grad_values = dict(zip(G.nodes(), vp2grad))

clustering = sk.AgglomerativeClustering(affinity='euclidean', connectivity=A, linkage='ward', n_clusters=30).fit(vp2grad[:, np.newaxis])

label = np.asarray(clustering.labels_)
pcdtabclassif = np.concatenate([np.asarray(pcd.points), label[:,np.newaxis]], axis=1)
np.savetxt('pcdclassifagglo.txt', pcdtabclassif, delimiter=",")
