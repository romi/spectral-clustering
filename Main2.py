import open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import SimilarityGraph as SGk
import time
# from mayavi import mlab
import scipy.cluster.vq as vq
# from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import sys
from types import ModuleType, FunctionType
from gc import get_referents

#start = time.time()

#pcd = open3d.read_point_cloud("data/arabette.ply")
pcd = open3d.read_point_cloud("arabi_densep_clean.ply")
r = 8

# p_light=open3d.voxel_down_sample(pcd,r)
G = SGk.genGraph(pcd, r)
SGk.drawGraphO3D(pcd, G)

#Connect = nx.number_connected_components(G)
#print("Nombre d'éléments connectés")
#print(Connect)
# A = nx.adjacency_matrix(G)
# np.ravel permet de passer de [[1,2,3],[4,5,6]] à [1 2 3 4 5 6]
# D = spsp.csr_matrix(np.diag(np.ravel(np.sum(A, axis=1))), dtype = 'float')
# Lcsr = D - A
# D = None
# A = None

# fonction condensée plus efficace en quantité de nuages :
Lcsr = nx.laplacian_matrix(G, weight='weight')
#print(type(Lcsr))
#print(Lcsr.shape[0])
Lcsr = spsp.csr_matrix.asfptype(Lcsr)

#end = time.time()


#print(Lcsr)
#print(end-start)


#sparsity = 1.0 - (Lcsr.count_nonzero() / float(Lcsr.shape[0] * Lcsr.shape[0]))
#density = 1 - sparsity
#print(sparsity)
#print(density)
#c = nx.number_of_edges(G)
#print(c)


# k nombre de partitions que l'on souhaite obtenir
k = 50
# On précise que l'on souhaite les k premières valeurs propres directement dans la fonction
# Les valeurs propres sont bien classées par ordre croissant
#Ldense = Lcsr.todense()
#eigenval, eigenvec = np.linalg.eigh(Ldense)
#keigenvec = eigenvec[:,:k]

# start = time.time()
#
# eval, evec = spsp.linalg.eigsh(Lcsr, k, which = 'SA')
# print(evec)
# print(evec.shape)
#
# end = time.time()
#
# print(end - start)

start2 = time.time()
#eigenval, eigenvec = np.linalg.eigh(Lcsr.todense())
keigenval, keigenvec = spsp.linalg.eigsh(Lcsr,k=k,sigma=0, which='LM')

#keigenvec = eigenvec[:,:k]
end2 = time.time()
# test pour vérifier que la matrice keigenvec est non nulle
#all_zeros = np.all(keigenvec == 0)
#print(all_zeros)
#print(type(keigenvec))

#print(keigenvec)
#print(keigenvec.shape)

#print(end2-start2)


# means,labels = vq.kmeans2(evec, k, minit='points', missing='warn')
# labels = np.asarray(labels.reshape(Lcsr.shape[0], 1), dtype= np.float64)
k = 49
#SGk.VisuEigenvecPts(pcd, keigenvec, k, 1)
#SGk.VisuEspaceSpecDim3(keigenvec, 1, 1, 2, 3)


#k = 2
#means,labels = vq.kmeans2(keigenvec, k, minit='points', missing='warn')
#labels = np.asarray(labels.reshape(Lcsr.shape[0], 1), dtype= np.float64)


# Code pour les courbes représentant les différents vecteurs propres en chaque point du nuage
"""
figure = plt.figure(0)
figure.clf()

sortkeigenvec = keigenvec[keigenvec[:,1].argsort()]
for i_vec, vec in enumerate(np.transpose(np.around(sortkeigenvec,10))):
    figure.add_subplot(5,10,i_vec+1)
    figure.gca().set_title("Eigenvector "+str(i_vec+1))
    figure.gca().plot(range(len(vec)),vec,color='blue')

figure.set_size_inches(20,10)
figure.subplots_adjust(wspace=0,hspace=0)
figure.tight_layout()
figure.savefig("eigenvectors.png")

# Extraction du premier vecteur propre pour comprendre ce qu'il se passe.
fielder = keigenvec[:,0]
fielder = fielder[fielder.argsort()]
figurefielder = plt.figure(0)
figurefielder.clf()
figurefielder.gca().plot(range(len(np.transpose(fielder))),np.transpose(fielder), color='red')
figurefielder.set_size_inches(20,10)
figurefielder.subplots_adjust(wspace=0,hspace=0)
figurefielder.tight_layout()
figure.savefig("fielder.png")
"""

# Graphique contenant les valeurs propres des 50 premiers vecteurs propres
figureval = plt.figure(0)
figureval.clf()
figureval.gca().plot(range(len(np.transpose(keigenval))),np.transpose(keigenval), 'bo')
figureval.set_size_inches(20,10)
figureval.subplots_adjust(wspace=0,hspace=0)
figureval.tight_layout()
figureval.savefig("ValeursPropres.png")


#pcdtabclassif = np.concatenate([np.asarray(pcd.points), labels], axis = 1)

#np.savetxt('Centroides.txt', means, delimiter= ',')
#np.savetxt('labels.txt', labels, delimiter= ",")
#np.savetxt('pcdclassifkmeans.txt', pcdtabclassif, delimiter = ",")
