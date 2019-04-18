import open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import SimilarityGraph as SGk
import time
# from mayavi import mlab
import scipy.cluster.vq as vq
# from sklearn.cluster import DBSCAN

import sys
from types import ModuleType, FunctionType
from gc import get_referents

#start = time.time()

#pcd = open3d.read_point_cloud("data/arabette.ply")
pcd = open3d.read_point_cloud("arabi_dense.ply")
r = 8

# p_light=open3d.voxel_down_sample(pcd,r)
G = SGk.genGraph(pcd, r)
SGk.drawGraphO3D(pcd, G)

# A = nx.adjacency_matrix(G)
# np.ravel permet de passer de [[1,2,3],[4,5,6]] à [1 2 3 4 5 6]
# D = spsp.csr_matrix(np.diag(np.ravel(np.sum(A, axis=1))), dtype = 'float')
# Lcsr = D - A
# D = None
# A = None

# fonction condensée plus efficace en quantité de nuages :
Lcsr = nx.laplacian_matrix(G, weight='weight')
print(type(Lcsr))
print(Lcsr.shape[0])
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
k = 30
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
eigenval, eigenvec = np.linalg.eigh(Lcsr.todense())
keigenvec = eigenvec[:,:k]
end2= time.time()

print(keigenvec)
print(keigenvec.shape)

print(end2-start2)


# means,labels = vq.kmeans2(evec, k, minit='points', missing='warn')
# labels = np.asarray(labels.reshape(Lcsr.shape[0], 1), dtype= np.float64)


"""
k = 2
means,labels = vq.kmeans2(evec, k, minit='points', missing='warn')
labels = np.asarray(labels.reshape(Lcsr.shape[0], 1), dtype= np.float64)

b = DBSCdAN(eps=.03, min_samples=10).fit(evec[:,1:10])
labels=db.labels_

colors = np.random.uniform(0,1,[k,3])

pcd.colors = open3d.Vector3dVector(colors[labels])

open3d.draw_geometries([pcd])
"""
"""
pcdtabclassif = np.concatenate([np.asarray(pcd.points), labels], axis = 1)

np.savetxt('Centroides.txt', means, delimiter= ',')
np.savetxt('labels.txt', labels, delimiter= ",")
np.savetxt('pcdclassifkmeans.txt', pcdtabclassif, delimiter = ",")

"""