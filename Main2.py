import open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import SimilarityGraph as SGk
# from mayavi import mlab
import scipy.cluster.vq as vq
# from sklearn.cluster import DBSCAN

import sys
from types import ModuleType, FunctionType
from gc import get_referents

#pcd = open3d.read_point_cloud("data/arabette.ply")
pcd = open3d.read_point_cloud("arabi_dense.ply")
r = 5
# p_light=open3d.voxel_down_sample(pcd,r)
G = SGk.genGraph(pcd, r)
SGk.drawGraphO3D(pcd, G)

# A = nx.adjacency_matrix(G)


# np.ravel permet de passer de [[1,2,3],[4,5,6]] à [1 2 3 4 5 6]
#D = spsp.csr_matrix(np.diag(np.ravel(np.sum(A, axis=1))), dtype = 'float')
#Lcsr = D - A
#D = None
#A = None

# fonction condensée plus efficace en quantité de nuages :
Lcsr = nx.laplacian_matrix(G)
print(type(Lcsr))
print(Lcsr.shape[0])
Lcsr = spsp.csr_matrix.asfptype(Lcsr)

# k nombre de partitions que l'on souhaite obtenir
k = 20
# On précise que l'on souhaite les k premières valeurs propres directement dans la fonction
# Les valeurs propres sont bien classées par ordre croissant
eval, evec = spsp.linalg.eigsh(Lcsr, k)
print(eval)
print(evec)

k = 20
means,labels = vq.kmeans2(evec, k, minit='points', missing='warn')
labels = np.asarray(labels.reshape(Lcsr.shape[0], 1), dtype= np.float64)
"""db = DBSCAN(eps=.03, min_samples=10).fit(evec[:,1:10])
labels=db.labels_

colors = np.random.uniform(0,1,[k,3])

pcd.colors = open3d.Vector3dVector(colors[labels])

open3d.draw_geometries([pcd])"""
pcdtabclassif = np.concatenate([np.asarray(pcd.points), labels], axis = 1)

np.savetxt('Centroides.txt', means, delimiter= ',')
np.savetxt('labels.txt', labels, delimiter= ",")
np.savetxt('pcdclassifkmeans.txt', pcdtabclassif, delimiter = ",")