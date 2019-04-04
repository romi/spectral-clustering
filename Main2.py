import open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
import SimilarityGraph as SGk
# from mayavi import mlab
import scipy.cluster.vq as vq
# from sklearn.cluster import DBSCAN

#pcd = open3d.read_point_cloud("data/arabette.ply")
pcd = open3d.read_point_cloud("arabi_ascii_segm.ply")
r = .001
# p_light=open3d.voxel_down_sample(pcd,r)
G = genGraph(pcd, 1.5*r)
G = SGk.genGraph(pcd, 3*r)

A = nx.adjacency_matrix(G)
# np.ravel permet de passer de [[1,2,3],[4,5,6]] à [1 2 3 4 5 6]
D = spsp.csr_matrix(np.diag(np.ravel(np.sum(A, axis=1))), dtype = 'float')

Lcsr = D - A
Lcsr = spsp.csr_matrix.asfptype(Lcsr)
#L = spsp.csr_matrix.toarray(Lcsr)
#print(type(L[1,1]))
#print(type(L))

k = 10

eval, evec = spsp.linalg.eigsh(Lcsr, k)
print(evec)

"""
k=500
means,labels = vq.kmeans2(U[:,1:5],k)
#db = DBSCAN(eps=.03, min_samples=10).fit(U[:,1:10])
#labels=db.labels_

colors = np.random.uniform(0,1,[k,3])

pcd.colors = open3d.Vector3dVector(colors[labels])

open3d.draw_geometries([pcd])
"""