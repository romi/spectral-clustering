import open3d
import networkx as nx
import numpy as np
import scipy.sparse as spsp
# from mayavi import mlab
import scipy.cluster.vq as vq
# from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import spectral_clustering.similarity_graph as SGk

# import spectral_clustering.branching_graph as TCk

#pcd = open3d.read_point_cloud("data/arabette.ply")
pcd = open3d.read_point_cloud("Data/as_cylinders.ply")
r = 4.5

# p_light=open3d.voxel_down_sample(pcd,r)
G = SGk.gen_graph(pcd, method='radius', radius=r)
# SGk.drawGraphO3D(pcd, G)

#Connect = nx.number_connected_components(G)
#print("Nombre d'éléments connectés")
#print(Connect)
# A = nx.adjacency_matrix(G)
# np.ravel permet de passer de [[1,2,3],[4,5,6]] à [1 2 3 4 5 6]
# D = spsp.csr_matrix(np.diag(np.ravel(np.sum(A, axis=1))), dtype = 'float')
# Lcsr = D - A
# D = None
# A = None


k = 10
# On précise que l'on souhaite les k premières valeurs propres directement dans la fonction
# Les valeurs propres sont bien classées par ordre croissant

# Calcul des k premiers vecteurs et valeurs propres
keigenval, keigenvec = SGk.graph_spectrum(G, k=k, smallest_first=False)

# TCk.ploteigenvec(keigenvec)
# Nombre de clusters attendus
#c = 2
#means,labels = vq.kmeans2(keigenvec, c, minit='points', missing='warn')
#labels = np.asarray(labels.reshape(len(G), 1), dtype= np.float64)

# Fonctions d'export
SGk.export_eigenvectors_on_pointcloud(pcd, keigenvec, k=9, filename='vp9last')
#SGk.export_pointcloud_on_eigenvectors_3d(keigenvec, 1, 2, 3)




#pcdtabclassif = np.concatenate([np.asarray(pcd.points), labels], axis=1)

#np.savetxt('Centroides.txt', means, delimiter= ',')
#np.savetxt('labels.txt', labels, delimiter= ",")
#np.savetxt('pcdclassifkmeans.txt', pcdtabclassif, delimiter=",")



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

"""
# Graphique contenant les valeurs propres des 50 premiers vecteurs propres
figureval = plt.figure(0)
figureval.clf()
figureval.gca().plot(range(len(np.transpose(keigenval))),np.transpose(keigenval), 'bo')
figureval.set_size_inches(20,10)
figureval.subplots_adjust(wspace=0,hspace=0)
figureval.tight_layout()
figureval.savefig("ValeursPropres.png")
"""



