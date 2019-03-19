# Ouverture du fichier
import numpy as np
import open3d as op

# Calcul matrice de similarité
import math as m
import scipy.spatial as sps
import scipy.cluster as spc

# Calcul matrice des degrés
import math as math

# Outils matrices creuses
import scipy.sparse as spmat

# Ouverture et stockage du nuage de points
pcd = op.read_point_cloud("arabi_ascii_segm.ply")
print(pcd)

# op.draw_geometries([pcd])

# Permet de lire le nuage de point comme un tableau
pcdtab = np.asarray(pcd.points)

# print(pcdtab)


# Méthode Seuil
# pour l'instant, simple calcul de distances pour avoir une matrice contenant des pondérations
# Cette méthode python renvoie un array de type "Ndarray"
disimatrix = sps.distance_matrix(pcdtab, pcdtab, p=2, threshold=100000)

# Division des termes pour obtenir une matrice de disimilarité
# Gestion des distances égales à zéro (on remplace les zéros par des nombres négatifs pour les remplacer par 0 par la suite)
disimatrix[disimatrix == 0] = -1
simatrix = 1/disimatrix
simatrix[simatrix < 0] = 0

disimatrix = None

# Fonction seuil pour avoir une matrice creuse et éliminer les côtés peu pondérés
simatrix[simatrix < 700] = 0
print(simatrix)

# Calcul de la densité de la matrice
sparsity = 1.0 - (np.count_nonzero(simatrix) / float(simatrix.size))
density = 1 - sparsity
print(sparsity)
print(density)

# Stockage du graphe réalisé sous forme d'un maillage ? Avec les points reliés en fonction de la matrice de similarité ?

# Matrice des degrés
# Somme selon un axe (ici les lignes)
# Attention à la fonction np.sum, la sortie présente un problème de dimension, interet du .reshape
size = simatrix.shape[0]
degre = np.sum(simatrix, axis=1, dtype='float').reshape(1, size)
identitymat = np.identity(size)
# Récupération des indices de la diagonale et insertion des degrés dans la matrice
di = np.diag_indices(size)
degremat = identitymat
degremat[di] = degre
print(degremat)

# Calcul matrice Laplacienne L = D - W
L = degremat - simatrix
print(L)


degremat = None
simatrix = None
identitymat = None
degre = None

# Calcul des vecteurs propres
# eigh pour les matrices réelles symétriques

# k nombre de partitions que l'on souhaite obtenir
k = 25
# On garde les k premières valeurs propres
eigenval, eigenvec = np.linalg.eigh(L)
keigenvec = eigenvec[:,:k]
# print(eigenvec)
eigenvec = None
eigenval = None
L = None

# On souhaite trier en fonction des lignes, et non pas des colonnes. Transposition NON nécessaire, voir def fonction kmeans
# u = np.transpose(keigenvec)

centroides, label = spc.vq.kmeans2(keigenvec, k, minit='points', missing='warn')
label1 = np.asarray(label.reshape(size, 1), dtype= np.float64)
keigenvec = None

#print(type(pcdtab))
#print(type(pcdtab[0,0]))
#print(type(label1[0,0]))


pcdtabclassif = np.concatenate([pcdtab, label1], axis = 1)

np.savetxt('Centroides.txt', centroides, delimiter= ',')
np.savetxt('labels.txt', label1, delimiter= ",")
np.savetxt('pcdclassifkmeans.txt', pcdtabclassif, delimiter = ",")