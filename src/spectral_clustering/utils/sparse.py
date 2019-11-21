import numpy as np


# Densité d'une matrice
def sparsity(simatrix):
    sparsity = 1.0 - (np.count_nonzero(simatrix) / float(simatrix.size))
    print(sparsity)
    return sparsity


def density(simatrix):
    density = 1 - sparsity(simatrix)
    print(density)
    return density
