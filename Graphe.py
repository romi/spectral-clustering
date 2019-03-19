# Calcul/représentation graphe de similarité
from vplants.cellcomplex.property_topomesh.property_topomesh_creation import edge_topomesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D display
figure = plt.figure()
figure.add_subplot(111, projection='3d')
x = pcdtab[0]
y = pcdtab[1]
z = pcdtab[2]
s,t = np.meshgrid(np.arange(len(x)),np.arange(len(x)))
sources = s[simatrix<3]
targets = t[simatrix<3]
sources, targets = sources[sources!=targets], targets[sources!=targets]

figure.gca().scatter(x,y,z,color='b')
for s,t in zip(sources,targets):
  figure.gca().plot([x[s],x[t]],[y[s],y[t]],[z[s],z[t]],color='r')

figure.gca().axis('equal')
figure.show()

# Voir open3d
# Création d'un dictionnaire de points
# id = np.arange(pcdtab.shape[0])


# Solution, travail avec Guillaume
#topomesh = edge_topomesh(edges, points)