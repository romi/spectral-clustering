# Tests sur graphes chaînes

########### Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import SimilarityGraph as SGk
import open3d
########### Définitions Fonctions

# Création de la tige principale
# n , Taille de la chaîne principale
def initige(n):
    # Création chaîne via Networkx
    G = nx.Graph()
    list_nodes = [i for i in range(n)]
    #print(list_nodes)
    G.add_nodes_from(list_nodes)
    # Créer tuples reliant un point avec le précédent
    list_edge = [(i, i+1) for i in range(n-1)]
    #print(list_edge)
    G.add_edges_from(list_edge)
    # Export visuel
    # Création de points / coordonnées dans une matrice
    # ici ils sont espacés de 1 à chaque fois.
    C = np.asarray([[0, 0, i] for i in range(n)])
    #print(C)
    #print(type(C))
    #print(np.shape(C))
    return G, C

# Ajout d'une branche à la tige principale
# Paramètres en entrée : Tb taille de la branche
# Eb nom du node sur lequel la branche est ajoutée
# G graphe de travail.
# Matrice de coordonnées C tige origine
# Facteur pour nommer les noeuds F : à modifier si l'on fait deux branches sur le même noeud
# Choisir F supérieur à 1000 et par *10.
# Prendre en compte la taille de la tige principale.
# Sens pour voir les deux branches dans deux directions différentes
# Jouer avec Dir si on a plus de deux branches au même endroit.
def ajoutbranche(G, C, Tb, Eb, F = 1000, Sens = 1, Dir = 0):
    # Création de la branche et ajout dans le graphe
    list_nodes = [i+(Eb*F) for i in range(Tb)]
    G.add_nodes_from(list_nodes)
    list_edge = [(i+(Eb*F), i+1+(Eb*F)) for i in range(Tb-1)]
    G.add_edges_from(list_edge)
    # On relie la branche au point d'insertion
    G.add_edge(Eb, Eb*F)
    # Ajout de coordonnées pour représentation graphique
    Cbranche = np.asarray([[Dir, Sens*(i + 1), C[Eb, 2]] for i in range(Tb)])
    Cbranche = Cbranche.reshape(Tb, 3)
    #print(np.shape(Cbranche))
    Ctot = np.concatenate((C, Cbranche), axis = 0)
    C = Ctot
    #print(np.shape(C))
    return G, C

# Calcul des vecteurs propres du graphe
# G le graphe
def calculvp(G):
    # Appli Laplacien
    L = nx.laplacian_matrix(G, weight='weight')
    #print(type(L))
    #print(np.shape(L))
    L = L.toarray()
    #print(type(L))
    #print(np.shape(L))
    # Calcul vecteurs propres
    # Utilisation de eigsh impossible lorsque le graphe est petit.
    # eigh calcul tous les vecteurs propres.
    keigenval, keigenvec = np.linalg.eigh(L)
    #print(type(keigenvec))
    #print(np.shape(keigenvec))
    return keigenvec, keigenval

# Cette fonction génère des fichiers contenant les nuages de points pour chaque vecteur propre.
# k le nombre de vecteurs propres voulus
# C la matrice des coordonnées
# keigenvec la matrice des vecteurs propres
def exportsurnuage(k, C, keigenvec):
    keigenvec = np.asarray(keigenvec[:,:k])
    #print(np.shape(keigenvec))
    #print(np.shape(C))
    # Concaténation avec les labels (vecteurs propres)
    # Faire boucle pour sortir tous les nuages de points associés à chaque vecteur propre.
    for i in range(k):
        keigenvecK = keigenvec[:, i].reshape(keigenvec.shape[0], 1)
        #print(np.shape(keigenvecK))
        pcdtabclassif = np.concatenate((C, keigenvecK), axis=1)
        # Sortie de tous les nuages
        NomFichier = 'testchain' + str(i)
        print(NomFichier)
        np.savetxt(NomFichier + '.txt', pcdtabclassif, delimiter=",")

def ploteigenval(keigenval):
    figureval = plt.figure(0)
    figureval.clf()
    figureval.gca().plot(range(len(np.transpose(keigenval))),np.transpose(keigenval), 'bo')
    figureval.set_size_inches(20,10)
    figureval.subplots_adjust(wspace=0,hspace=0)
    figureval.tight_layout()
    figureval.savefig("ValeursPropres.png")

def ploteigenvec(keigenvec):
    figure = plt.figure(0)
    figure.clf()
    keigenvec = keigenvec[:,:50]
    sortkeigenvec = keigenvec[keigenvec[:,1].argsort()]
    for i_vec, vec in enumerate(np.transpose(np.around(sortkeigenvec,10))):
        figure.add_subplot(5,10,i_vec+1)
        figure.gca().set_title("Eigenvector "+str(i_vec+1))
        figure.gca().plot(range(len(vec)),vec,color='blue')
    figure.set_size_inches(20,10)
    figure.subplots_adjust(wspace=0,hspace=0)
    figure.tight_layout()
    figure.savefig("eigenvectors.png")

def ploteigenvectopo(keigenvec):
    figure = plt.figure(0)
    figure.clf()
    keigenvec = keigenvec[:,:50]
    for i_vec, vec in enumerate(np.transpose(np.around(keigenvec,10))):
        figure.add_subplot(5,10,i_vec+1)
        figure.gca().set_title("Eigenvector "+str(i_vec+1))
        figure.gca().plot(range(len(vec)),vec,color='blue')
    figure.set_size_inches(20,10)
    figure.subplots_adjust(wspace=0,hspace=0)
    figure.tight_layout()
    figure.savefig("eigenvectorstopo.png")

def Convert(lst):
    res_dct = {i : v for i,v in enumerate(lst)}
    return res_dct

########## Corps du programme : tests

[G, C] = initige(100)
[G, C] = ajoutbranche(G, C, 20, 25)
[G, C] = ajoutbranche(G, C, 10, 50)
[G, C] = ajoutbranche(G, C, 15, 60, 1000, 1)
[G, C] = ajoutbranche(G, C, 10, 60, 10000, -1)
[G, C] = ajoutbranche(G, C, 25, 35, 1000, 1)
[G, C] = ajoutbranche(G, C, 15, 40, 1000, -1)
[G, C] = ajoutbranche(G, C, 8, 47, 1000, 1)
[G, C] = ajoutbranche(G, C, 20, 50, 1000, -1)
[G, C] = ajoutbranche(G, C, 25, 55, 1000, 1)
[G, C] = ajoutbranche(G, C, 15, 62, 1000, -1)
[G, C] = ajoutbranche(G, C, 20, 63, 1000, 1)
[G, C] = ajoutbranche(G, C, 3, 70, 1000, -1)
[G, C] = ajoutbranche(G, C, 18, 75, 1000, 1)
[G, C] = ajoutbranche(G, C, 20, 79, 1000, -1)

keigenvec, keigenval = calculvp(G)
#print(type(keigenvec))
#vp2 = dict(enumerate(np.transpose(keigenvec[:,1])))
vp2 = dict(zip(G.nodes(),np.transpose(keigenvec[:,1])))
#print(vp2)
nx.set_node_attributes(G, vp2, 'vecp2')
#print(G.nodes[1]['valp2'])
#nx.write_graphml(G, "graphetestattributs")


figure = plt.figure(0)
figure.clf()

figure.add_subplot(1,2,1)
nx.drawing.nx_pylab.draw_networkx(G,
                                  ax=figure.gca(),
                                  pos=nx.kamada_kawai_layout(G),
                                  with_labels=False,
                                  node_size=10,
                                  node_color=[G.nodes[i]['vecp2'] for i in G.nodes()],
                                  cmap=plt.get_cmap('plasma'))
branch_relative_valp2 = {}
for i in G.nodes:
    branch_id = i//1000
    branch_nodes = [j for j in G.nodes() if j//1000 == branch_id]

    branch_min_valp2 = np.min([G.nodes[j]['vecp2'] for j in branch_nodes])
    branch_max_valp2 = np.max([G.nodes[j]['vecp2'] for j in branch_nodes])
    branch_relative_valp2[i] = (G.nodes[i]['vecp2']-branch_min_valp2)/(branch_max_valp2-branch_min_valp2)
nx.set_node_attributes(G, branch_relative_valp2, 'branch_relative_vecp2')

figure.add_subplot(1,2,2)
nx.drawing.nx_pylab.draw_networkx(G,
                                  ax=figure.gca(),
                                  pos=nx.kamada_kawai_layout(G),
                                  with_labels=False,
                                  node_size=10,
                                  node_color=[G.nodes[i]['branch_relative_vecp2'] for i in G.nodes()],
                                  cmap=plt.get_cmap('plasma'))

figure = plt.figure(1)
figure.clf()

figure.gca().scatter([i%1000 + i//1000 for i in G.nodes],[G.nodes[i]['vecp2'] for i in G.nodes()],s=10)
figure.gca().set_xlim(0,100)

#ploteigenval(keigenval)
#ploteigenvectopo(keigenvec)
#ploteigenvec(keigenvec)
#exportsurnuage(2, C, keigenvec)