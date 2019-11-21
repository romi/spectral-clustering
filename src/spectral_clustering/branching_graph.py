# Tests sur graphes chaînes

########### Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsp


########### Définitions Fonctions

class BranchingGraph(nx.Graph):
    """
    A graph structure that allows adding branches to a main stem. The resulting
    graph is formed by chains of nodes linked together at branching points.
    Each branch is characterized by an order relatively to the main stem.
    """

    def __init__(self, stem_size=100):
        """Initialize the graph with a main stem.

        The graph is created with a single stem of length stem_size.

        Parameters
        ----------
        stem_size : int
            The number of nodes of the main stem.
        """
        super().__init__(self)

        self.node_coords = None
        self.branch_nodes = {}
        self.branch_linking_node = {}
        self.branch_order = {}

        self.keigenvec = None
        self.keigenval = None

        self.init_stem(stem_size)

    # Création de la tige principale
    # n , Taille de la chaîne principale
    def init_stem(self, stem_size=100):
        """Initialize the main stem of the graph.

        Nodes are added and connected together by edges to form the main stem
        of the graph.

        Parameters
        ----------
        stem_size : int
            The number of nodes of the main stem.

        Returns
        -------
        None
        """

        # Création chaîne via Networkx
        list_nodes = [i for i in range(stem_size)]
        self.add_nodes_from(list_nodes)

        # Stocker les noeuds de la tige
        branch_id = np.max(list(self.branch_nodes.keys()))+1 if len(self.branch_nodes) > 0 else 0
        self.branch_nodes[branch_id] = list_nodes
        self.branch_linking_node[branch_id] = 0
        self.branch_order[branch_id] = 0
        #print(list_nodes)

        # Associer l'id de la tige comme attribut des noeuds
        nx.set_node_attributes(self, dict(zip(list_nodes,[branch_id for _ in list_nodes])), 'branch_id')
        nx.set_node_attributes(self, dict(zip(list_nodes,[self.branch_order[branch_id] for _ in list_nodes])), 'branch_order')

        # Créer tuples reliant un point avec le précédent
        list_edge = [(i, i+1) for i in range(stem_size-1)]
        #print(list_edge)
        self.add_edges_from(list_edge)
        # Export visuel
        # Création de points / coordonnées dans une matrice
        # ici ils sont espacés de 1 à chaque fois.
        self.node_coords = np.asarray([[0, 0, i] for i in range(stem_size)])
        #print(C)
        #print(type(C))
        #print(np.shape(C))
        # return G, C

    # Ajout d'une branche à la tige principale
    # Paramètres en entrée : Tb taille de la branche
    # Eb nom du node sur lequel la branche est ajoutée
    # G graphe de travail.
    # Matrice de coordonnées C tige origine
    # Facteur pour nommer les noeuds F : à modifier si l'on fait deux branches sur le même noeud
    # Choisir F supérieur à 1000 et par *10.
    # Prendre en compte la taille de la tige principale.
    # y_orientation pour voir les deux branches dans deux directions différentes
    # Jouer avec x_offset si on a plus de deux branches au même endroit.
    def add_branch(self, branch_size, linking_node, y_orientation=1, x_offset=0):
        """Create a branch and add it to the graph.

        Parameters
        ----------
        branch_size : int
            The number of nodes in the branch.
        linking_node : int
            The node on which to attach the branch.
        y_orientation : int
            Whether to go left or right on the Y axis (-1 or 1).
        x_offset : float
            The offset on the X axis.

        Returns
        -------
        None
        """
        # Création de la branche et ajout dans le graphe

        starting_node = np.max(list(self.nodes))+1
        list_nodes = [starting_node+i for i in range(branch_size)]
        self.add_nodes_from(list_nodes)

        branch_id = np.max(list(self.branch_nodes.keys()))+1 if len(self.branch_nodes) > 0 else 0
        self.branch_nodes[branch_id] = list_nodes

        linking_branch_id = self.nodes[linking_node]['branch_id']
        linking_branch_order = self.branch_order[linking_branch_id]
        self.branch_linking_node[branch_id] = linking_node
        self.branch_order[branch_id] = linking_branch_order+1

        nx.set_node_attributes(self, dict(zip(list_nodes,[branch_id for _ in list_nodes])), 'branch_id')
        nx.set_node_attributes(self, dict(zip(list_nodes,[self.branch_order[branch_id] for _ in list_nodes])), 'branch_order')

        list_edge = [(starting_node+i, starting_node+i+1) for i in range(branch_size-1)]
        self.add_edges_from(list_edge)

        # On relie la branche au point d'insertion
        self.add_edge(linking_node, starting_node)
        # Ajout de coordonnées pour représentation graphique
        branch_node_coords = np.array([[x_offset, y_orientation*(i+1), self.node_coords[linking_node,2]] for i in range(branch_size)])
        # branch_node_coords = branch_node_coords.reshape(branch_size, 3)
        #print(np.shape(Cbranche))
        self.node_coords = np.concatenate((self.node_coords, branch_node_coords), axis=0)
        # C = Ctot
        #print(np.shape(C))
        # return G, C

    # Calcul des vecteurs propres du graphe
    # G le graphe
    def compute_graph_eigenvectors(self, is_sparse=False, k=50):
        """TODO

        Parameters
        ----------
        is_sparse
        k

        Returns
        -------

        """
        # Appli Laplacien
        L = nx.laplacian_matrix(self, weight='weight')
        L = L.toarray()

        # if isinstance(L, np.ndarray):
        if not is_sparse:
            # Calcul vecteurs propres
            # Utilisation de eigsh impossible lorsque le graphe est petit.
            # eigh calcul tous les vecteurs propres.
            self.keigenval, self.keigenvec = np.linalg.eigh(L)
        else:
            self.keigenval, self.keigenvec = spsp.linalg.eigsh(L, k=k, sigma=0, which='LM')
        #
        # return keigenvec, keigenval

    # Cette fonction génère des fichiers contenant les nuages de points pour chaque vecteur propre.
    # k le nombre de vecteurs propres voulus
    # C la matrice des coordonnées
    # keigenvec la matrice des vecteurs propres

    def add_eigenvector_value_as_attribute(self, k=2, compute_branch_relative=True):
        """TODO

        Parameters
        ----------
        k
        compute_branch_relative

        Returns
        -------

        """
        if self.keigenvec is None:
            self.compute_graph_eigenvectors()

        # print(type(keigenvec))
        # vp2 = dict(enumerate(np.transpose(keigenvec[:,1])))
        node_eigenvector_values = dict(zip(self.nodes(), np.transpose(self.keigenvec[:,k-1])))
        # print(vp2)
        nx.set_node_attributes(self, node_eigenvector_values, 'eigenvector_'+str(k))
        # print(G.nodes[1]['valp2'])
        # nx.write_graphml(G, "graphetestattributs")

        if compute_branch_relative:
            branch_relative_values = {}
            for i in self.nodes:
                branch_id = self.nodes[i]['branch_id']
                branch_nodes = self.branch_nodes[branch_id]

                branch_min_value = np.min([self.nodes[j]['eigenvector_'+str(k)] for j in branch_nodes])
                branch_max_value = np.max([self.nodes[j]['eigenvector_'+str(k)] for j in branch_nodes])
                branch_relative_values[i] = (self.nodes[i]['eigenvector_'+str(k)] - branch_min_value) / (branch_max_value - branch_min_value)
            nx.set_node_attributes(self, branch_relative_values, 'branch_relative_eigenvector_'+str(k))


    def export_eigenvectors_on_pointcloud(self, path=".", k=50):
        """TODO

        Parameters
        ----------
        path
        k

        Returns
        -------

        """
        if self.keigenvec is None:
            self.compute_graph_eigenvectors()

        keigenvec = np.asarray(self.keigenvec[:,:k])
        #print(np.shape(keigenvec))
        #print(np.shape(C))
        # Concaténation avec les labels (vecteurs propres)
        # Faire boucle pour sortir tous les nuages de points associés à chaque vecteur propre.
        for i in range(k):
            # keigenvecK = keigenvec[:, i].reshape(keigenvec.shape[0], 1)
            #print(np.shape(keigenvecK))
            pcdtabclassif = np.concatenate((self.node_coords, keigenvec[:,i][:,np.newaxis]), axis=1)
            # Sortie de tous les nuages
            filename = 'testchain' + str(i)
            print(filename)
            np.savetxt(path + "/" + filename + '.txt', pcdtabclassif, delimiter=",")


def save_eigenval_plot(G, filename="ValeursPropres.png"):
    if G.keigenval is None:
        G.compute_graph_eigenvectors()
    figureval = plt.figure(0)
    figureval.clf()
    figureval.gca().plot(range(len(np.transpose(G.keigenval))),np.transpose(G.keigenval), 'bo')
    figureval.set_size_inches(20,10)
    figureval.subplots_adjust(wspace=0,hspace=0)
    figureval.tight_layout()
    figureval.savefig(filename)


def save_eigenvec_plot(G, sort_values=True, filename="eigenvectors.png"):
    if G.keigenvec is None:
        G.compute_graph_eigenvectors()
    figure = plt.figure(0)
    figure.clf()
    keigenvec = G.keigenvec[:,:50]
    if sort_values:
        keigenvec = keigenvec[keigenvec[:,1].argsort()]
    for i_vec, vec in enumerate(np.transpose(np.around(keigenvec,10))):
        figure.add_subplot(5,10,i_vec+1)
        figure.gca().set_title("Eigenvector "+str(i_vec+1))
        figure.gca().plot(range(len(vec)),vec,color='blue')
    figure.set_size_inches(20,10)
    figure.subplots_adjust(wspace=0,hspace=0)
    figure.tight_layout()
    figure.savefig(filename)


def save_single_eigenvec_plot(G, k=2, sort_values=True, filename=None):
    if G.keigenvec is None:
        G.compute_graph_eigenvectors()
    figure = plt.figure(0)
    figure.clf()
    vec = G.keigenvec[:,k-1]
    branches = np.array([G.nodes[i]['branch_id'] for i in G.nodes])
    if sort_values:
        vec = vec[G.keigenvec[:,1].argsort()]
        branches = branches[G.keigenvec[:,1].argsort()]
    figure.gca().set_title("Eigenvector "+str(k))
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    figure.gca().scatter(range(len(vec)),vec,c=branches,cmap='jet')
    figure.set_size_inches(10,10)
    figure.subplots_adjust(wspace=0,hspace=0)
    figure.tight_layout()
    if filename is None:
        filename = "eigenvector_"+str(k)+".png"
    figure.savefig(filename)


def save_eigenvector_value_along_stem_plot(G, k=2, filename="eigenvector_along_stem.png"):
    if not 'eigenvector_'+str(k) in G.nodes[0]:
        G.add_eigenvector_value_as_attribute(k)

    figure = plt.figure(0)
    figure.clf()

    order_colors = {0: 'darkred', 1: 'darkgoldenrod', 2: 'chartreuse', 3:'darkcyan'}

    node_x = {}
    for branch_id in np.sort(list(G.branch_order.keys())):
        link = G.branch_linking_node[branch_id]
        link_x = node_x[link] if link in node_x else 0
        branch_node_x = [link_x + 1 + (i - np.min(G.branch_nodes[branch_id])) for i in G.branch_nodes[branch_id]]
        node_x.update(dict(zip(G.branch_nodes[branch_id],branch_node_x)))

        branch_node_y = [G.nodes[i]['eigenvector_'+str(k)] for i in G.branch_nodes[branch_id]]

        zero_nodes = np.array(G.branch_nodes[branch_id])[:-1][np.array(branch_node_y)[:-1]*np.array(branch_node_y)[1:]<0]

        for i in zero_nodes:
            figure.gca().scatter(np.mean(branch_node_x[i:i+2]),np.mean(branch_node_y[i:i+2]),color='k')

        branch_order = G.branch_order[branch_id]
        figure.gca().plot(branch_node_x,branch_node_y,color=order_colors[branch_order])


    figure.set_size_inches(20,10)
    figure.subplots_adjust(wspace=0,hspace=0)
    figure.tight_layout()
    figure.savefig(filename)


def save_graph_plot(G, attribute_names=[None], colormap='plasma', node_size=10, attribute_as_size=False, plot_zeros=True, filename="graph.png"):

    figure = plt.figure(0)
    figure.clf()

    graph_layout = nx.kamada_kawai_layout(G)

    for i_a,attribute_name in enumerate(attribute_names):

        figure.add_subplot(1, len(attribute_names), i_a+1)

        if attribute_name is None or attribute_name not in G.nodes[0]:
            node_color = [0 for _ in G.nodes()]
            node_sizes = node_size
        else:
            node_color = [G.nodes[i][attribute_name] for i in G.nodes()]
            if attribute_as_size:
                node_sizes = [node_size*(np.abs(G.nodes[i][attribute_name])/np.max([np.abs(G.nodes[j][attribute_name]) for j in G.nodes()])) for i in G.nodes()]
            else:
                node_sizes = node_size

        nx.drawing.nx_pylab.draw_networkx(G,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=False,
                                          node_size=node_sizes,
                                          node_color=node_color,
                                          cmap=plt.get_cmap(colormap))

        if attribute_name.startswith('eigenvector_') and plot_zeros:
            zero_edges = np.array(G.edges())[np.prod([[G.nodes[i][attribute_name] for i in e] for e in np.array(G.edges)], axis=1) < 0]

            for e in zero_edges:
                edge_points = np.array([graph_layout[i] for i in e])
                figure.gca().scatter(np.mean(edge_points[:,0]),np.mean(edge_points[:,1]),color='k')

        figure.gca().set_title(attribute_name,size=24)

    figure.set_size_inches(10*len(attribute_names),10)
    figure.subplots_adjust(wspace=0,hspace=0)
    figure.tight_layout()
    figure.savefig(filename)



# def Convert(lst):
#     res_dct = {i : v for i,v in enumerate(lst)}
#     return res_dct

########## Corps du programme : tests

if __name__ == '__main__':

    G = BranchingGraph(100)
    G.add_branch(branch_size=20,linking_node=25)
    G.add_branch(branch_size=20,linking_node=110)
    # G.add_branch(branch_size=100,linking_node=0)
    # G.add_branch(branch_size=10,linking_node=50)
    G.add_branch(branch_size=20,linking_node=60)
    G.add_branch(branch_size=15,linking_node=150)
    G.add_branch(branch_size=5,linking_node=165)
    G.add_branch(branch_size=10,linking_node=60,y_orientation=-1)
    G.add_branch(branch_size=25,linking_node=35)
    # G.add_branch(branch_size=15,linking_node=40,y_orientation=-1)
    # G.add_branch(branch_size=8,linking_node=47)
    # G.add_branch(branch_size=20,linking_node=50,y_orientation=-1)
    G.add_branch(branch_size=25,linking_node=55)
    # G.add_branch(branch_size=15,linking_node=62,y_orientation=-1)
    # G.add_branch(branch_size=20,linking_node=63)
    G.add_branch(branch_size=3,linking_node=70,y_orientation=-1)
    # G.add_branch(branch_size=18,linking_node=75)
    G.add_branch(branch_size=20,linking_node=79,y_orientation=-1)

    G.compute_graph_eigenvectors()
    G.add_eigenvector_value_as_attribute(2)
    G.add_eigenvector_value_as_attribute(3)
    save_graph_plot(G, attribute_names=['eigenvector_2', 'branch_relative_eigenvector_2', 'eigenvector_3'], filename='graph_first_eigenvectors.png')

    for k in range (11):
        G.add_eigenvector_value_as_attribute(len(G)-k)
    save_graph_plot(G,attribute_names=['eigenvector_'+str(len(G)-k) for k in range(11)],plot_zeros=False,attribute_as_size=True,node_size=50,filename='graph_last_eigenvectors.png')

    save_eigenvector_value_along_stem_plot(G,k=2)

    save_eigenval_plot(G)
    save_single_eigenvec_plot(G,len(G))
    save_eigenvec_plot(G)
    save_eigenvec_plot(G,sort_values=False,filename="eigenvectorstopo.png")
    G.export_eigenvectors_on_pointcloud(k=2)
