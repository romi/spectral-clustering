from treex import *
from treex.simulation import *
from treex.simulation.galton_watson import __discrete_distribution  # only for generating discrete observations
from spectral_clustering.Original_TreeX import *
from spectral_clustering.display_and_export import *
import networkx as nx

def read_pointcloudgraph_into_treex(pointcloudgraph):
    mst=pointcloudgraph
    save_object(mst, 'test_spanning_tree_attributes.p')
    st_tree = read_object('test_spanning_tree_attributes.p')
    return st_tree

def increment_spanning_tree(st_tree, root, t , list_of_nodes, list_att):
    for neighbor in st_tree.neighbors(root):
        if neighbor not in list_of_nodes:
            s = Tree()
            s.add_attribute_to_id('nx_label', neighbor)
            for att in list_att:
                s.add_attribute_to_id(att, st_tree.nodes[neighbor][att])
            list_of_nodes.append(neighbor)
            increment_spanning_tree(st_tree, neighbor, s, list_of_nodes, list_att)
            t.add_subtree(s)


def build_spanning_tree(st_tree, root_label, list_att=['planarity', 'linearity', 'scattering']):
    t = Tree()
    t.add_attribute_to_id('nx_label', root_label)
    for att in list_att:
        t.add_attribute_to_id(att, st_tree.nodes[root_label][att])
    list_of_nodes = [root_label]
    increment_spanning_tree(st_tree, root_label, t, list_of_nodes, list_att)
    return t


def add_attributes_to_spanning_tree(st_tree, t, list_att=['planarity', 'linearity', 'scattering']):
    dict = t.dict_of_ids()
    for node in t.list_of_ids():
        st_node = dict[node]['attributes']['nx_label']
        for att in list_att:
            t.add_attribute_to_id(att, st_tree.nodes[st_node][att], node)


def add_viterbi_results_to_quotient_graph(quotientgraph, t, list_semantics=['leaf', 'stem', 'NSP']):
    dict = t.dict_of_ids()
    for n in t.list_of_ids():
        qg_node = dict[n]['attributes']['nx_label']
        quotientgraph.nodes[qg_node]['viterbi_class'] = dict[n]['attributes']['viterbi_type']


def create_observation_list(t, list_obs=['planarity', 'linearity', 'scattering'], name='observations'):
    dict = t.dict_of_ids()
    for node in t.list_of_ids():
        obs = []
        for att in list_obs:
            obs.append(dict[node]['attributes'][att])
        t.add_attribute_to_id(name, obs, node)

######### Main

if __name__ == '__main__':
    #################################### toy example
    QG_toy = nx.Graph()
    mutige = 0.3
    sigmatige = 0.1
    mupet = 0.8
    sigmapet = 0.2
    muf = 0.8
    sigmaf = 0.3
    QG_toy.add_nodes_from([
        (1, {"observation": random.gauss(mutige, sigmatige)}),
        (2, {"observation": random.gauss(mupet, sigmapet)}),
        (3, {"observation": random.gauss(muf, sigmaf)}),
        (4, {"observation": random.gauss(mupet, sigmapet)}),
        (5, {"observation": random.gauss(muf, sigmaf)}),
        (6, {"observation": random.gauss(mupet, sigmapet)}),
        (7, {"observation": random.gauss(muf, sigmaf)}),
        (8, {"observation": random.gauss(mupet, sigmapet)}),
        (9, {"observation": random.gauss(muf, sigmaf)})
    ])

    QG_toy.add_edges_from([(1, 2), (2, 3), (1, 4), (4, 5), (1, 6), (6, 7), (1, 8), (8, 9)])
    #################################

    st_tree = read_pointcloudgraph_into_treex(pointcloudgraph=QG_t2)
    rt = 8
    #t = build_spanning_tree(st_tree, rt, list_att=['observation'])

    t = build_spanning_tree(st_tree, rt, list_att=['planarity2', 'linearity', 'intra_class_node_number'])

    #def create_observation_list(t, list_obs=['planarity2', 'linearity']):
    #   dict = t.dict_of_ids()
    #    for node in t.list_of_ids():
    #        obs = []
    #        for att in list_obs:
    #            obs.append(dict[node]['attributes'][att])
    #        t.add_attribute_to_id('observations', obs, node)


    create_observation_list(t)



    #########################################################################

    initial_distribution = [1, 0]
    #initial_distribution = [1, 0, 0]
    transition_matrix = [[0.2, 0.8], [0, 1]]
    #transition_matrix = [[0.2, 0, 0.8], [0, 0.8, 0.2], [0, 0.8, 0.2]]
    #transition_matrix = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
    # insertion bruit
    #transition_matrix = [[0.2, 0.7, 0.1], [0, 0.9, 0.1], [0.3, 0.3, 0.4]]
    continuous_obs = True

    if continuous_obs:  # observations are Gaussian
        parameterstot = [[[0.4, 0.4], [0.8, 0.2]], [[0.8, 0.3], [0.4, 0.2]]]
        #parameterstot = [[[0.3, 0.2], [0.8, 0.2], [0.0, 0.05]], [[0.8, 0.2], [0.3, 0.2], [0.1, 0.2]], [[0.30, 0.30], [0.8, 0.2], [0.20, 0.1]]]
        #parameterstot = [[[0.73, 0.1], [0.25, 0.1], [0.03, 0.05]], [[0.66, 0.04], [0.33, 0.05], [0.0001, 0.1]],
        #                 [[0.10, 0.20], [0.6, 0.35], [0.20, 0.1]]]
        #parameterstot = [[[0.8, 0.6], [0.2, 0.3], [0.8, 0.2]]]
        # insertion bruit
        #parameterstot = [[[0.2, 0.2], [0.8, 0.2], [1000, 10000]], [[0.6, 0.2], [0.3, 0.2], [1000, 10000]], [[0.5, 1], [0.5, 1], [110, 100]]]
        def gen_emission(k, parameters):  # Gaussian emission
            return random.gauss(parameters[k][0], parameters[k][1])

        def pdf_emission_dim1(x, moy, sd):  # Gaussian emission
            return 1.0 / (sd * math.sqrt(2 * math.pi)) * math.exp(
                -1.0 / (2 * sd ** 2) * (x - moy) ** 2)
        def pdf_emission_dimn(x, k, parameterstot):
            p = 1
            if len(parameterstot) == 1:
                p = pdf_emission_dim1(x, moy=parameterstot[0][k][0], sd=parameterstot[0][k][1])
                return p
            else:
                for i in range(len(parameterstot[0])):
                    p *= pdf_emission_dim1(x[i], moy=parameterstot[k][i][0], sd=parameterstot[k][i][1])
                return p

    viterbi(t, 'observations', initial_distribution, transition_matrix, pdf_emission_dimn, parameterstot)


    add_viterbi_results_to_quotient_graph(QG_t2, t, list_semantics=['leaf', 'stem', 'NSP'])
    add_viterbi_results_to_quotient_graph(QG, t, list_semantics=['leaf', 'stem', 'NSP'])
    #display_and_export_quotient_graph_matplotlib(quotient_graph=QG_t, node_sizes=20, filename="quotient_graph_observation", data_on_nodes='observations', data=True, attributekmeans4clusters = False)
    display_and_export_quotient_graph_matplotlib(quotient_graph=QG_t2, node_sizes=20, filename="quotient_graph_viterbi", data_on_nodes='viterbi_class', data=True, attributekmeans4clusters = False)

    export_quotient_graph_attribute_on_point_cloud(QG, attribute = 'viterbi_class')

    #export_quotient_graph_attribute_on_point_cloud(QG, attribute='linearity')

