from treex import *
from treex.simulation import *
from treex.simulation.galton_watson import __discrete_distribution  # only for generating discrete observations


initial_distribution = [0.7, 0.1, 0.1, 0.1]

transition_matrix = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7], [0.2, 0.1, 0.6, 0.1], [0.1, 0.6, 0.2, 0.1]]

continuous_obs = False

parameters = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5]]

def gen_emission(k, parameters):  # Discrete emission
    return __discrete_distribution('', parameters[k])



save_object(mst, 'test_spanning_tree_attributes')
st_tree = read_object('test_spanning_tree_attributes')

rt = 5

t = Tree()
t.add_attribute_to_id('nx_label',rt)

list_of_nodes = [rt]


def increment_spanning_tree(t , list_of_nodes):
    for children in st_tree.neighbors(rt):
        if children not in list_of_nodes:
            s = Tree()
            s.add_attribute_to_id('nx_label',children)
            list_of_nodes.append(children)
            increment_spanning_tree(s , list_of_nodes)
            t.add_subtree(s)


st_tree.nodes
rt = 5
t = Tree()
t.add_attribute_to_id('nx_label',rt)
list_of_nodes = [rt]
def increment_spanning_tree(t , root_label , list_of_nodes):
    for children in st_tree.neighbors(root_label):
        if children not in list_of_nodes:
            s = Tree()
            s.add_attribute_to_id('nx_label', children)
            list_of_nodes.append(children)
            increment_spanning_tree(s, children, list_of_nodes)
            t.add_subtree(s)
increment_spanning_tree(t , rt , list_of_nodes)


dict = t.dict_of_ids()
for children in t.list_of_ids():
    st_node = dict[children]['attributes']['nx_label']
    t.add_attribute_to_id('linearity', st_tree.nodes[st_node]['linearity'], children)