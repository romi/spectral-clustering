from treex import *
from treex.simulation import *
from treex.simulation.galton_watson import __discrete_distribution  # only for generating discrete observations


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
    list_of_nodes=[root_label]
    increment_spanning_tree(st_tree, root_label, t, list_of_nodes, list_att)
    return t


def add_attributes_to_spanning_tree(t, list_att=['planarity', 'linearity', 'scattering']):
    dict = t.dict_of_ids()
    for node in t.list_of_ids():
        st_node = dict[node]['attributes']['nx_label']
        for att in list_att:
            t.add_attribute_to_id(att, st_tree.nodes[st_node][att], node)


st_tree = read_pointcloudgraph_into_treex(pointcloudgraph=QG_t)
rt = 40
t = build_spanning_tree(st_tree, rt)