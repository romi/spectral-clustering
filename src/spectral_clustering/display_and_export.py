import spectral_clustering.PointCloudGraph as kpcg
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import open3d as open3d
from mpl_toolkits.mplot3d import Axes3D


def export_some_graph_attributes_on_point_cloud(pointcloudgraph, graph_attribute='quotient_graph_node', filename='graph_attribute.txt'):
    G = pointcloudgraph
    new_classif = np.asarray(list((dict(G.nodes(data=graph_attribute)).values())))
    new_classif = new_classif[:, np.newaxis]
    kpcg.export_anything_on_point_cloud(G, attribute=new_classif, filename=filename)


def display_and_export_quotient_graph_matplotlib(quotient_graph, node_sizes=20, filename="quotient_graph_matplotlib", data_on_nodes='intra_class_node_number', data=True, attributekmeans4clusters = False):

    figure = plt.figure(0)
    figure.clf()
    graph_layout = nx.kamada_kawai_layout(quotient_graph)
    colormap = 'jet'

    if attributekmeans4clusters and data:
        labels_from_attributes = dict(quotient_graph.nodes(data=data_on_nodes))
        # Rounding the data to allow an easy display
        for dict_value in labels_from_attributes:
            labels_from_attributes[dict_value] = round(labels_from_attributes[dict_value], 2)
        node_color_from_attribute = dict(quotient_graph.nodes(data='seed_colors')).values()
        node_color = [quotient_graph.nodes[i]['kmeans_labels'] / 4 for i in quotient_graph.nodes()]
        nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=True,
                                          node_size=node_sizes,
                                          node_color=node_color_from_attribute,
                                          labels=labels_from_attributes,
                                          cmap=plt.get_cmap(colormap))

    elif attributekmeans4clusters is False and data is False:
        nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=False,
                                          node_size=node_sizes,
                                          node_color="r",
                                          cmap=plt.get_cmap(colormap))

    elif attributekmeans4clusters is False and data:
        labels_from_attributes = dict(quotient_graph.nodes(data=data_on_nodes))
        # Rounding the data to allow an easy display
        for dict_value in labels_from_attributes:
            if data_on_nodes != 'semantic_label':
                labels_from_attributes[dict_value] = round(labels_from_attributes[dict_value], 2)
        nx.drawing.nx_pylab.draw_networkx(quotient_graph,
                                          ax=figure.gca(),
                                          pos=graph_layout,
                                          with_labels=True,
                                          node_size=node_sizes,
                                          node_color="r",
                                          labels=labels_from_attributes,
                                          cmap=plt.get_cmap(colormap))


    #nx.drawing.nx_pylab.draw_networkx_edge_labels(quotient_graph, pos=graph_layout, font_size=20, font_family="sans-sherif")

    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du graphe quotient matplotlib")


#3d
def draw_quotientgraph_cellcomplex(pcd, QG, G, color_attribute='quotient_graph_node'):

    if type(pcd) is open3d.geometry.PointCloud:
        pcdtab = np.asarray(pcd.points)
    else:
        pcdtab = pcd
    # s, t = np.meshgrid(np.arange(len(pcdtab)), np.arange(len(pcdtab)))
    # sources = s[simatrix > 0]
    # targets = t[simatrix > 0]
    # sources, targets = sources[sources < targets], targets[sources < targets]

    from cellcomplex.property_topomesh.creation import edge_topomesh
    from cellcomplex.property_topomesh.visualization.vtk_actor_topomesh import VtkActorTopomesh
    from cellcomplex.property_topomesh.visualization.vtk_tools import vtk_display_actors
    from cellcomplex.property_topomesh.analysis import compute_topomesh_property
    from cellcomplex.property_topomesh.creation import vertex_topomesh

    topomesh = edge_topomesh(np.array([e for e in QG.edges if e[0]!=e[1]]), dict(zip(np.asarray([n for n in QG.nodes]), pcdtab)))

    if color_attribute == 'quotient_graph_node':
        topomesh.update_wisp_property('quotient_graph_node', 0, dict(zip([n for n in QG.nodes], [n for n in QG.nodes])))
    else:
        topomesh.update_wisp_property(color_attribute, 0,
                                     dict(zip([n for n in QG.nodes], [QG.nodes[n][color_attribute] for n in QG.nodes])))

    compute_topomesh_property(topomesh, 'length', 1)

    edge_actor = VtkActorTopomesh()
    edge_actor.set_topomesh(topomesh, 1, property_name='length')
    edge_actor.line_glyph = 'tube'
    edge_actor.glyph_scale = 0.33
    edge_actor.update(colormap="gray")

    vertex_actor = VtkActorTopomesh(topomesh, 0, property_name=color_attribute)
    # vertex_actor.point_glyph = 'point'
    vertex_actor.point_glyph = 'sphere'
    vertex_actor.glyph_scale = 2
    vertex_actor.update(colormap="jet")

    graph_topomesh = vertex_topomesh(dict(zip([n for n in G.nodes], [G.nodes[n]['pos'] for n in G.nodes])))
    graph_topomesh.update_wisp_property(color_attribute, 0, dict(
        zip([n for n in G.nodes], [G.nodes[n][color_attribute] for n in G.nodes])))
    point_cloud_actor = VtkActorTopomesh(graph_topomesh, 0, property_name=color_attribute)
    point_cloud_actor.point_glyph = 'point'
    point_cloud_actor.update(colormap="jet")

    vtk_display_actors([vertex_actor.actor, edge_actor.actor, point_cloud_actor.actor], background=(0.9, 0.9, 0.9))



def draw_quotientgraph_matplotlib_3D(nodes_coords_moy, QG):
    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        for i in range(nodes_coords_moy.shape[0]):
            xi = nodes_coords_moy[i, 0]
            yi = nodes_coords_moy[i, 1]
            zi = nodes_coords_moy[i, 2]

            ax.scatter(xi, yi, zi, s=10**2, edgecolors='k', alpha=0.7)

        for i, j in enumerate(QG.edges()):
            corresp = dict(zip(QG.nodes, range(len(QG.nodes))))
            x = np.array((nodes_coords_moy[corresp[j[0]], 0], nodes_coords_moy[corresp[j[1]], 0]))
            y = np.array((nodes_coords_moy[corresp[j[0]], 1], nodes_coords_moy[corresp[j[1]], 1]))
            z = np.array((nodes_coords_moy[corresp[j[0]], 2], nodes_coords_moy[corresp[j[1]], 2]))
            ax.plot(x, y, z, c='black', alpha=0.5)
        ax.view_init(elev=30)
        ax.set_axis_off()
        plt.show()
