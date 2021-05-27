import networkx as nx
import numpy as np
import open3d as open3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from visu_core.matplotlib import glasbey

def export_quotient_graph_attribute_on_point_cloud(QG, attribute):
    labels_from_qg = np.zeros((len(QG.point_cloud_graph), 4))
    i = 0
    G = QG.point_cloud_graph
    for n in G.nodes:
        labels_from_qg[i, 0:3] = G.nodes[n]['pos']
        labels_from_qg[i, 3] = QG.nodes[G.nodes[n]['quotient_graph_node']][attribute]
        i += 1
    np.savetxt('pcd_' + attribute + '.txt', labels_from_qg, delimiter=",")


def export_clustering_labels_on_point_cloud(G, filename="pcd_clustered.txt"):
    pcd_clusters = np.concatenate([G.nodes_coords, G.clustering_labels], axis=1)
    np.savetxt(filename, pcd_clusters, delimiter=",")
    print("Export du nuage avec les labels de cluster")


def export_anything_on_point_cloud(G, attribute, filename="pcd_attribute.txt"):
    if attribute.ndim == 1:
        attribute = attribute[:,np.newaxis]
    pcd_attribute = np.concatenate([G.nodes_coords, attribute], axis=1)
    np.savetxt(filename, pcd_attribute, delimiter=",")
    print("Export du nuage avec les attributs demandés")


def export_gradient_of_fiedler_vector_on_pointcloud(G, filename="pcd_vp2_grad.txt"):
    pcd_vp2_grad = np.concatenate([G.nodes_coords, G.gradient_on_fiedler], axis=1)
    np.savetxt(filename, pcd_vp2_grad, delimiter=",")
    print("Export du nuage avec gradient du vecteur propre 2")


def export_fiedler_vector_on_pointcloud(G, filename="pcd_vp2.txt"):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    np.savetxt(filename, pcd_vp2, delimiter=",")
    print("Export du nuage avec le vecteur propre 2")


def export_some_graph_attributes_on_point_cloud(pointcloudgraph, graph_attribute='quotient_graph_node', filename='graph_attribute.txt'):
    G = pointcloudgraph
    new_classif = np.asarray(list((dict(G.nodes(data=graph_attribute)).values())))
    new_classif = new_classif[:, np.newaxis]
    export_anything_on_point_cloud(G, attribute=new_classif, filename=filename)


def display_and_export_graph_of_fiedler_vector(G, filename="fiedler_vector", sorted_by_fiedler_vector=True):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_sort_by_vp2 = pcd_vp2[pcd_vp2[:, 3].argsort()]
    figure = plt.figure(0)
    figure.clf()
    figure.gca().set_title("fiedler vector")
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    if sorted_by_fiedler_vector:
        figure.gca().scatter(range(len(pcd_vp2_sort_by_vp2)), pcd_vp2_sort_by_vp2[:, 3], color='blue')
    if sorted_by_fiedler_vector is False:
        figure.gca().scatter(range(len(pcd_vp2)), pcd_vp2[:, 3], color='blue')
    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du vecteur propre 2")


def display_and_export_graph_of_gradient_of_fiedler_vector(G, filename="Gradient_of_fiedler_vector", sorted_by_fiedler_vector=True, sorted_by_gradient=False):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_grad_vp2 = np.concatenate([pcd_vp2, G.gradient_on_fiedler], axis=1)
    pcd_vp2_grad_vp2_sort_by_vp2 = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 3].argsort()]
    pcd_vp2_grad_vp2_sort_by_grad = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 4].argsort()]

    figure = plt.figure(1)
    figure.clf()
    figure.gca().set_title("Gradient of fiedler vector")
    plt.autoscale(enable=True, axis='both', tight=None)
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    if sorted_by_fiedler_vector and sorted_by_gradient is False:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_vp2[:, 4], color='blue')
    if sorted_by_fiedler_vector is False and sorted_by_gradient is False:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2)), pcd_vp2_grad_vp2[:, 4], color='blue')

    if sorted_by_gradient:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_grad[:, 4], color='blue')

    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du gradient")


def display_and_export_graph_of_fiedler_vector(G, filename="fiedler_vector", sorted_by_fiedler_vector=True):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_sort_by_vp2 = pcd_vp2[pcd_vp2[:, 3].argsort()]
    figure = plt.figure(0)
    figure.clf()
    figure.gca().set_title("fiedler vector")
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    if sorted_by_fiedler_vector:
        figure.gca().scatter(range(len(pcd_vp2_sort_by_vp2)), pcd_vp2_sort_by_vp2[:, 3], color='blue')
    if sorted_by_fiedler_vector is False:
        figure.gca().scatter(range(len(pcd_vp2)), pcd_vp2[:, 3], color='blue')
    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du vecteur propre 2")


def display_and_export_graph_of_gradient_of_fiedler_vector(G, filename="Gradient_of_fiedler_vector", sorted_by_fiedler_vector=True, sorted_by_gradient=False):
    vp2 = G.keigenvec[:, 1]
    pcd_vp2 = np.concatenate([G.nodes_coords, vp2[:, np.newaxis]], axis=1)
    pcd_vp2_grad_vp2 = np.concatenate([pcd_vp2, G.gradient_on_fiedler], axis=1)
    pcd_vp2_grad_vp2_sort_by_vp2 = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 3].argsort()]
    pcd_vp2_grad_vp2_sort_by_grad = pcd_vp2_grad_vp2[pcd_vp2_grad_vp2[:, 4].argsort()]

    figure = plt.figure(1)
    figure.clf()
    figure.gca().set_title("Gradient of fiedler vector")
    plt.autoscale(enable=True, axis='both', tight=None)
    # figure.gca().plot(range(len(vec)),vec,color='blue')
    if sorted_by_fiedler_vector and sorted_by_gradient is False:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_vp2[:, 4], color='blue')
    if sorted_by_fiedler_vector is False and sorted_by_gradient is False:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2)), pcd_vp2_grad_vp2[:, 4], color='blue')

    if sorted_by_gradient:
        figure.gca().scatter(range(len(pcd_vp2_grad_vp2_sort_by_vp2)), pcd_vp2_grad_vp2_sort_by_grad[:, 4], color='blue')

    figure.set_size_inches(10, 10)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    figure.savefig(filename)
    print("Export du gradient")


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


def display_gradient_vector_field(G, normalized=True, scale= 1., filename="gradient_vectorfield_3d.png"):
    from cellcomplex.property_topomesh.creation import vertex_topomesh
    from cellcomplex.property_topomesh.visualization.vtk_actor_topomesh import VtkActorTopomesh
    from visu_core.vtk.display import vtk_display_actors, vtk_save_screenshot_actors

    n_points = G.nodes_coords.shape[0]

    topomesh = vertex_topomesh(dict(zip(range(n_points), G.nodes_coords)))

    if normalized:
        vectors = G.direction_gradient_on_fiedler_scaled
    if normalized is False:
        vectors = G.gradient_on_fiedler * G.direction_gradient_on_fiedler_scaled

    topomesh.update_wisp_property('vector', 0, dict(zip(range(n_points), vectors)))

    actors = []

    vector_actor = VtkActorTopomesh(topomesh, degree=0, property_name='vector')
    vector_actor.vector_glyph = 'arrow'
    vector_actor.glyph_scale = scale
    vector_actor.update(colormap='Reds', value_range=(0,0))
    actors += [vector_actor.actor]

    # Change of background
    ren, _, _ = vtk_display_actors(actors, background=(0.9, 0.9, 0.9))
    cam = ren.GetActiveCamera()
    vtk_save_screenshot_actors(actors, image_filename=filename, camera=cam)


#3d
def draw_quotientgraph_cellcomplex(pcd, QG, G, color_attribute='quotient_graph_node', filename="graph_and_quotientgraph_3d.png"):

    if type(pcd) is open3d.geometry.PointCloud:
        pcdtab = np.asarray(pcd.points)
    else:
        pcdtab = pcd
    # s, t = np.meshgrid(np.arange(len(pcdtab)), np.arange(len(pcdtab)))
    # sources = s[simatrix > 0]
    # targets = t[simatrix > 0]
    # sources, targets = sources[sources < targets], targets[sources < targets]

    from cellcomplex.property_topomesh.creation import vertex_topomesh, edge_topomesh
    from cellcomplex.property_topomesh.analysis import compute_topomesh_property
    from cellcomplex.property_topomesh.visualization.vtk_actor_topomesh import VtkActorTopomesh
    from visu_core.vtk.display import vtk_display_actors, vtk_save_screenshot_actors

    topomesh = edge_topomesh(np.array([e for e in QG.edges if e[0]!=e[1]]), dict(zip(np.asarray([n for n in QG.nodes]), pcdtab)))

    if color_attribute == 'quotient_graph_node':
        topomesh.update_wisp_property('quotient_graph_node', 0, dict(zip([n for n in QG.nodes], [n for n in QG.nodes])))
    else:
        topomesh.update_wisp_property(color_attribute, 0,
                                     dict(zip([n for n in QG.nodes], [QG.nodes[n][color_attribute] for n in QG.nodes])))

    compute_topomesh_property(topomesh, 'length', 1)

    actors = []

    edge_actor = VtkActorTopomesh()
    edge_actor.set_topomesh(topomesh, 1, property_name='length')
    edge_actor.line_glyph = 'tube'
    edge_actor.glyph_scale = 0.33
    edge_actor.update(colormap="gray")
    actors += [edge_actor]

    vertex_actor = VtkActorTopomesh(topomesh, 0, property_name=color_attribute)
    # vertex_actor.point_glyph = 'point'
    vertex_actor.point_glyph = 'sphere'
    vertex_actor.glyph_scale = 2
    vertex_actor.update(colormap="jet")
    actors += [vertex_actor]

    graph_topomesh = vertex_topomesh(dict(zip([n for n in G.nodes], [G.nodes[n]['pos'] for n in G.nodes])))
    graph_topomesh.update_wisp_property(color_attribute, 0, dict(
        zip([n for n in G.nodes], [G.nodes[n][color_attribute] for n in G.nodes])))
    point_cloud_actor = VtkActorTopomesh(graph_topomesh, 0, property_name=color_attribute)
    point_cloud_actor.point_glyph = 'point'
    point_cloud_actor.update(colormap="jet")
    actors += [point_cloud_actor]

    ren, _, _ = vtk_display_actors(actors, background=(0.9, 0.9, 0.9))
    cam = ren.GetActiveCamera()
    vtk_save_screenshot_actors(actors, image_filename=filename, camera=cam)


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
