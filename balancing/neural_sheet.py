import numpy as np
import networkx as nx

from scipy.sparse.csgraph import laplacian

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def generate_neural_sheet(grid_size, w0, periodic=True):

    # set up basic grid
    g0 = nx.grid_graph([grid_size, grid_size], periodic=periodic)

    # set up directed graph
    dg = nx.DiGraph()
    dg.add_nodes_from(g0)
    dg.add_weighted_edges_from([(u, v, w0) for (u, v) in g0.edges()])
    dg.add_weighted_edges_from([(v, u, w0) for (u, v) in g0.edges()])    
    return dg

def update_graph_weights(base_graph, J, h=None):
    # updates the edges in base_graph with weights in J
    g = base_graph.copy()
    weights=[]
    for (u, v) in g.edges():
        w = J[list(g.nodes).index(u), list(g.nodes).index(v)]
        weights.append((u,v,w))
    g.add_weighted_edges_from(weights)

    if h is not None:
        for i, n in enumerate(g.nodes):
            g.nodes[n]['h'] = h[i]

    return g


def _select_directed_edge(e, k):
    # returns True if edge matches the pattern specified by key k
    ((xi, yi), (xj, yj)) = e
    if k == 'ne':
        res = \
        ((xi == xj - 1) and yi == yj) or \
        ((xi == xj) and (yi == yj - 1))
    elif k == 'sw':
        res = \
        ((xi == xj + 1) and yi == yj) or \
        ((xi == xj) and (yi == yj + 1))        
    return res

def gen_nesw_edge_partition(gf):
    return {
        k : [e for e in gf.edges() if _select_directed_edge(e, k)]
        for k in ['ne', 'sw']         
        }

def gen_comparative_edge_partition(gf):
    def _select_comparative_edge(e, k):
        u, v = e
        uv_larger = gf.get_edge_data(u,v)['weight'] > \
                    gf.get_edge_data(v,u)['weight']
        if k=='larger':
            return uv_larger
        elif k=='smaller':
            return ~uv_larger
    potentiated_edges = {}
    for k in ['larger', 'smaller']:
        for e in gf.edges():
            potentiated_edges[k] = [
                e for e in gf.edges()
                if _select_comparative_edge(e, k) and \
                (_select_directed_edge(e, 'ne') or \
                    _select_directed_edge(e, 'sw'))
                ]    
    return potentiated_edges


def gen_potentiated_edge_partition(gf, thresh):
    def _select_potentiated_edge(k, v, thresh):
        if k=='potentiated':
            return (v > thresh)
        elif k=='depotentiated':
            return (v < thresh)
    potentiated_edges = {}
    for k in ['potentiated', 'depotentiated']:
        for e in gf.edges():
            potentiated_edges[k] = [
                e for e in gf.edges()
                if _select_potentiated_edge(
                    k,
                    gf.get_edge_data(*e)['weight'],
                    thresh) and \
                (_select_directed_edge(e, 'ne') or \
                    _select_directed_edge(e, 'sw'))
                ]    
    return potentiated_edges
    
def plot_grid(
    dg, edge_partition=None, vmin=None, vmax=None, cmap=None, **plotting_args
    ):    
    # a visualization of the weighted adjacency matrix of graph dg
    # plots the subset of edges specified by edge_direction
    
    grid_size = max([i for (i,j) in dg.nodes])
    weights = [dg.edges[e]['weight'] for e in dg.edges()]
    if edge_partition is None:
        edge_partition = gen_nesw_edge_partition(dg)
    if vmin is None:
        vmin = min(weights)
    if vmax is None:
        vmax = max(weights)
    if cmap is None:
        cmap = 'viridis'

    # set up colormap
    edgemapper = cm.ScalarMappable(
        norm=Normalize(
            vmin=vmin,
            vmax=vmax,
            clip=True),
        cmap=cm.get_cmap(cmap)
        )     

    node_values = []
    for n in dg.nodes:
        if 'h' in dg.nodes[n]:
            v = dg.nodes[n]['h']
        else:
            v = 0.
        node_values.append(v)
    nodemapper = cm.ScalarMappable(
            norm=Normalize(
                vmin=min(node_values),
                vmax=max(node_values),
                clip=True),
            cmap=cm.get_cmap('Greys')
            )
        
    fig, axes = plt.subplots(1, len(edge_partition), **plotting_args)    
    for i, (k, edge_set) in enumerate(edge_partition.items()):
        # select edges to plot        

        edge_colors = [edgemapper.to_rgba(dg.edges[e]['weight'])
                       for e in edge_set]        
        node_colors = [nodemapper.to_rgba(v) for v in node_values]
        
        circles = [Circle((x, y), radius=0.2) for x, y in dg.nodes]

        # create plotting objects
        lc = LineCollection(
            edge_set,
            colors=edge_colors,
            linewidths=4,
            zorder=1,
        )                    
        pc = PatchCollection(circles, zorder=2)
        pc.set_facecolor(node_colors)
        pc.set_edgecolor('grey')

        ax = axes[i]
        ax.add_collection(lc)        
        ax.add_collection(pc)
        ax.plot()
        ax.set_title(k)
        ax.axis('square')
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.axis('off')
        
    plt.show()

def calc_cost_matrix(J, p=2, alpha=1):
    # p is a scalar
    # alpha is anything that will broadcast elementwise multiplication with J
    return alpha * np.abs(J)**p


def task_preserving_transform(J0, h):
    return J0 * np.exp((h[:,None] - h[None,:]))


def calc_eigenmodes(dg, ii, **cost_args):
    # dg is the directed graph
    # C is the synaptic cost matrix at equilibrium    
    J = np.array(nx.adjacency_matrix(dg).todense())
    C = calc_cost_matrix(J, **cost_args)
    Cbar = C + C.T
    L = laplacian(Cbar)    
    evals, evecs = np.linalg.eigh(L)
    dhs, dJs = [], []
    for i in ii:
        dh = evecs[:,i]
        dJ = task_preserving_transform(J, dh) - J
        dhs.append(dh)
        dJs.append(dJ)
    return dhs, dJs
