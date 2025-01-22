import polars as pl
import networkx as nx
import numpy as np
from networkx_function import generate_shortest_path_length_matrix
import graphblas as gb

def generate_jacobian_matrix(nx_tree_grid: nx.DiGraph, line_data: pl.DataFrame, slack_node_id: int)-> gb.Matrix: # type: ignore


    # remove slack node from list of nodes
    nodes = np.array(nx_tree_grid.nodes())
    nodes = nodes[nodes != slack_node_id]

    n_nodes = nx_tree_grid.number_of_nodes()
    n_edges = n_nodes-1

    # Matrix values represent the sum of longitudinal resistances (or reactance) of edges in the path connecting the slack node to the 
    # lowest common ancestor of the column index node and row index node.
    coords, ancestor = list(zip(*nx.all_pairs_lowest_common_ancestor(nx_tree_grid)))
    x, y = list(zip(*coords))

    r_mapping = nx.shortest_path_length(nx_tree_grid, source=0, weight="r_pu")
    r_val = list(map(lambda x: -2*r_mapping[x], ancestor))

    h_Vnode_Pload = gb.Matrix.from_coo(x, y, r_val, nrows=n_nodes, ncols=n_nodes).select("!=", 0)[nodes, nodes] # type: ignore
    h_Vnode_Pload = gb.select.offdiag(h_Vnode_Pload).T + h_Vnode_Pload # type: ignore

    x_mapping = nx.shortest_path_length(nx_tree_grid, source=0, weight="x_pu")
    x_val = list(map(lambda x: -2*x_mapping[x], ancestor))

    h_Vnode_Qload = gb.Matrix.from_coo(x, y, x_val, nrows=n_nodes, ncols=n_nodes).select("!=", 0)[nodes, nodes] # type: ignore
    h_Vnode_Qload = gb.select.offdiag(h_Vnode_Qload).T + h_Vnode_Qload # type: ignore

    # Matrix value is one if column index node is downstream row index node
    h_Pflow_Pload = generate_shortest_path_length_matrix(nx_grid = nx_tree_grid, forced_weight=1)[nodes, nodes]
    h_Pflow_Pload = gb.ss.concat(tiles=[[h_Pflow_Pload], [h_Pflow_Pload]]) # type: ignore

    h_Qflow_Qload = h_Pflow_Pload.dup()

    h_Pflow_Qload = gb.Matrix(float, nrows=2*n_edges, ncols=n_edges) # type: ignore
    h_Qflow_Pload = gb.Matrix(float, nrows=2*n_edges, ncols=n_edges) # type: ignore

    h_Sload = gb.Matrix.from_coo(range(2*n_edges), range(2*n_edges), [1.0]*2*n_edges, nrows=2*n_edges, ncols=2*n_edges +1) # type: ignore

    h_Pflow_V0 = gb.Matrix(float, nrows=2*n_edges, ncols=1) # type: ignore
    h_Qflow_V0 = gb.Matrix(float, nrows=2*n_edges, ncols=1) # type: ignore
    h_Qflow_V0[:n_edges, 0] = line_data["b_pu"] # type: ignore
    h_Qflow_V0[n_edges:, 0] = -line_data["b_pu"]

    h_Vnode_V0 = gb.Matrix.from_coo(range(n_edges), [0]*n_edges, [1.0]*n_edges, nrows=n_edges, ncols=1) # type: ignore

    h = gb.ss.concat(tiles=[ # type: ignore
        [h_Pflow_Pload, h_Pflow_Qload, h_Pflow_V0],
        [h_Qflow_Pload, h_Qflow_Qload, h_Qflow_V0],
        [h_Vnode_Pload, h_Vnode_Qload, h_Vnode_V0]
    ])

    h = gb.ss.concat(tiles=[[h_Sload], [h]]) # type: ignore
    return h