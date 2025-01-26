import polars as pl
import networkx as nx
import numpy as np
import graphblas as gb

from networkx_function import (
    generate_shortest_path_length_matrix, generate_bfs_tree_with_edge_data, generate_nx_edge)
from general_function import generate_log


log = generate_log(name=__name__)

def generate_full_jacobian_matrix(grid_edge_data: pl.DataFrame, slack_node_id: int)-> np.array: # type: ignore
    """

    Parameters
    ----------
    grid_edge_data (pl.DataFrame)
    slack_node_id (int)
    

    Returns
    -------
    np.array
    """
    list_col = set(list('u_of_edge', 'v_of_edge', 'r_pu', 'x_pu', 'b_pu', 'g_pu', 'n_transfo', 'type'))
    # Create nx_tree from line data
    nx_grid: nx.Graph = nx.Graph()
    _ = grid_edge_data\
    .with_columns(
        pl.struct(pl.all()).pipe(generate_nx_edge, nx_graph=nx_grid)
    )
    if not nx.is_tree(nx_grid):
        log.error("The graph is not a tree")
        return gb.Matrix() # type: ignore
    else:
        nx_tree_grid : nx.DiGraph = generate_bfs_tree_with_edge_data(nx_grid, slack_node_id)
        
    # remove slack node from list of nodes
    nodes = np.array(nx_tree_grid.nodes())
    nodes = nodes[nodes != slack_node_id]

    n_nodes = nx_tree_grid.number_of_nodes()
    n_edges = nx_tree_grid.number_of_edges()
    
    descendent_matrix = generate_shortest_path_length_matrix(nx_grid = nx_tree_grid, forced_weight=1)
    
    # Create edge data graphblas vector
    g_pu = gb.Vector.from_dense(grid_edge_data["g_pu"]) # type: ignore
    b_pu = gb.Vector.from_dense(grid_edge_data["b_pu"]) # type: ignore
    n_transfo = gb.Vector.from_dense(grid_edge_data["n_transfo"]) # type: ignore

    # Matrix values represent the sum of longitudinal resistances (or reactance) of edges in the path connecting the slack node to the 
    # lowest common ancestor of i ang j (row and column index node).
    coords, ancestor = list(zip(*nx.all_pairs_lowest_common_ancestor(nx_tree_grid)))
    x, y = list(zip(*coords))

    r_mapping = nx.shortest_path_length(nx_tree_grid, source=slack_node_id, weight="r_pu")
    r_val = list(map(lambda x: -2*r_mapping[x], ancestor))

    vn_pload_gb = gb.Matrix.from_coo(x, y, r_val, nrows=n_nodes, ncols=n_nodes).select("!=", 0)[nodes, nodes] # type: ignore
    vn_pload = (
        gb.select.offdiag(vn_pload_gb).T # type: ignore
        .ewise_add(vn_pload_gb) # type: ignore
        .to_dense(fill_value=0.0) # type: ignore
    )

    x_mapping = nx.shortest_path_length(nx_tree_grid, source=slack_node_id, weight="x_pu")
    x_val = list(map(lambda x: -2*x_mapping[x], ancestor))

    vn_qload_gb = gb.Matrix.from_coo(x, y, x_val, nrows=n_nodes, ncols=n_nodes).select("!=", 0)[nodes, nodes] # type: ignore
    vn_qload = (
        gb.select.offdiag(vn_qload_gb).T # type: ignore
        .ewise_add(vn_qload_gb) # type: ignore
        .to_dense(fill_value=0.0)
    )
    
    # Matrix value is 1 if j (column index node) is downstream i (row index node) 0 otherwise
    pflow_pload = descendent_matrix[nodes, nodes].to_dense(fill_value=0.0)
    qflow_qload = descendent_matrix[nodes, nodes].to_dense(fill_value=0.0)
    
    #TODO Check if we add or not half of upstream node
    # Matrix value is the sum of transverse susceptance (or conductance) of downstream edge (branch or transformer)
    pflow_v0 = (
        gb.select.offdiag(descendent_matrix)[nodes, nodes] # type: ignore
        .ewise_mult(g_pu)
        .reduce_rowwise(gb.monoid.plus) # type: ignore
        .to_dense(fill_value=0.0)# type: ignore
    ).reshape(-1, 1)

    qflow_v0 = (
        gb.select.offdiag(descendent_matrix)[nodes, nodes] # type: ignore
        .ewise_mult(b_pu)  # type: ignore
        .reduce_rowwise(gb.monoid.plus)  # type: ignore
        .to_dense(fill_value=0.0)
        ).reshape(-1, 1)# type: ignore
        
    # Matrix value is the multiplication of transformer ratio found in upstream edge (for switch and branch n_transfo == 1)
    vn_v0 = (
        descendent_matrix.T[nodes, nodes]  # type: ignore
        .ewise_mult(n_transfo) # type: ignore
        .reduce_rowwise(gb.monoid.times) # type: ignore
        .to_dense(fill_value=0.0)
    ).reshape(-1, 1) # type: ignore
    
    # Simple matrix
    sload_sload = np.eye(2*n_edges)
    sload_v0 = np.zeros([2*n_edges, 1]) # type: ignore
    
    pflow_qload = np.zeros([n_edges,n_edges]) # type: ignore
    qflow_pload = np.zeros([n_edges,n_edges]) # type: ignore

    v0_sload = np.zeros([1, 2*n_edges])
    v0_v0 = np.ones([1,1])
    
    # Matrix concatenation
    h = np.concatenate([
        np.concatenate([sload_sload, sload_v0], axis=1),
        np.concatenate([pflow_pload, pflow_qload, pflow_v0], axis=1),
        np.concatenate([qflow_pload, qflow_qload, qflow_v0], axis=1),
        np.concatenate([vn_pload, vn_qload, vn_v0], axis=1),
        np.concatenate([v0_sload, v0_v0], axis=1),
    ], axis=0)
    
    return h