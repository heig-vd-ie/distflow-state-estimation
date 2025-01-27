r"""
Generate the full Jacobian matrix for the given distribution flow schema.

Args:
    distflow_schema (DistFlowSchema): The schema containing the edge data for the distribution flow.
    slack_node_id (int): The ID of the slack node in the grid.

Returns:
    np.array: The full Jacobian matrix.

Raises:
    ValueError: If edge_data is empty, contains parallel edges, or if the slack node is not in the grid.
    ValueError: If the grid is not a connected tree.
    
The Jacobian matrix is a block matrix with the following structure:

.. math::
    :label: jacobian-matrix
    :nowrap:
    
    \begin{align}
    \Large{
        H(x)= \begin{bmatrix}
            \frac{\partial P_\text{load}}{\partial P_\text{load}} & \frac{\partial P_\text{load}}{\partial Q_\text{load}} & 
            \frac{\partial P_\text{load}}{\partial V_\text{0}^{2}} \\
            \frac{\partial Q_\text{load}}{\partial P_\text{load}} & \frac{\partial Q_\text{load}}{\partial Q_\text{load}} & 
            \frac{\partial Q_\text{load}}{\partial V_\text{0}^{2}} \\
            \frac{\partial P_\text{flow}}{\partial P_\text{load}} & \frac{\partial P_\text{flow}}{\partial Q_\text{load}} & 
            \frac{\partial P_\text{flow}}{\partial V_\text{0}^{2}} \\
            \frac{\partial Q_\text{flow}}{\partial P_\text{load}} & \frac{\partial Q_\text{flow}}{\partial Q_\text{load}} & 
            \frac{\partial Q_\text{flow}}{\partial V_\text{0}^{2}} \\
            \frac{\partial V^{2}}{\partial P_\text{load}} & \frac{\partial V^{2}}{\partial Q_\text{load}} & 
            \frac{\partial V^{2}}{\partial V_\text{0}^{2}} \\
            \frac{\partial V_\text{0}^{2}}{\partial P_\text{load}} & \frac{\partial V_\text{0}^{2}}{\partial Q_\text{load}} & 
            \frac{\partial V_\text{0}^{2}}{\partial V_\text{0}^{2}}
        \end{bmatrix}
    }
    \end{align}

.. math::
    :nowrap:
    
    \begin{align}
        \frac{\partial P_\text{load}}{\partial P_\text{load}} = 
        \frac{\partial Q_\text{load}}{\partial Q_\text{load}} = I
    \end{align}


.. math::
    :nowrap:
    
    \begin{align}
        \frac{\partial P_\text{flow}^{i}}{\partial P_\text{load}^{j}} = 
        \frac{\partial Q_\text{flow}^{i}}{\partial Q_\text{load}^{j}} = 
        \begin{cases}
            1 &\text{if node } i \text{ is downstream node } j \\
            0 &\text{otherwise}
        \end{cases}
    \end{align}

.. math::
    :nowrap:
    
    \begin{align}
        \frac{\partial V^{i}}{\partial P_\text{load}^{j}} = - 2 \cdot \displaystyle\sum_{k \in K} R_{k}
    \end{align}

.. math::
    :nowrap:
    
    \begin{align}
        \frac{\partial V^{i}}{\partial Q_\text{load}^{j}} = - 2 \cdot \displaystyle\sum_{k \in K} X_{k}
    \end{align}        

where :math:`K` is the set of edges in the path connecting the slack node to the lowest common ancestor of nodes 
:math:`i` and :math:`j`.

.. math::
    :nowrap:
    
    \begin{align}
        \frac{\partial P_\text{flow}^{i}}{\partial V_\text{0}^{2}} = - \displaystyle\sum_{l \in L} G_{l}
    \end{align}    

.. math::
    :nowrap:
    
    \begin{align}
        \frac{\partial Q_\text{flow}^{i}}{\partial V_\text{0}^{2}} = - \displaystyle\sum_{l \in L} B_{l}
    \end{align} 

where :math:`L` is the set of edges connected downstream the node :math:`i`.    
    
.. math::
    :nowrap:
    
    \begin{align}
        \frac{\partial V^{2}}{\partial V_\text{0}^{2}} = - \displaystyle\prod_{m \in M} N_{m}
    \end{align} 


where :math:`M` is the set of edges connected upstream the node :math:`i`.

.. math::
    :nowrap:
    
    \begin{align}
        \frac{\partial P_\text{load}}{\partial Q_\text{load}},\frac{\partial Q_\text{load}}{\partial P_\text{load}},
        \frac{\partial P_\text{load}}{\partial V_\text{0}^{2}}, \frac{\partial Q_\text{load}}{\partial V_\text{0}^{2}}, 
        \frac{\partial P_\text{flow}}{\partial Q_\text{load}}, \frac{\partial Q_\text{flow}}{\partial P_\text{load}}, 
        \frac{\partial V_\text{0}^{2}}{\partial P_\text{load}}, \frac{\partial V_\text{0}^{2}}{\partial Q_\text{load}}= {0}
    \end{align} 

"""

import polars as pl
from polars import col as c
import patito as pt
import networkx as nx
import numpy as np
import graphblas as gb

from networkx_function import (
    generate_shortest_path_length_matrix, generate_bfs_tree_with_edge_data, generate_nx_edge)
from general_function import generate_log

from distflow_schema import DistFlowSchema

log = generate_log(name=__name__)

def generate_full_jacobian_matrix(distflow_schema: DistFlowSchema, slack_node_id: int)-> np.array: # type: ignore
    r"""
    Generate the full Jacobian matrix for the given distribution flow schema.

    Args:
        distflow_schema (DistFlowSchema): The schema containing the edge data for the distribution flow.
        slack_node_id (int): The ID of the slack node in the grid.

    Returns:
        np.array: The full Jacobian matrix.

    Raises:
        ValueError: If edge_data is empty, contains parallel edges, or if the slack node is not in the grid.
        ValueError: If the grid is not a connected tree.
        
    The Jacobian matrix is a block matrix with the following structure:
    
    .. math::
        :label: jacobian-matrix
        :nowrap:
        
        \begin{align}
        \Large{
            H(x)= \begin{bmatrix}
                \frac{\partial P_\text{load}}{\partial P_\text{load}} & \frac{\partial P_\text{load}}{\partial Q_\text{load}} & 
                \frac{\partial P_\text{load}}{\partial V_\text{0}^{2}} \\
                \frac{\partial Q_\text{load}}{\partial P_\text{load}} & \frac{\partial Q_\text{load}}{\partial Q_\text{load}} & 
                \frac{\partial Q_\text{load}}{\partial V_\text{0}^{2}} \\
                \frac{\partial P_\text{flow}}{\partial P_\text{load}} & \frac{\partial P_\text{flow}}{\partial Q_\text{load}} & 
                \frac{\partial P_\text{flow}}{\partial V_\text{0}^{2}} \\
                \frac{\partial Q_\text{flow}}{\partial P_\text{load}} & \frac{\partial Q_\text{flow}}{\partial Q_\text{load}} & 
                \frac{\partial Q_\text{flow}}{\partial V_\text{0}^{2}} \\
                \frac{\partial V^{2}}{\partial P_\text{load}} & \frac{\partial V^{2}}{\partial Q_\text{load}} & 
                \frac{\partial V^{2}}{\partial V_\text{0}^{2}} \\
                \frac{\partial V_\text{0}^{2}}{\partial P_\text{load}} & \frac{\partial V_\text{0}^{2}}{\partial Q_\text{load}} & 
                \frac{\partial V_\text{0}^{2}}{\partial V_\text{0}^{2}}
            \end{bmatrix}
        }
        \end{align}
    
    .. math::
        :nowrap:
        
        \begin{align}
            \frac{\partial P_\text{load}}{\partial P_\text{load}} = 
            \frac{\partial Q_\text{load}}{\partial Q_\text{load}} = I
        \end{align}
    

    .. math::
        :nowrap:
        
        \begin{align}
            \frac{\partial P_\text{flow}^{i}}{\partial P_\text{load}^{j}} = 
            \frac{\partial Q_\text{flow}^{i}}{\partial Q_\text{load}^{j}} = 
            \begin{cases}
                1 &\text{if node } i \text{ is downstream node } j \\
                0 &\text{otherwise}
            \end{cases}
        \end{align}
    
    .. math::
        :nowrap:
        
        \begin{align}
            \frac{\partial V^{i}}{\partial P_\text{load}^{j}} = - 2 \cdot \displaystyle\sum_{k \in K} R_{k}
        \end{align}
    
    .. math::
        :nowrap:
        
        \begin{align}
            \frac{\partial V^{i}}{\partial Q_\text{load}^{j}} = - 2 \cdot \displaystyle\sum_{k \in K} X_{k}
        \end{align}        

    where :math:`K` is the set of edges in the path connecting the slack node to the lowest common ancestor of nodes 
    :math:`i` and :math:`j`.
    
    .. math::
        :nowrap:
        
        \begin{align}
            \frac{\partial P_\text{flow}^{i}}{\partial V_\text{0}^{2}} = - \displaystyle\sum_{l \in L} G_{l}
        \end{align}    
    
    .. math::
        :nowrap:
        
        \begin{align}
            \frac{\partial Q_\text{flow}^{i}}{\partial V_\text{0}^{2}} = - \displaystyle\sum_{l \in L} B_{l}
        \end{align} 
    
    where :math:`L` is the set of edges connected downstream the node :math:`i`.    
        
    .. math::
        :nowrap:
        
        \begin{align}
            \frac{\partial V^{2}}{\partial V_\text{0}^{2}} = - \displaystyle\prod_{m \in M} N_{m}
        \end{align} 
    
    
    where :math:`M` is the set of edges connected upstream the node :math:`i`.
    
    .. math::
        :nowrap:
        
        \begin{align}
            \frac{\partial P_\text{load}}{\partial Q_\text{load}},\frac{\partial Q_\text{load}}{\partial P_\text{load}},
            \frac{\partial P_\text{load}}{\partial V_\text{0}^{2}}, \frac{\partial Q_\text{load}}{\partial V_\text{0}^{2}}, 
            \frac{\partial P_\text{flow}}{\partial Q_\text{load}}, \frac{\partial Q_\text{flow}}{\partial P_\text{load}}, 
            \frac{\partial V_\text{0}^{2}}{\partial P_\text{load}}, \frac{\partial V_\text{0}^{2}}{\partial Q_\text{load}}= {0}
        \end{align} 
    
    """
    edge_data: pt.DataFrame = distflow_schema.edge_data
    if edge_data.is_empty():
        raise ValueError("edge_data is empty")
    # Check if there is no parallel edges (nx.Graph does not support parallel edges instead of nx.MultiGraph)
    if not edge_data.filter(pl.struct("u_of_edge", "v_of_edge").is_duplicated()).is_empty():
        raise ValueError("Edges in parallel in edge_data")
    # check if the slack node is in the grid
    if edge_data.filter(pl.any_horizontal(c("u_of_edge", "v_of_edge")== 0)).is_empty():
        raise ValueError("The slack node is not in the grid")
    
    # Create nx_tree from line data
    nx_grid: nx.Graph = nx.Graph()
    _ = edge_data\
    .with_columns(
        pl.struct(pl.all()).pipe(generate_nx_edge, nx_graph=nx_grid)
    )
    # Check if the grid is a connected tree
    if not nx.is_tree(nx_grid):
        raise ValueError("The grid is not a tree")
    elif not nx.is_connected(nx_grid):
        raise ValueError("The grid is not connected")

    nx_tree_grid : nx.DiGraph = generate_bfs_tree_with_edge_data(nx_grid, slack_node_id)
        
    # remove slack node from list of nodes
    nodes = np.array(nx_tree_grid.nodes())
    nodes = nodes[nodes != slack_node_id]

    n_nodes: int = nx_tree_grid.number_of_nodes()
    n_edges: int = nx_tree_grid.number_of_edges()
    
    descendent_matrix = generate_shortest_path_length_matrix(nx_grid = nx_tree_grid, forced_weight=1)
    
    # Create edge data graphblas vector
    g_pu = gb.Vector.from_dense(edge_data["g_pu"]) # type: ignore
    b_pu = gb.Vector.from_dense(edge_data["b_pu"]) # type: ignore
    n_transfo = gb.Vector.from_dense(edge_data["n_transfo"]) # type: ignore

    # Matrix values represent the sum of longitudinal resistances (or reactance) of edges in the path connecting the slack node to the 
    # lowest common ancestor of i ang j (row and column index node).
    coords, ancestor = list(zip(*nx.all_pairs_lowest_common_ancestor(nx_tree_grid)))
    x, y = list(zip(*coords))

    r_mapping = nx.shortest_path_length(nx_tree_grid, source=slack_node_id, weight="r_pu")
    r_val = list(map(lambda node_id: -2*r_mapping[node_id], ancestor))

    vn_pload_gb = gb.Matrix.from_coo(x, y, r_val, nrows=n_nodes, ncols=n_nodes).select("!=", 0)[nodes, nodes] # type: ignore
    vn_pload = (
        gb.select.offdiag(vn_pload_gb).T # type: ignore
        .ewise_add(vn_pload_gb) # type: ignore
        .to_dense(fill_value=0.0) # type: ignore
    )

    x_mapping = nx.shortest_path_length(nx_tree_grid, source=slack_node_id, weight="x_pu")
    x_val = list(map(lambda node_id: -2*x_mapping[node_id], ancestor))

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
        
    # Matrix value is the multiplication of transformer ratio found in upstream edge (for switch and branch, n_transfo == 1)
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