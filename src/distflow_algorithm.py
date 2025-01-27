
import polars as pl
from polars import col as c
import networkx as nx
import numpy as np

from copy import deepcopy
from typing import Union
from numba import njit

import graphblas as gb
from networkx_function import generate_nx_edge, generate_bfs_tree_with_edge_data, get_all_edge_data, generate_shortest_path_length_matrix
from general_function import generate_log


log = generate_log(name=__name__)

class DistFlow():
    def __init__(self, line_data: pl.DataFrame, ext_grid_id: str):
        self.line_data: pl.DataFrame = line_data
        self.ext_grid_id: Union[str, int, float] = ext_grid_id
        self.nx_tree_grid: nx.DiGraph = nx.DiGraph()
        self.node_parameters: pl.DataFrame = pl.DataFrame()
        self.node_id_to_nb_mapping: dict[Union[str, int, float], int] = {}
        self.node_nb_to_id_mapping: dict[Union[str, int, float], int]= {}
        self.ext_grid_nb: int = 0
        self.node_list: list[Union[str, int, float]] = []
        self.g_np: gb.Matrix # type: ignore
        self.h_np: gb.Matrix # type: ignore
        self.v_in_sq: gb.Matrix # type: ignore
        self.dv_prop_np: gb.Matrix # type: ignore
        self.z: gb.Matrix # type: ignore
        self.y: gb.Matrix # type: ignore
        self.z_init_np: gb.Matrix # type: ignore
        self.idx: int = 0

        self.generate_node_index()
        self.generate_tree_graph()
        self.get_node_parameters()

        self.generate_connections_matrix()
        self.generate_voltage_update_matrix()
        self.generate_initial_impedance_matrix()
        
    def generate_node_index(self):
        node_id_list = self.line_data\
            .unpivot(on=["u_of_edge", "v_of_edge"], value_name="node_id")\
            .unique("node_id", keep="first")["node_id"].to_list()
        self.node_id_to_nb_mapping = dict(zip(node_id_list, range(len(node_id_list))))
        self.node_nb_to_id_mapping = dict(zip(range(len(node_id_list)), node_id_list))

        self.line_data = self.line_data.with_columns(
            c(col).replace_strict(self.node_id_to_nb_mapping, default=None).alias(col)
            for col in ["u_of_edge", "v_of_edge"]
        )
        self.ext_grid_nb = self.node_id_to_nb_mapping[self.ext_grid_id]

    
    def generate_tree_graph(self):
        """
        Generate a tree graph from line data and an external grid ID.

        Parameters:
        line_data (pl.DataFrame): DataFrame containing line data.
        ext_grid_id (str): ID of the external grid node.

        Returns:
        nx.DiGraph: Directed graph representing the tree structure of the network.
        """
        nx_grid: nx.Graph = nx.Graph()
        _ = self.line_data\
        .with_columns(
            pl.struct(pl.all()).pipe(generate_nx_edge, nx_graph=nx_grid)
        )
        if not nx.is_tree(nx_grid):
            log.error("The graph is not a tree")
        self.nx_tree_grid : nx.DiGraph = generate_bfs_tree_with_edge_data(nx_grid, self.ext_grid_nb)


    def get_node_parameters(self):
        """
        Extract node parameters from a NetworkX tree graph.

        Parameters:
        nx_tree_grid (nx.DiGraph): Directed graph representing the tree structure of the network.

        Returns:
        pl.DataFrame: DataFrame containing node parameters including impedance and transformer data.
        """
        node_parameters = get_all_edge_data(nx_graph=self.nx_tree_grid)
        node_parameters = pl.concat([node_parameters, pl.DataFrame({"v_of_edge": [self.ext_grid_nb]})], how="diagonal_relaxed")

        node_parameters = node_parameters.with_columns(
            pl.when(c("type")== "branch")
            .then(c("b_pu")/2)
            .otherwise(c("b_pu"))
        )
        # Add downstream branch admittance
        downstream_b: pl.DataFrame = node_parameters.filter(c("type") == "branch").group_by("u_of_edge").agg(c("b_pu").sum().alias("b_tra_pu"))

        self.node_parameters = node_parameters.join(downstream_b, left_on="v_of_edge", right_on="u_of_edge", how="left").select(
            c("v_of_edge").alias("node_id"), 
            c("r_pu", "x_pu").fill_null(0.0),
            pl.sum_horizontal(c("b_tra_pu").fill_null(0), c("b_pu").fill_null(0)).alias("b_tra_pu"),
            c("g_pu").fill_null(0.0).alias("g_tra_pu"),
            c("n_transfo").fill_null(1.0)
        ).sort("node_id")
        
        self.node_list: list[Union[str, int, float]] = self.node_parameters["node_id"].to_list()
        # generate numpy arrays for impedance and admittance of each node
        self.z: gb.Vector = gb.Vector.from_dense(self.node_parameters["r_pu"].to_numpy() + 1j*self.node_parameters["x_pu"].to_numpy())  # type: ignore
        self.z_conj: gb.Vector = self.z.apply(gb.unary.conj).dup()  # type: ignore
        self.z_sq: gb.Vector = self.z.apply(gb.unary.abs)**2  # type: ignore
        self.y: gb.Vector = gb.Vector.from_dense(self.node_parameters["g_tra_pu"].to_numpy() + 1j*self.node_parameters["b_tra_pu"].to_numpy()) # type: ignore
    
    def generate_connections_matrix(self):
        self.g_np: gb.Matrix = gb.Matrix.from_edgelist(self.nx_tree_grid.edges, nrows=len(self.nx_tree_grid), ncols=len(self.nx_tree_grid)) # type: ignore
        self.h_np: gb.Matrix = generate_shortest_path_length_matrix(nx_grid = self.nx_tree_grid, forced_weight=1) # type: ignore
        
    def generate_voltage_update_matrix(self):
        
        n_transfo = gb.Vector.from_dense(self.node_parameters["n_transfo"].to_numpy(), dtype=float) # type: ignore
        idx_trafo = self.node_parameters.filter(c("n_transfo") != 1)["node_id"].to_list()

        self.dv_prop_np: gb.Matrix = self.h_np.T.dup() # type: ignore

        self.v_in_sq = self.dv_prop_np.ewise_mult(n_transfo).reduce_rowwise("times").dup()

        n_transfo = gb.select.offdiag(self.h_np).ewise_mult(n_transfo)[:, idx_trafo].T.dup() # type: ignore
        n_transfo = gb.Matrix.from_dense(n_transfo.to_dense(fill_value= 1)) # type: ignore
        for i, idx in enumerate(idx_trafo):
            idx = self.dv_prop_np[:, idx].to_coo()[0]
            self.dv_prop_np[idx, :] = self.dv_prop_np[idx, :].ewise_mult(n_transfo[i, :])

        # n_transfo = self.node_parameters["n_transfo"].to_numpy()
        # idx_trafo = self.node_parameters.with_row_index(name="idx").filter(c("n_transfo") != 1)["idx"].to_list()
        # self.dv_prop_np = deepcopy(self.h_np.transpose().astype(float))
        # self.v_in_sq = np.multiply(self.h_np.transpose(), n_transfo)
        # self.v_in_sq[self.v_in_sq == 0] = 1
        # self.v_in_sq = np.prod(self.v_in_sq, axis=1)

        # trafo_factors = np.multiply(self.h_np - np.eye(N=len(self.h_np)), self.node_parameters["n_transfo"].to_numpy()).transpose()[idx_trafo, :]
        # trafo_factors[trafo_factors==0] = 1

        # for i, trafo_factor in zip(idx_trafo, trafo_factors):
        #     idx = np.where(self.dv_prop_np[:, i] == 1)
        #     self.dv_prop_np[idx, :] = np.multiply(self.dv_prop_np[idx, :], trafo_factor)

    def generate_initial_impedance_matrix(self):

        r_init: gb.Matrix = generate_shortest_path_length_matrix(nx_grid = self.nx_tree_grid, weight_name="r_pu") # type: ignore
        x_init: gb.Matrix = generate_shortest_path_length_matrix(nx_grid = self.nx_tree_grid, weight_name="x_pu") # type: ignore
        self.z_init: gb.Matrix = r_init.ewise_add(1j* x_init) # type: ignore

    def distflow_algorithm_gb(
        self, s_node: gb.Vector, v0_sq: gb.Vector, max_iter: int = 100, tol: float = 1e-5, # type: ignore
            ): # type: ignore
        
        v_sq = v0_sq.dup()
        i_sq = gb.Vector.from_dense(np.zeros(len(self.node_id_to_nb_mapping))) # type: ignore
        s_up = self.h_np.mxv(s_node)

        for idx in range(max_iter):
            s_tran = v_sq.ewise_mult(self.y).dup()
            s_down = self.g_np.mxv(s_up).ewise_add(s_node).ewise_add(s_tran).dup()
            i_sq_update = (s_down.apply(gb.unary.abs)**2).ewise_mult(1/v_sq).dup() # type: ignore
            s_up = i_sq_update.ewise_mult(self.z).ewise_add(s_down).dup()
            dv_sq = self.z_sq.ewise_mult(i_sq_update).ewise_add(-2* self.z_conj.ewise_mult(s_up).apply(gb.unary.creal)).dup() # type: ignore
            v_sq_update = v0_sq.ewise_add(self.dv_prop_np.mxv(dv_sq)).dup()
            diff_i = i_sq.ewise_add(-i_sq_update).ewise_mult(1/i_sq_update).apply(gb.unary.abs).reduce(gb.binary.max).dup() # type: ignore
            diff_u = v_sq.ewise_add(-v_sq_update).ewise_mult(1/v_sq_update).apply(gb.unary.abs).reduce(gb.binary.max).dup() # type: ignore
            if (diff_u < tol) & (diff_i < tol):
                break
            else:
                v_sq = v_sq_update.dup()
                i_sq = i_sq_update.dup()
        self.idx = idx    
        return (
            s_up.apply(gb.unary.creal).dup(), 
            s_up.apply(gb.unary.cimag).dup(),
            i_sq_update.apply(gb.unary.sqrt).dup(), 
            v_sq_update.apply(gb.unary.sqrt).dup()
        )
        
    def distflow_algorithm_nb(
        self, s_node: gb.Vector, v0_sq: gb.Vector, max_iter: int = 100, tol: float = 1e-5, # type: ignore
        )
        # State variable initialization
        v_sq: np.array = deepcopy(v0_sq) # type: ignore
        i_sq: np.array = np.zeros_like(v_sq) # type: ignore
        s_up: np.array = np.matmul(self.h_np, s_node) # type: ignore
        # s_init: np.array = np.matmul(self.h_np, s_node + v_sq*self.y) # type: ignore
        # i_sq: np.array = np.square(np.absolute(s_init))/v_sq # type: ignore
        # s_up: np.array = s_init + np.matmul(self.z_init_np, i_sq) # type: ignore
        for idx in range(max_iter):
            # Update downstream side line power flow
            s_down: np.array = np.matmul(self.g_np, s_up) + s_node + v_sq*self.y # type: ignore
            # Update squared current
            i_sq_update: np.array = np.absolute(s_down)**2/v_sq # type: ignore
            # Update upstream side line power flow
            s_up = i_sq_update*self.z + s_down
            # Update squared voltage drop
            dv_sq: np.array = -2*np.real(np.multiply(np.conjugate(self.z), s_up)) + np.absolute(self.z)**2*i_sq_update # type: ignore
            # Update squared node voltage 
            v_sq_update: np.array = v0_sq + np.matmul(self.dv_prop_np, dv_sq) # type: ignore
            # Check convergence
            # print((np.abs(i_sq - i_sq_update)/i_sq_update).max())
            if max((np.abs(v_sq - v_sq_update)/v_sq_update).max(), (np.abs(i_sq - i_sq_update)/i_sq_update).max()) < tol:
                break
            else:
                v_sq = deepcopy(v_sq_update)
                i_sq = deepcopy(i_sq_update)
        self.idx = idx
        return np.real(s_up), np.imag(s_up), np.sqrt(v_sq_update), np.sqrt(i_sq_update)
    
    def timeseries_distflow_algorithm(
        self, p_node: np.array, q_node: np.array, v_ext_grid_sq: Union[float, np.array] = 1.0, # type: ignore
        max_iter=100, tol=1e-6): # type: ignore
        
        if np.shape(p_node) != np.shape(q_node):
            raise ValueError("Real and reactive power values must have the same shape.")
        if np.shape(p_node)[1] != len(self.node_list):
            raise ValueError("Power values must have the same length as the node list.")
        if isinstance(v_ext_grid_sq, float):
            v_ext_grid_sq = np.array([v_ext_grid_sq]*np.shape(p_node)[0])
        else:
            if np.shape(v_ext_grid_sq)[0] != np.shape(p_node)[0]:
                raise ValueError("External grid voltage values must have the same length as powers.")

        s_node: np.array = p_node + 1j*q_node # type: ignore
        v0_sq: np.array = self.v_in_sq * v_ext_grid_sq.reshape(np.shape(v_ext_grid_sq)[0], 1) # type: ignore
        p_flow: np.array = np.zeros_like(s_node) # type: ignore
        q_flow: np.array = np.zeros_like(s_node) # type: ignore
        v: np.array = np.zeros_like(s_node) # type: ignore
        i: np.array = np.zeros_like(s_node) # type: ignore
        for idx in range(np.shape(p_node)[0]):
            p_flow[idx, :], q_flow[idx, :], v[idx, :], i[idx, :] = self.distflow_algorithm(
                s_node[idx, :], v0_sq[idx, :], max_iter=max_iter, tol=tol
            )
            
        return p_flow, q_flow, v, i
