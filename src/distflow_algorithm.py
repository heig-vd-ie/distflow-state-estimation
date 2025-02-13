"""



"""


import polars as pl
from polars import col as c
import networkx as nx
import numpy as np
from copy import deepcopy
from typing import Union, Literal
import graphblas as gb
from networkx_function import (
    generate_tree_graph_from_edge_data, get_all_edge_data, generate_shortest_path_length_matrix)
from general_function import generate_log

log = generate_log(name=__name__)
class DistFlow():
    def __init__(self, line_data: pl.DataFrame, ext_grid_id: str, max_iter: int = 150, tol: float=1e-6):
        
        self.max_iter: int = max_iter
        self.tol: float = tol
        
        self.line_data: pl.DataFrame = line_data
        self.ext_grid_id: Union[str, int, float] = ext_grid_id
        
        self.node_parameters: pl.DataFrame
        self.node_id_to_nb_mapping: dict[Union[str, int, float], int]
        self.node_nb_to_id_mapping: dict[Union[str, int, float], int]
        self.ext_grid_nb: int
        self.node_list: list[Union[str, int, float]]
        self.idx: int = 0
        
        self.g_gb: gb.Matrix # type: ignore
        self.h_gb: gb.Matrix # type: ignore
        self.v_in_sqr_gb: gb.Vector # type: ignore
        self.dv_prop_gb: gb.Vector # type: ignore
        self.z_gb: gb.Vector # type: ignore
        self.z_conj_gb: gb.Vector # type: ignore
        self.z_sqr_gb: gb.Vector # type: ignore
        self.y_gb: gb.Vector # type: ignore
        
        self.g_np: np.array # type: ignore
        self.h_np: np.array # type: ignore
        self.v_in_sqr_np: np.array # type: ignore
        self.dv_prop_np: np.array # type: ignore
        self.z_np: np.array # type: ignore
        self.z_conj_np: np.array # type: ignore
        self.z_sqr_np: np.array # type: ignore
        self.y_np: np.array # type: ignore
        
        self.generate_node_index()
        
        self.nx_tree_grid: nx.DiGraph = generate_tree_graph_from_edge_data(
            edge_data=self.line_data, slack_node_id=self.ext_grid_nb)
        
        self.get_node_parameters()
        self.generate_connections_matrix()
        self.generate_voltage_update_matrix()
        self.graphblas_to_numpy()
        
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

    def get_node_parameters(self):
        node_parameters = get_all_edge_data(nx_graph=self.nx_tree_grid)
        node_parameters = pl.concat([
            node_parameters, pl.DataFrame({"v_of_edge": [self.ext_grid_nb]})], how="diagonal_relaxed")
        node_parameters = node_parameters.with_columns(
            pl.when(c("type")== "branch")
            .then(c("b_pu")/2)
            .otherwise(c("b_pu"))
        )
        # Add downstream branch admittance
        downstream_b: pl.DataFrame = node_parameters\
            .filter(c("type") == "branch")\
            .group_by("u_of_edge")\
            .agg(c("b_pu").sum().alias("b_tra_pu"))
            
        self.node_parameters = node_parameters\
            .join(downstream_b, left_on="v_of_edge", right_on="u_of_edge", how="left")\
            .select(
                c("v_of_edge").alias("node_id"), 
                c("r_pu", "x_pu").fill_null(0.0),
                pl.sum_horizontal(c("b_tra_pu").fill_null(0), c("b_pu").fill_null(0)).alias("b_tra_pu"),
                c("g_pu").fill_null(0.0).alias("g_tra_pu"),
                c("n_transfo").fill_null(1.0)
            ).sort("node_id")
        
        self.node_list: list[Union[str, int, float]] = self.node_parameters["node_id"].to_list()
        # generate numpy arrays for impedance and admittance of each node
        self.z_gb: gb.Vector = gb.Vector.from_dense(self.node_parameters["r_pu"].to_numpy() + 1j*self.node_parameters["x_pu"].to_numpy())  # type: ignore
        self.z_conj_gb: gb.Vector = self.z_gb.apply(gb.unary.conj).dup()  # type: ignore
        self.z_sqr_gb: gb.Vector = self.z_gb.apply(gb.unary.abs)**2  # type: ignore
        self.y_gb: gb.Vector = gb.Vector.from_dense(self.node_parameters["g_tra_pu"].to_numpy() + 1j*self.node_parameters["b_tra_pu"].to_numpy()) # type: ignore
    
    def generate_connections_matrix(self):
        self.g_gb: gb.Matrix = gb.Matrix.from_edgelist(self.nx_tree_grid.edges, nrows=len(self.nx_tree_grid), ncols=len(self.nx_tree_grid)) # type: ignore
        self.h_gb: gb.Matrix = generate_shortest_path_length_matrix(nx_grid = self.nx_tree_grid, forced_weight=1) # type: ignore
        
    def generate_voltage_update_matrix(self):
        
        n_transfo = gb.Vector.from_dense(self.node_parameters["n_transfo"].to_numpy(), dtype=float) # type: ignore
        idx_trafo = self.node_parameters.filter(c("n_transfo") != 1)["node_id"].to_list()
        self.dv_prop_gb: gb.Matrix = self.h_gb.T.dup() # type: ignore
        self.v_in_sqr_gb = self.dv_prop_gb.ewise_mult(n_transfo).reduce_rowwise("times").dup()
        n_transfo = gb.select.offdiag(self.h_gb).ewise_mult(n_transfo)[:, idx_trafo].T.dup() # type: ignore
        n_transfo = gb.Matrix.from_dense(n_transfo.to_dense(fill_value= 1)) # type: ignore
        for i, idx in enumerate(idx_trafo):
            idx = self.dv_prop_gb[:, idx].to_coo()[0]
            self.dv_prop_gb[idx, :] = self.dv_prop_gb[idx, :].ewise_mult(n_transfo[i, :])
        
    def graphblas_to_numpy(self):
        self.g_np = self.g_gb.to_dense(fill_value=0.0)
        self.h_np = self.h_gb.to_dense(fill_value=0.0)
        self.v_in_sqr_np = self.v_in_sqr_gb.to_dense()
        self.dv_prop_np = self.dv_prop_gb.to_dense(fill_value=0.0)
        self.z_np = self.z_gb.to_dense()
        self.z_conj_np = self.z_conj_gb.to_dense()
        self.z_sqr_np = self.z_sqr_gb.to_dense()
        self.y_np = self.y_gb.to_dense()
        
    def distflow_algorithm(
        self, s_node: np.array, v0_sqr: np.array, engine: Literal["numpy", "graphblas"] = "graphblas"): # type: ignore
        
        if engine == "numpy":
            s_up, v_sqr_update, i_sqr_update = self.distflow_algorithm_np(s_node, v0_sqr)
        elif engine == "graphblas":
            s_up_gb, v_sqr_update_gb, i_sqr_update_gb = self.distflow_algorithm_gb(
                gb.Vector.from_dense(s_node), gb.Vector.from_dense(v0_sqr) # type: ignore
            )
            s_up = s_up_gb.to_dense()
            v_sqr_update = v_sqr_update_gb.to_dense()
            i_sqr_update = i_sqr_update_gb.to_dense()
        else:
            raise ValueError("The engine parameter must be either 'numpy' or 'graphblas'")
        
        if self.idx == self.max_iter:
            log.error("The algorithm did not converge after {} iterations".format(self.max_iter))
            
        return s_up, v_sqr_update, i_sqr_update
        
    def distflow_algorithm_gb(
        self, s_node: gb.Vector, v0_sqr: gb.Vector # type: ignore
            ): # type: ignore
        # State variable initialization
        v_sqr = v0_sqr.dup()
        i_sqr = gb.Vector.from_dense(np.zeros(len(self.node_id_to_nb_mapping))) # type: ignore
        s_up = self.h_gb.mxv(s_node)
        for idx in range(self.max_iter):
            # Update downstream side line power flow
            s_down = self.g_gb.mxv(s_up).ewise_add(s_node).ewise_add(v_sqr.ewise_mult(self.y_gb))
            # Update squared current
            i_sqr_update = (s_down.apply(gb.unary.abs)**2).ewise_mult(1/v_sqr) # type: ignore
            # Update upstream side line power flow
            s_up = i_sqr_update.ewise_mult(self.z_gb).ewise_add(s_down)
            # Update squared voltage drop
            dv_sqr = self.z_sqr_gb.ewise_mult(i_sqr_update).ewise_add(-2* self.z_conj_gb.ewise_mult(s_up).apply(gb.unary.creal)) # type: ignore
            # Update squared node voltage 
            v_sqr_update = v0_sqr.ewise_add(self.dv_prop_gb.mxv(dv_sqr))
            # Check convergence
            diff_i = i_sqr.ewise_add(-i_sqr_update).ewise_mult(1/i_sqr_update).apply(gb.unary.abs).reduce(gb.binary.max) # type: ignore
            diff_u = v_sqr.ewise_add(-v_sqr_update).ewise_mult(1/v_sqr_update).apply(gb.unary.abs).reduce(gb.binary.max) # type: ignore
            if (diff_u < self.tol) & (diff_i < self.tol):
                break
            else:
                v_sqr = v_sqr_update.dup()
                i_sqr = i_sqr_update.dup()
        self.idx = idx
        if self.idx == self.max_iter:
            log.error("The algorithm did not converge after {} iterations".format(self.max_iter))
        return s_up, v_sqr_update, i_sqr_update
        
    def distflow_algorithm_np(
        self, s_node: np.array, v0_sqr: np.array # type: ignore
        ):
        # State variable initialization
        v_sqr: np.array = deepcopy(v0_sqr) # type: ignore
        i_sqr: np.array = np.zeros_like(v_sqr) # type: ignore
        s_up: np.array = np.matmul(self.h_np, s_node) # type: ignore
        for idx in range(self.max_iter):
            # Update downstream side line power flow
            s_down: np.array = np.matmul(self.g_np, s_up) + s_node + np.multiply(v_sqr, self.y_np) # type: ignore
            # Update squared current
            i_sqr_update: np.array = np.absolute(s_down)**2/v_sqr # type: ignore
            # Update upstream side line power flow
            s_up = i_sqr_update*self.z_np + s_down
            # Update squared voltage drop
            dv_sqr: np.array = -2*np.real(np.multiply(self.z_conj_np, s_up)) + np.multiply(self.z_sqr_np, i_sqr_update) # type: ignore
            # Update squared node voltage 
            v_sqr_update: np.array = v0_sqr + np.matmul(self.dv_prop_np, dv_sqr) # type: ignore
            # Check convergence
            if max((np.abs(v_sqr - v_sqr_update)/v_sqr_update).max(), (np.abs(i_sqr - i_sqr_update)/i_sqr_update).max()) < self.tol:
                break
            else:
                v_sqr = deepcopy(v_sqr_update)
                i_sqr = deepcopy(i_sqr_update)
        self.idx = idx
        if self.idx == self.max_iter:
            log.error("The algorithm did not converge after {} iterations".format(self.max_iter))
        return s_up, v_sqr_update, i_sqr_update
    
    def timeseries_distflow_algorithm(
        self, s_node: np.array,  v_ext_grid_sqr: Union[float, np.array] = 1.0, # type: ignore
        engine: Literal["numpy", "graphblas"] = "graphblas"
        ): # type: ignore
        
        if np.shape(s_node)[1] != len(self.node_list):
            raise ValueError("Power values must have the same length as the node list.")
        if isinstance(v_ext_grid_sqr, float):
            v_ext_grid_sqr = np.array([v_ext_grid_sqr]*np.shape(s_node)[0])
        else:
            if np.shape(v_ext_grid_sqr)[0] != np.shape(s_node)[0]:
                raise ValueError("External grid voltage values must have the same length as powers.")
        v0_sqr: np.array = self.v_in_sqr_np * v_ext_grid_sqr.reshape(np.shape(v_ext_grid_sqr)[0], 1) # type: ignore
        
        if engine == "numpy":
            s_flow: np.array = np.zeros_like(s_node, dtype= np.complex128) # type: ignore
            v_sqr: np.array = np.zeros_like(s_node, dtype= np.float64) # type: ignore
            i_sqr: np.array = np.zeros_like(s_node, dtype= np.float64) # type: ignore
            for idx in range(np.shape(s_node)[0]):
                s_flow[idx, :], v_sqr[idx, :], i_sqr[idx, :] = self.distflow_algorithm_np(s_node[idx, :], v0_sqr[idx, :])  
                
        elif engine == "graphblas":
            s_node_gb: gb.Matrix = gb.Matrix.from_dense(s_node) # type: ignore
            v0_sqr_gb: gb.Matrix = gb.Matrix.from_dense(v0_sqr) # type: ignore
            nrows, ncols = np.shape(s_node)
            s_flow_gb: gb.Matrix = gb.Matrix(dtype=np.complex128, nrows=nrows, ncols=ncols) # type: ignore
            v_sqr_gb: gb.Matrix = gb.Matrix(dtype=float, nrows=nrows, ncols=ncols) # type: ignore
            i_sqr_gb: gb.Matrix = gb.Matrix(dtype=float, nrows=nrows, ncols=ncols) # type: ignore
            for idx in range(np.shape(s_node)[0]):
                s_flow_gb[idx, :], v_sqr_gb[idx, :], i_sqr_gb[idx, :] = self.distflow_algorithm_np(
                    s_node_gb[idx, :], v0_sqr_gb[idx, :]) 
                
            s_flow = s_flow_gb.to_dense()
            v_sqr = v_sqr_gb.to_dense()
            i_sqr = i_sqr_gb.to_dense()
        else:
            raise ValueError("The engine parameter must be either 'numpy' or 'graphblas'")                
        return s_flow, v_sqr, i_sqr
