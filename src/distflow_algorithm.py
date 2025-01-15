import re
import polars as pl
from polars import col as c
import networkx as nx

import numpy as np
import scipy as sp
from copy import deepcopy
from typing import Union

from more_itertools import locate
from itertools import product

from networkx_function import generate_nx_edge, generate_bfs_tree_with_edge_data
from general_function import generate_log


log = generate_log(name=__name__)

class DistFlow():
    def __init__(self, line_data: pl.DataFrame, ext_grid_id: str):
        self.line_data: pl.DataFrame = line_data
        self.ext_grid_id: Union[str, int, float] = ext_grid_id
        self.nx_tree_grid: nx.DiGraph = nx.DiGraph()
        self.node_parameters: pl.DataFrame = pl.DataFrame()
        self.node_list: list[Union[str, int, float]] = []
        self.g_np: np.array = np.array([]) # type: ignore
        self.h_np: np.array = np.array([]) # type: ignore
        self.v_in_sq: np.array = np.array([]) # type: ignore
        self.dv_prop_np: np.array = np.array([]) # type: ignore
        self.z: np.array = np.array([]) # type: ignore
        self.y: np.array = np.array([]) # type: ignore

        self.generate_tree_graph()
        self.get_node_parameters()
        self.generate_connections_matrix()
        self.generate_voltage_update_matrix()
    
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
        self.nx_tree_grid : nx.DiGraph = generate_bfs_tree_with_edge_data(nx_grid, self.ext_grid_id)


    def get_node_parameters(self):
        """
        Extract node parameters from a NetworkX tree graph.

        Parameters:
        nx_tree_grid (nx.DiGraph): Directed graph representing the tree structure of the network.

        Returns:
        pl.DataFrame: DataFrame containing node parameters including impedance and transformer data.
        """
        node_data_list: list[dict] = []
        for node in self.nx_tree_grid.nodes:
            # Find every predecessor and successor edge data
            pred_data = self.nx_tree_grid.pred[node]
            succ_data = self.nx_tree_grid.succ[node]
            b_tra_pu = 0
            node_data = {"n_transfo": 1, "r_pu": 0, "x_pu": 0, "g_pu": 0}
            if pred_data:
                node_data = list(pred_data.values())[0]
                # Calculate the total shunt susceptance of predecessor edges and devide it by 2 if it is a branch
                if node_data["type"] == "transformer":
                    b_tra_pu += node_data["b_pu"]
                else:
                    b_tra_pu += node_data["b_pu"]/2
            # Calculate the total shunt susceptance of connected successor edges only if irt is a branch
            if succ_data:   
                b_tra_pu += pl.from_dicts(list(succ_data.values())).filter(c("type") == "transformer")["b_pu"].sum()/2
            node_data_list.append({"node_id": node} | node_data | {"b_tra_pu": b_tra_pu})
            
        self.node_parameters: pl.DataFrame = pl.from_dicts(node_data_list).rename({"g_pu": "g_tra_pu"}).drop("b_pu")
        self.node_list: list[Union[str, int, float]] = self.node_parameters["node_id"].to_list()
        # generate numpy arrays for impedance and admittance of each node
        self.z: np.array = self.node_parameters["r_pu"].to_numpy() + 1j*self.node_parameters["x_pu"].to_numpy() # type: ignore
        self.y: np.array =  self.node_parameters["g_tra_pu"].to_numpy() + 1j*self.node_parameters["b_tra_pu"].to_numpy() # type: ignore
        
    def generate_connections_matrix(self):
        self.g_np: np.array = nx.adjacency_matrix(self.nx_tree_grid, nodelist=self.node_list).toarray() # type: ignore
        self.h_np: np.array = sp.linalg.inv(np.eye(len(self.nx_tree_grid)) - self.g_np).astype(int) # type: ignore
        
    def generate_voltage_update_matrix(self):
        
        self.v_in_sq: np.array = np.array([1.0]*len(self.node_list)) # type: ignore
        self.dv_prop_np: np.array = deepcopy(self.h_np).transpose().astype(float) # type: ignore
        for data in self.node_parameters.filter(c("n_transfo") != 1).to_dicts():
            upstream_node: list[int] = list(locate(
                self.node_list, lambda x: x in nx.ancestors(self.nx_tree_grid, data["node_id"])
            ))
            if len(upstream_node) != 0:
                # Find array index pairs for the voltage drop matrix
                downstream_node: list[int] = list(locate(
                    self.node_list, 
                    lambda x: x in [data["node_id"]] + list(nx.descendants(self.nx_tree_grid, data["node_id"]))
                ))
                rows, cols = zip(*list(product(downstream_node, upstream_node)))
                # Update the voltage drop matrix
                self.dv_prop_np[rows, cols] = data["n_transfo"]*self.dv_prop_np[rows, cols]
                self.v_in_sq[downstream_node] = data["n_transfo"]*self.v_in_sq[downstream_node]
    

    def distflow_algorithm(
        self, s_node: np.array, v0_sq: np.array, max_iter: int = 100, tol: float = 1e-8, # type: ignore
            ): # type: ignore
        # State variable initialization
        v_sq: np.array = deepcopy(v0_sq) # type: ignore
        i_sq: np.array = np.zeros_like(v_sq) # type: ignore
        s_up: np.array = np.matmul(self.h_np, s_node) # type: ignore
        for idx in range(max_iter):
            # Update downstream side line power flow
            s_down: np.array = np.matmul(self.g_np, s_up) + s_node + v_sq*self.y # type: ignore
            # Update squared current
            i_sq_update: np.array = np.absolute(s_down)**2/v_sq # type: ignore
            # Update upstream side line power flow
            s_up = i_sq_update*self.z + s_down
            # Update squared voltage drop
            dv_sq: np.array = -2*np.real(np.conjugate(self.z)*s_up) + np.absolute(self.z)**2*i_sq_update # type: ignore
            # Update squared node voltage 
            v_sq_update: np.array = v0_sq + np.matmul(self.dv_prop_np, dv_sq) # type: ignore
            # Check convergence
            if max(np.abs(v_sq - v_sq_update).max(), np.abs(i_sq - i_sq_update).max()) < tol:
                break
            else:
                v_sq = deepcopy(v_sq_update)
                i_sq = deepcopy(i_sq_update)

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