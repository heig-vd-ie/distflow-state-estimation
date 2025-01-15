
from math import dist
import unittest
import polars as pl
from polars import col as c
import networkx as nx
from networkx_function import generate_nx_edge
from shapely import node
from typing_extensions import Union
import numpy as np
import scipy as sp
import pandapower as pp

from data_connector import pandapower_to_distflow
from distflow_algorithm import DistFlow





class TestDistFlow(unittest.TestCase):
    
    def test_small_grid(self):
        ext_grid_id = "1" 
        n = 0.95
        v_ext_grid_sq = 1.05

        line_data: pl.DataFrame = pl.DataFrame({
            "line_id":np.arange(1, 13).astype(str),
            "type": ["branch"]*10 + ["transformer"]*2,
            "u_of_edge": ["2", "2", "2", "4", "5",  "6", "7", "10", "12", "13", "8", "9"],
            "v_of_edge": ["11", "1", "3", "1", "4", "4", "8",  "9", "11", "11", "4", "8"],
            "n_transfo": [1.0]* 10 + [n**2, n**2],
            "x_pu": np.arange(1, 13)*2e-3,
            "r_pu": np.arange(1, 13)*1e-3,
            "b_pu": list(np.arange(1, 11)*- 1e-3) + [0.001, 0.001],
            "g_pu": [0.]* 10 + [0.001, 0.001]
        })

        node_data: pl.DataFrame = pl.DataFrame({
            "node_id":np.arange(1, 14).astype(str),
            "v_base": [400.0]*7 + [200.0, 400.0  , 400.0  , 200.0 , 100.0, 100.0],
            "p_node_pu": np.array([0, 10, 2, 0, 0, -2, 0, 7, 25, 0, 100, 0.5, 10])*1e-2,
            "q_node_pu": np.array([0, 0, 0.2, 2, 0, -0.2, 0, 0.7, 0, 0, 0, 0.5, 1])*1e-2,
            }, strict=False)

        expected_p_flow = np.array([ 
            1.53340168e+00,  1.22914685e+00,  3.04254824e-01,  1.10626770e+00,
            2.00011490e-02,  3.27291823e-08, -1.99975577e-02,  3.23898433e-01,
            5.00021661e-03,  1.00096321e-01,  8.04252483e-08,  2.51739040e-01,
            1.07587806e-07])

        expected_v = np.array([
            1.02469508, 1.0222593 , 1.02336771, 1.02116547, 1.02219805,
            1.02339329, 1.02354522, 0.96840687, 1.02111597, 1.02008932,
            0.96845432, 0.91674541, 0.91680409])
        
        distflow = DistFlow(line_data=line_data, ext_grid_id=ext_grid_id)
        v0_sq = distflow.v_in_sq*v_ext_grid_sq
        node_data = pl.DataFrame({"node_id": distflow.node_list}).join(node_data, on="node_id", how="left")
        s_node = node_data["p_node_pu"].to_numpy() + 1j*node_data["q_node_pu"].to_numpy()
        p_flow, _, v, _ = distflow.distflow_algorithm(s_node=s_node, v0_sq=v0_sq)
        
        assert np.abs(p_flow - expected_p_flow).max() < 1e-7
        assert np.abs(v - expected_v).max() < 1e-7
        
        p_node_timeseries = node_data["p_node_pu"].to_numpy() * np.random.rand(3600, 1)
        q_node_timeseries = node_data["q_node_pu"].to_numpy() * np.random.rand(3600, 1)
        distflow.timeseries_distflow_algorithm(
            p_node=p_node_timeseries, q_node=q_node_timeseries, v_ext_grid_sq=v_ext_grid_sq)
        
    def test_pandapower_comparison(self):
        s_base = 1e6
        net: pp.pandapowerNet = pp.from_pickle("data/input_grid/modified_cigre_network_lv.p")
        pp.runpp(net)
        
        ext_grid_id: str = str(net.ext_grid["bus"][0])
        v_ext_grid_sq: float = net.ext_grid["vm_pu"][0]**2
        node_data, line_data = pandapower_to_distflow(net=net, s_base=s_base)
        
        distflow = DistFlow(line_data=line_data, ext_grid_id=ext_grid_id)
        v0_sq = distflow.v_in_sq*v_ext_grid_sq
        node_data = pl.DataFrame({"node_id": distflow.node_list}).join(node_data, on="node_id", how="left")
        s_node = node_data["p_node_pu"].to_numpy() + 1j*node_data["q_node_pu"].to_numpy()
        p_flow, q_flow, v, i = distflow.distflow_algorithm(s_node=s_node, v0_sq=v0_sq)
        
        line_res_pp = pl.DataFrame({
            "node_id": list(net.line["to_bus"]),
            "p_pp": list(net.res_line["p_from_mw"]),
            "q_pp": list(net.res_line["q_from_mvar"])
        })

        result: pl.DataFrame = node_data.select(
            c("node_id").cast(pl.Int32),
            pl.Series(v).alias("v"),
            pl.Series(p_flow).alias("p"),
            pl.Series(q_flow).alias("q"),
        ).sort("node_id").with_columns(
            pl.Series(list(net.res_bus["vm_pu"])).alias("v_pp")
        ).join(line_res_pp, on="node_id", how="left").with_columns(
            (c("v") - c("v_pp")).abs().alias("diff_v"),
            (c("p") - c("p_pp")).abs().alias("diff_p"),
            (c("q") - c("q_pp")).abs().alias("diff_q")
        )
        print(i)
        assert result["diff_v"].max() < 1e-3 # type: ignore
        assert result["diff_p"].max() < 1e-4 # type: ignore
        assert result["diff_q"].max() < 1e-2 # type: ignore
        
if __name__ == "__main__":
    unittest.main()