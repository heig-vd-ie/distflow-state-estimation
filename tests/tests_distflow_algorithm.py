import unittest
import polars as pl
from polars import col as c
import numpy as np
import pandapower as pp

from data_connector import pandapower_to_distflow
from distflow_algorithm import DistFlow

def check_results(net: pp.pandapowerNet,distflow: DistFlow, s_flow: np.array, v: np.array): # type: ignore
    line_res_pp = pl.DataFrame({
            "node_id": list(net.line["to_bus"].astype(str)),
            "p_pp": list(net.res_line["p_from_mw"]),
            "q_pp": list(net.res_line["q_from_mvar"])
        })

    node_res_pp = pl.from_pandas(net.res_bus["vm_pu"],include_index=True).select(
        c("index").cast(pl.Utf8).alias("node_id"), "vm_pu"
    )

    result = pl.DataFrame({
        "v": np.sqrt(v),
        "p": np.real(s_flow),
        "q": np.imag(s_flow),
    }).with_row_index(name="node_id").with_columns(
        c("node_id").replace_strict(distflow.node_nb_to_id_mapping, default=None).alias("node_id")
    ).join(line_res_pp, on="node_id", how="left")\
    .join(node_res_pp, on="node_id", how="left")\
    .with_columns(
        ((c("v") - c("vm_pu"))/c("vm_pu")).abs().alias("diff_v"),
        ((c("p") - c("p_pp"))/c("p_pp")).abs().alias("diff_p"),
        ((c("q") - c("q_pp"))/c("p_pp")).abs().alias("diff_q")
    )
    assert result["diff_v"].max() < 4e-4 # type: ignore
    assert result["diff_p"].max() < 2e-4 # type: ignore
    assert result["diff_q"].max() < 2e-2 # type: ignore

class TestDistFlow(unittest.TestCase):

    
    def test_pandapower_comparison(self):
        s_base = 1e6
        net: pp.pandapowerNet = pp.from_pickle("data/input_grid/modified_cigre_network_lv.p")
        pp.runpp(net)
        
        ext_grid_id: str = str(net.ext_grid["bus"][0])
        v_ext_grid_sq: float = net.ext_grid["vm_pu"][0]**2
        node_data, line_data = pandapower_to_distflow(net=net, s_base=s_base)
        
        distflow = DistFlow(line_data=line_data, ext_grid_id=ext_grid_id)
        node_data = node_data.with_columns(c("node_id")
            .replace_strict(distflow.node_id_to_nb_mapping, default=None).alias("idx")).sort("idx")
        v0_sq = distflow.v_in_sq_np*v_ext_grid_sq
        s_node =node_data["p_node_pu"].to_numpy() + 1j*node_data["q_node_pu"].to_numpy()
        
        s_flow, v, _ = distflow.distflow_algorithm(s_node=s_node, v0_sq=v0_sq, engine="numpy")
        check_results(net=net,distflow=distflow, s_flow=s_flow, v=v)
        s_flow, v, _ = distflow.distflow_algorithm(s_node=s_node, v0_sq=v0_sq, engine="graphblas")
        check_results(net=net, distflow=distflow, s_flow=s_flow, v=v)
    
    def test_timeseries_simulation(self):
        s_base = 1e6
        net: pp.pandapowerNet = pp.from_pickle("data/input_grid/modified_cigre_network_lv.p")
    
        ext_grid_id: str = str(net.ext_grid["bus"][0])
        v_ext_grid_sq: float = net.ext_grid["vm_pu"][0]**2
        node_data, line_data = pandapower_to_distflow(net=net, s_base=s_base)
        
        distflow = DistFlow(line_data=line_data, ext_grid_id=ext_grid_id)
        node_data = node_data.with_columns(c("node_id")
            .replace_strict(distflow.node_id_to_nb_mapping, default=None).alias("idx")).sort("idx")
        s_node =node_data["p_node_pu"].to_numpy() + 1j*node_data["q_node_pu"].to_numpy()
        
        s_node = np.multiply(s_node.reshape(-1, 1), np.random.random(20) + 0.5).transpose()
        s_flow, v, _ = distflow.timeseries_distflow_algorithm(s_node=s_node, v_ext_grid_sq=v_ext_grid_sq, engine="numpy")
        s_flow, v, _ = distflow.timeseries_distflow_algorithm(s_node=s_node, v_ext_grid_sq=v_ext_grid_sq, engine="graphblas")
        
if __name__ == "__main__":
    unittest.main()