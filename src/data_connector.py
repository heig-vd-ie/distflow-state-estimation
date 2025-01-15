import polars as pl
from polars import col as c
from typing_extensions import Union
import numpy as np
import pandapower as pp
from polars_function import (
    get_transfo_admittance, get_transfo_impedance, get_transfo_conductance, get_transfo_imaginary_component)

def pandapower_to_distflow(net: pp.pandapowerNet, s_base: float=1e6) -> tuple[pl.DataFrame, pl.DataFrame]:
    node_data: pl.DataFrame = pl.from_pandas(net.bus)
    load: pl.DataFrame = pl.from_pandas(net.load)
    pv: pl.DataFrame = pl.from_pandas(net.sgen)

    pv = pv.group_by("bus").agg(
        (-c("p_mw").sum()*1e6/s_base).alias("p_pv"),
        (-c("q_mvar").sum()*1e6/s_base).alias("q_pv")
    )

    load = load.group_by("bus").agg(
        (c("p_mw").sum()*1e6/s_base).alias("p_load"),
        (c("q_mvar").sum()*1e6/s_base).alias("q_load")
    )

    load = load.join(pv, on="bus", how="full", coalesce=True).select(
        c("bus").alias("node_id"),
        pl.sum_horizontal([c("p_load").fill_null(0.), c("p_pv").fill_null(0.)]).alias("p_node_pu"),
        pl.sum_horizontal([c("q_load").fill_null(0.), c("q_pv").fill_null(0.)]).alias("q_node_pu")
    )

    node_data = node_data[["vn_kv"]].with_row_index(name="node_id")\
        .join(load, on="node_id", how="left")\
        .select(
            c("node_id").cast(pl.Utf8),
            (c("vn_kv")*1e3).alias("v_base"),
            c("p_node_pu").fill_null(0.0),
            c("q_node_pu").fill_null(0.0),
        )
    line: pl.DataFrame = pl.from_pandas(net.line)
    line = line\
        .with_columns(
            c("from_bus").cast(pl.Utf8).alias("u_of_edge"),
            c("to_bus").cast(pl.Utf8).alias("v_of_edge"),
        ).join(node_data["node_id", "v_base"], left_on="u_of_edge", right_on="node_id", how="left")\
        .with_columns(
            (c("v_base")**2 /s_base).alias("z_base"), 
        ).select(
            "u_of_edge", "v_of_edge", "name",
            (c("r_ohm_per_km")*c("length_km")/c("z_base")).alias("r_pu"),
            (c("x_ohm_per_km")*c("length_km")/c("z_base")).alias("x_pu"),
            (-c("c_nf_per_km")*c("length_km")*1e-9*2*np.pi*50/c("z_base")).alias("b_pu"),
            pl.lit("branch").alias("type"), 
        )

    trafo: pl.DataFrame = pl.from_pandas(net.trafo)
    trafo = trafo.with_columns(
        c("hv_bus").cast(pl.Utf8).alias("u_of_edge"),
        c("lv_bus").cast(pl.Utf8).alias("v_of_edge"),
    ).join(node_data.select("node_id", c("v_base").alias("v_base1")), left_on="u_of_edge", right_on="node_id", how="left")\
    .join(node_data.select("node_id", c("v_base").alias("v_base2")), left_on="v_of_edge", right_on="node_id", how="left")\
    .with_columns(
        (c("v_base2")**2 / s_base).alias("z_base_grid"), 
        (c("v_base2")/c("v_base1")).alias("n_grid"),
        ((c("vn_lv_kv")/c("vn_hv_kv"))).alias("n_trafo"),
    ).with_columns(
        get_transfo_impedance(rated_s=c("sn_mva")*1e6, rated_v=c("vn_lv_kv")*1e3, voltage_ratio=c("vk_percent")).alias("z"),
        get_transfo_impedance(rated_s=c("sn_mva")*1e6, rated_v=c("vn_lv_kv")*1e3, voltage_ratio=c("vkr_percent")).alias("r"),
        get_transfo_admittance(rated_s=c("sn_mva")*1e6, rated_v=c("vn_lv_kv")*1e3, oc_current_ratio=c("i0_percent")).alias("y"),
        get_transfo_conductance(rated_v=c("vn_lv_kv")*1e3, iron_losses=c("pfe_kw")*1e3).alias("g"),
    ).with_columns(
        get_transfo_imaginary_component(module = c("z"), real = c("r")).alias("x"),
        get_transfo_imaginary_component(module = c("y"), real = c("g")).alias("b"),
    ).select(
        "u_of_edge", "v_of_edge", "name",
        (c("r")/c("z_base_grid")).alias("r_pu"),
        (c("x")/c("z_base_grid")).alias("x_pu"),
        (c("g")*c("z_base_grid")).alias("g_pu"),
        (c("b")*c("z_base_grid")).alias("b_pu"),
        pl.lit("transformer").alias("type"),
        ((c("n_trafo")/c("n_grid"))**2).alias("n_transfo"),
    )

    switch: pl.DataFrame = pl.from_pandas(net.switch)

    switch = switch.select(
        c("name"),
        c("bus").cast(pl.Utf8).alias("u_of_edge"),
        c("element").cast(pl.Utf8).alias("v_of_edge"),
        pl.lit("switch").alias("type"),
    )

    null_val_mapping: dict[str, Union[int, float]] = {"r_pu": 0,	"x_pu": 0, "b_pu": 0, "g_pu": 0, "n_transfo": 1.0}

    line_data = pl.concat(
        [line, trafo, switch], how="diagonal_relaxed"
    ).with_columns(
        c(name).fill_null(val)
        for name, val in null_val_mapping.items()
    )
    
    return node_data, line_data