import os
from typing import Optional
import tqdm

import polars as pl
import polars.selectors as cs
from polars import col as c
import patito as pt
import pandas as pd
import duckdb

from twindigrid_sql.entries.abstraction import PHYSICAL
from twindigrid_sql.entries.equipment_class import TRANSFORMER, BRANCH, SWITCH, INDIRECT_FEEDER, BUSBAR_SECTION, ENERGY_CONSUMER
from twindigrid_sql.entries.source import SCADA
from twindigrid_sql.schema.enum import TerminalSide, Diff,TerminalPhases
from twindigrid_changes import models as changes_models
from twindigrid_changes.schema import ChangesSchema

from config import settings

from utility.general_function import build_non_existing_dirs, generate_log, camel_to_snake, pl_to_dict, snake_to_camel, dict_to_gpkg, table_to_gpkg
from utility.polars_operation import (
    generate_random_uuid, point_list_to_linestring_col, geoalchemy2_to_wkt)

log = generate_log(name=__name__)

def get_cn_fk_mapping(
        changes_schema: ChangesSchema, eq_class: str| list[str],  side: Optional[str] = None
    ) -> dict[str, str]:

    if isinstance(eq_class, str):
        eq_class = [eq_class]
    if side is None:
        side = TerminalSide.T1.value

    connectivity = changes_schema.connectivity\
        .filter(c("eq_class").is_in(eq_class))\
        .filter(c("side") == side)[["eq_fk", "cn_fk"]]

    return pl_to_dict(
        changes_schema.resource.filter(c("concrete_class").is_in(eq_class))
        .join(connectivity, left_on="uuid", right_on="eq_fk", how="left")[["dso_code", "cn_fk"]]
    )

def get_cn_fk_parent_fk_mapping(changes_schema: ChangesSchema) -> dict[str, str]:
    parent_fk_mapping: dict[str, str] = pl_to_dict(changes_schema.container[["uuid", "parent_fk"]])
    return  pl_to_dict(
        changes_schema.connectivity
            .with_columns(
                c("container_fk").replace(parent_fk_mapping, default=c("container_fk"), return_dtype=pl.Utf8)
                .alias("parent_fk")
            ).drop_nulls("container_fk")
            .filter(c("cn_fk").is_first_distinct())[["cn_fk", "parent_fk"]]
    )

def get_bay_fk_mapping(
        changes_schema: ChangesSchema, side: Optional[str] = None, fk_column: str = "cn_fk") -> dict[str, str]:
    if side is None:
        side = TerminalSide.T1.value
    connectivity = changes_schema.connectivity\
            .filter(c("eq_class") == SWITCH)\
            .filter(c("side") == side)[["container_fk", fk_column]]
    return pl_to_dict(
        changes_schema.resource\
            .filter(c("concrete_class")=="bay")\
            .join(connectivity, left_on="uuid", right_on="container_fk", how="left", coalesce=True)[["dso_code", fk_column]]
    )

def generate_connectivity_table(
        changes_schema: ChangesSchema, eq_table: pl.DataFrame, raw_data_table: str
    ) -> ChangesSchema:

    if eq_table.is_empty():
        log.warning(f"Empty connectivity table {raw_data_table}")
        return changes_schema
    terminal_names: str = "^(" + "|".join([e.value for e in TerminalSide]) + ")$"

    eq_name: str = eq_table["concrete_class"][0]
    eq_table = eq_table.drop_nulls("uuid")\
        .with_columns(
            (c("uuid") if "eq_fk" not in eq_table.columns else c("eq_fk")).alias("eq_fk"),
            (pl.lit(TerminalPhases.ABCn.value) if "phases" not in eq_table.columns else c("phases")).alias("phases"),
            (pl.lit(False) if "indirect" not in eq_table.columns else c("indirect")).alias("indirect"),
            (pl.lit(PHYSICAL) if "abstraction_fk" not in eq_table.columns else c("abstraction_fk")).alias("abstraction_fk"),
            c("concrete_class").alias("eq_class")
        )
    if eq_name == TRANSFORMER:
        connectivity: pl.DataFrame = eq_table
    elif eq_name in [BRANCH, INDIRECT_FEEDER]:
        connectivity: pl.DataFrame = eq_table.with_columns(
            pl.struct(TerminalSide.T1.value, "t1_container_fk")
            .struct.rename_fields(["cn_fk", "container_fk"]).alias(TerminalSide.T1.value),
            pl.struct(TerminalSide.T2.value, "t2_container_fk")
            .struct.rename_fields(["cn_fk", "container_fk"]).alias(TerminalSide.T2.value),
        ).melt(
            id_vars=cs.all() - cs.matches(terminal_names), value_vars=cs.matches(terminal_names), 
            variable_name="side", value_name="cn_fk"
        ).unnest("cn_fk")
    else:
        connectivity: pl.DataFrame = eq_table\
            .melt(
                id_vars=cs.all() - cs.matches(terminal_names), value_vars=cs.matches(terminal_names), 
                variable_name="side", value_name="cn_fk"
            )
    check_null_values(
        df=connectivity, columns="cn_fk", raw_table_name=raw_data_table)

    connectivity = connectivity\
        .with_columns(
            c("eq_fk").pipe(generate_random_uuid).alias("uuid"), # create random uuid for connectivity
            c("cn_fk").fill_null(c("cn_fk").pipe(generate_random_uuid)).alias("cn_fk") # create fictive connectivity node for null values
        )


    new_tables_pl: dict[str, pl.DataFrame] = {
        "Connectivity": connectivity, "Terminal": connectivity
    }

    changes_schema = add_table_to_changes_schema(
        schema=changes_schema, new_tables_pl=new_tables_pl,
        raw_table_name=raw_data_table)
    
    return changes_schema

def add_table_to_changes_schema(
        schema: ChangesSchema, new_tables_pl: dict[str, pl.DataFrame], raw_table_name: str
    ) -> ChangesSchema:
    changes_schema = schema
    new_tables_pt: dict[str, pt.Model] = {}
    
    for table_class_name, new_table in new_tables_pl.items():

        if not new_table.is_empty():
            if table_class_name == "Tap":
                primary_key: list[str] = ["eq_fk", "side", "value"]
            elif table_class_name  in ["TransformerEnd", "Terminal"]:
                primary_key = ["eq_fk", "side"]
            elif table_class_name == "Heartbeat":
                primary_key = ["heartbeat"]
            elif table_class_name == "BaseVoltage":
                primary_key = ["nominal_voltage"]
            elif table_class_name == "Connectivity":
                primary_key = ["eq_fk", "side", "abstraction_fk", "diff"]
            else:
                primary_key = ["uuid"]

            table_name: str = camel_to_snake(table_class_name)
            old_table: pl.DataFrame = pl.DataFrame(getattr(changes_schema, table_name))
        
            intersect_column: list[str] = list(set(old_table.columns).intersection(new_table.columns)) 
            new_table = new_table[intersect_column].with_columns(
                pl.lit(Diff.CREATED.value).alias("diff") if "diff" not in new_table.columns else c("diff")
            )

            check_double_values(
                df=new_table, raw_table_name=raw_table_name + "_" + table_name, 
                columns=primary_key, combined_columns=True)
            
            table: pl.DataFrame = pl.concat([old_table, new_table], how="diagonal_relaxed")\
                .filter(pl.struct(primary_key).is_first_distinct())
            new_tables_pt[table_name]: pt.Model = ( # type: ignore
                pt.Model.DataFrame(table) 
                .set_model(getattr(changes_models, table_class_name))
            ).cast() 
    return changes_schema.replace(**new_tables_pt) # type: ignore

def check_null_values(
        df: pl.DataFrame, columns: str | list[str], raw_table_name: str
    ):
    if isinstance(columns, str):
        column_list: list[str] = [columns]
    elif isinstance(columns, list):
        column_list: list[str] = columns
    else:
        raise ValueError("columns is not a string or a list of string")

    null_df: pl.DataFrame = df.filter(pl.any_horizontal(c(column_list).is_null()))
    if not null_df.is_empty():
        error_message = f"{null_df.height} Null values over {df.height} detected in {raw_table_name} for columns {" ".join(column_list)}"
        log.warning(error_message)


def check_double_values(
        df: pl.DataFrame, columns: str | list[str], raw_table_name: str, combined_columns: bool = False):

    if isinstance(columns, str):
        column_list: list[str] = [columns]
    elif isinstance(columns, list):
        column_list: list[str] = columns
    else:
        raise ValueError("columns is not a string or a list of string")
    if combined_columns:
        duplicated_df: pl.DataFrame = df.filter(pl.struct(column_list).is_duplicated())
    else:    
        duplicated_df: pl.DataFrame = df.filter(pl.any_horizontal(pl.col(col).is_duplicated() for col in column_list))

    if not duplicated_df.is_empty():
        error_message = f"{duplicated_df.height} double values over {df.height} detected in {raw_table_name} for columns {" ".join(column_list)}"
        log.warning(error_message)

def log_error(
        df: pl.DataFrame, sheet_name: str, log_file_name:str, error_message: str, 
        add_column: Optional[list[str]] = None, save_log: bool = True
    ):
    log.warning(error_message)
    if save_log:
        sheet_name = sheet_name[:30]
        log_file = os.path.join(settings.LOG_FOLDER, log_file_name)
        log_columns: list[str]= list(set(df.columns).intersection(set(settings.LOG_COLUMN))) + (add_column if add_column else [])
        with pd.ExcelWriter(log_file, engine='openpyxl',  mode='a' if os.path.exists(log_file) else "w") as writer:   
            df[log_columns].to_pandas().to_excel(writer, sheet_name=sheet_name, index=False) 
            
def initialize_output_files(file_path: str):
    build_non_existing_dirs(file_path=os.path.dirname(file_path))
    if os.path.exists(file_path):
        os.remove(file_path)
    
def changes_schema_to_duckdb(changes_schema: ChangesSchema, file_path: str):
    
    initialize_output_files(file_path)
    with duckdb.connect(file_path) as con:
        for table_name, table_pl in tqdm.tqdm(changes_schema.__dict__.items(), desc="Save change schema into duckdb file", ncols=150):
            query = f"CREATE TABLE {table_name} AS SELECT * FROM table_pl"
            con.execute(query)

def duckdb_to_changes_schema(file_path: str) -> ChangesSchema:
    schema_dict: dict[str, pt.Model.DataFrame] = {} # type: ignore
    pbar = tqdm.tqdm(
        total=1, ncols=150, desc="Read and validate tables from {} file".format(os.path.basename(file_path)))
    
    with pbar:
        with duckdb.connect(database=file_path) as con:
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            for table_name in con.execute(query).fetchall():
                query: str = f"SELECT * FROM {table_name[0]}"
                pt_model: pt.Model = getattr(changes_models, snake_to_camel(table_name[0]))
                schema_dict[table_name[0]] = pt.Model.DataFrame(con.execute(query).pl()).set_model(pt_model).cast()
        new_schema = ChangesSchema().replace(**schema_dict)
        pbar.update()
    return new_schema


def get_base_voltage_mapping(
        changes_schema: ChangesSchema, eq_list: list[str] = [SWITCH, BUSBAR_SECTION, BRANCH]) -> dict[str, int]:

    connectivity: pl.DataFrame = changes_schema.connectivity\
        .filter(c("eq_class").is_in(eq_list))[["eq_fk", "cn_fk"]]\
        .unique(subset=["eq_fk"])

    return pl_to_dict(
        changes_schema.connectivity_node.join(connectivity, left_on="uuid", right_on="cn_fk", how="inner")
        [["eq_fk", "base_voltage_fk"]]
    )


def schema_to_gpkg(
        changes_schema: ChangesSchema, file_path: str, add_building_conn: bool = True, 
        get_feeder_name: bool = True):
    
    initialize_output_files(file_path)
    base_voltage_mapping: dict[str, int] = get_base_voltage_mapping(changes_schema=changes_schema)
    geo: pl.DataFrame = changes_schema.geo_event.select(
        c("geo").pipe(geoalchemy2_to_wkt).alias("geometry"),
        c("res_fk").alias("uuid")
    )

    connectivity : pl.DataFrame = changes_schema.connectivity
    resource: pl.DataFrame  = geo.join(
            changes_schema.resource[["uuid", "concrete_class", "dso_code", "name", "feeder_fk", "metadata"]], 
            on="uuid", how="left"
        ).with_columns(
            c("uuid").replace_strict(base_voltage_mapping, default = None).alias("base_voltage_fk"),
        )
    
    if get_feeder_name:
        feeder: pl.DataFrame = changes_schema.measurement.filter(c("source_fk") == SCADA).select(
            pl.concat_str("resource_fk", "terminal_side").alias("id"), "name"
        )

        feeder_name_mapping = pl_to_dict(
            feeder.join(
                connectivity.select("cn_fk", pl.concat_str("eq_fk", "side").alias("id")), on="id", how="left"
            )[["cn_fk", "name"]])
        
        resource = resource.with_columns(
            c("feeder_fk").replace_strict(feeder_name_mapping, default=None).alias("feeder_name")
        )

    grid_dict: dict[str, pl.DataFrame] = {}
    for concrete_class in resource["concrete_class"].unique().to_list():
        grid_dict[concrete_class] = getattr(changes_schema, concrete_class).join(resource, on="uuid", how="inner")
        if concrete_class == BRANCH:
            grid_dict[concrete_class] = grid_dict[concrete_class].join(
                changes_schema.branch_parameter_event.drop(["diff", "uuid", "timestamp", "heartbeat", "source_fk"]),
                left_on="uuid", right_on="eq_fk", how="left", coalesce=True
            )
        if concrete_class == SWITCH:
            grid_dict[concrete_class] = grid_dict[concrete_class].join(
                changes_schema.switch_event[["eq_fk", "open"]],
                left_on="uuid", right_on="eq_fk", how="left", coalesce=True
            )
    dict_to_gpkg(grid_dict, file_path=file_path)
    if add_building_conn:
        building_connection: pl.DataFrame = generate_building_connection(
            connectivity=connectivity, resource=resource)
        table_to_gpkg(table=building_connection, gpkg_file_name=file_path, layer_name="building_connection")

def generate_building_connection(connectivity: pl.DataFrame, resource: pl.DataFrame) -> pl.DataFrame:
    energy_consumer = resource\
        .filter(c("concrete_class") == ENERGY_CONSUMER)\
        .join(connectivity[["eq_fk", "container_fk"]], left_on="uuid", right_on="eq_fk", how="left")\
        .join(
            resource.select("uuid", c("geometry").alias("container_geo")), 
            left_on="container_fk", right_on="uuid", how="left")

    building_connection: pl.DataFrame = energy_consumer.drop_nulls("container_fk")\
        .select(
        "name",
        pl.concat_list(["geometry", "container_geo"])
        .pipe(point_list_to_linestring_col).alias("geometry"),
        )
    return building_connection


def add_building_connection(changes_schema, file_path: str):
    connectivity: pl.DataFrame = changes_schema.connectivity
    geo: pl.DataFrame = changes_schema.geo_event
    resource: pl.DataFrame = changes_schema.resource[["uuid", "concrete_class", "dso_code", "name", "feeder_fk", "metadata"]] 
    geo = geo.select(
            c("geo").pipe(geoalchemy2_to_wkt).alias("geometry"),
            c("res_fk").alias("uuid")
        )
    resource: pl.DataFrame  = geo.join(resource, on="uuid", how="left")

    energy_consumer = resource\
        .filter(c("concrete_class") == ENERGY_CONSUMER)\
        .join(connectivity[["eq_fk", "container_fk"]], left_on="uuid", right_on="eq_fk", how="left")\
        .join(geo.rename({"geometry":"container_geo"}), left_on="container_fk", right_on="uuid", how="left")

    building_connection: pl.DataFrame = energy_consumer.drop_nulls("container_fk")\
        .select(
        pl.concat_list(["geometry", "container_geo"])
                .pipe(point_list_to_linestring_col).alias("geometry"),
        "name"
    )
    table_to_gpkg(table=building_connection, gpkg_file_name=file_path, layer_name="building_connection")
    
