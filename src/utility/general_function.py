# """
# Auxiliary functions
# """
# # import dotenv
# from typing import Optional
# import logging
# import os
# import uuid
# import coloredlogs
# import polars as pl
# from polars import col as c

# import re 
# import owncloud
# import tqdm
# import duckdb
# import pandas as pd
# import geopandas as gpd

# from twindigrid_sql.entries import NAMESPACE_UUID

# from shapely import Geometry, LineString, from_wkt, intersection, distance
# from shapely.ops import nearest_points
# from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point

# from geoalchemy2.shape import from_shape, to_shape
# from geoalchemy2.elements import WKBElement
# from config import settings



# def scan_switch_directory(oc: owncloud.Client, local_folder_path: str, switch_folder_path: str, download_anyway: bool) -> list[str]:
#     file_list = []
#     build_non_existing_dirs(os.path.join(local_folder_path, switch_folder_path))
#     for file_data in oc.list(switch_folder_path): # type: ignore
#         file_path: str = file_data.path
#         if file_data.file_type == "dir":
#             file_list.extend(scan_switch_directory(oc=oc, local_folder_path=local_folder_path, switch_folder_path=file_path[1:], download_anyway=download_anyway))
#         else:
#             if (not os.path.exists(local_folder_path + file_path)) | download_anyway:
#                 file_list.append(file_path)
#     return file_list

# def download_from_switch(switch_folder_path: str, local_folder_path: str= ".cache", download_anyway: bool = False):
#     oc: owncloud.Client = owncloud.Client.from_public_link(settings.SWITCH_LINK, folder_password=settings.SWITCH_PASS)
#     with tqdm.tqdm(total = 1, desc=f"Scan {switch_folder_path} Switch remote directory", ncols=120) as pbar:
#         file_list: list[str] = scan_switch_directory(
#             oc=oc, local_folder_path=local_folder_path, switch_folder_path=switch_folder_path, download_anyway=download_anyway)
#         pbar.update()
#     for file_path in tqdm.tqdm(file_list, desc= f"Download files from {switch_folder_path} Switch remote directory ", ncols=120):
#         oc.get_file(file_path, local_folder_path + file_path)
        

# def generate_log(name: str) -> logging.Logger:
#     """
#     load configs of environment
#     :return: log for logging, config is list of all env vars
#     """
#     log = logging.getLogger(name)
#     coloredlogs.install(level="info")
#     return log


# def build_non_existing_dirs(file_path: str):
#     """
#     build non existing directories
#     :param file_path:
#     :return: True
#     """
#     file_path = os.path.normpath(file_path)
#     # Split the path into individual directories
#     dirs = file_path.split(os.sep)
#     # Check if each directory exists and create it if it doesn't
#     current_path = ""
#     for directory in dirs:
#         current_path = os.path.join(current_path, directory)
#         if not os.path.exists(current_path):
#             os.mkdir(current_path)
#     return True


# def pl_to_dict(df: pl.DataFrame) -> dict:
#     """
#     Convert a Polars DataFrame with two columns into a dictionary.

#     Args:
#         df (pl.DataFrame): Polars DataFrame with two columns.

#     Returns:
#         dict: Dictionary representation of the DataFrame.
#     """
#     if df.shape[1] != 2:
#         raise ValueError("DataFrame is not composed of two columns")

#     columns_name = df.columns[0]
#     df = df.drop_nulls(columns_name)
#     if df[columns_name].is_duplicated().sum() != 0:
#         raise ValueError("Key values are not unique")
#     return dict(df.to_numpy())

# def modify_string(string: str, format_str: dict) -> str:
#     """
#     Modify a string by replacing substrings according to a format dictionary.

#     Args:
#         string (str): Input string.
#         format_str (dict): Dictionary containing the substrings to be replaced and their replacements.

#     Returns:
#         str: Modified string.
#     """

#     for str_in, str_out in format_str.items():
#         string = re.sub(str_in, str_out, string)
#     return string

# def camel_to_snake(s):
#     return (
#         ''.join(
#             [ '_'+ c.lower() if c.isupper() else c for c in s ]
#         ).lstrip('_')
#     )

# def snake_to_camel(snake_str):
#     return "".join(x.capitalize() for x in snake_str.lower().split("_"))

# def geoalchemy2_to_shape(geo_str: str) -> Geometry:
#     return to_shape(WKBElement(str(geo_str)))

# def shape_to_geoalchemy2(geo: Geometry) -> str:
#     if isinstance(geo, Geometry):
#         return from_shape(geo, srid=settings.GPS_SRID).desc
#     return None

# def duckdb_to_dict(file_path: str) -> dict[str, pl.DataFrame]:
#     data: dict[str, pl.DataFrame] = {}
#     pbar = tqdm.tqdm(
#         total=1, ncols=150, 
#         desc="Read tables from {} file".format(os.path.basename(file_path))
#     )
#     with pbar:
#         with duckdb.connect(database=file_path) as con:
#             query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
#             for table_name in con.execute(query).fetchall():
#                 query: str = f"SELECT * FROM {table_name[0]}"
#                 data[table_name[0]] = con.execute(query).pl()
#         pbar.update()
#     return data


# def convert_list_to_string(list_data)-> str:
#     return ", ".join(map(str, list_data))

# def table_to_gpkg(table: pl.DataFrame, gpkg_file_name: str, layer_name: str, ):
#     list_columns: list[str] = [name for name, col_type in dict(table.schema).items() if type(col_type) == pl.List]
#     table_pd: pd.DataFrame = table.with_columns(
#         c(name).map_elements(convert_list_to_string, return_dtype=pl.Utf8) for name in list_columns
#     ).to_pandas()

#     table_pd["geometry"] = table_pd["geometry"].apply(from_wkt)
#     table_pd = table_pd[table_pd.geometry.notnull()]
#     table_gpd: gpd.GeoDataFrame = gpd.GeoDataFrame(
#         table_pd.dropna(axis=0, subset="geometry"), crs=settings.SWISS_SRID) # type: ignore
#     table_gpd = table_gpd[~table_gpd["geometry"].is_empty] # type: ignore
#     table_gpd.to_file(gpkg_file_name, layer=layer_name) 


# def dict_to_gpkg(data: dict, file_path:str):

#     with tqdm.tqdm(range(1), ncols=100, desc="Save input data in gpkg format") as pbar:
#         for layer_name, table in data.items():
#             if isinstance(table, pl.DataFrame):
#                 if not table.is_empty():
#                     table_to_gpkg(table=table, gpkg_file_name=file_path, layer_name=layer_name)
#         pbar.update()


# def get_polygon_list(geo_shape: Geometry) -> list[str]:
#     if isinstance(geo_shape, MultiPolygon):
#         return list(map(lambda x: x.wkt, geo_shape.geoms))
#     elif isinstance(geo_shape, Polygon):
#         return [geo_shape.wkt]
#     return []

# def point_list_to_linestring(point_list_str: list[str]) -> str:
#     return LineString(list(map(from_wkt, point_list_str))).wkt



# def get_polygon_multipoint_intersection(polygon_str: str, multipoint: MultiPoint) -> Optional[list[str]]:
#     point_shape: Geometry = intersection(from_wkt(polygon_str), multipoint)
    
#     if isinstance(point_shape, MultiPoint):
#         return list(map(lambda x: x.wkt, point_shape.geoms))
#     if isinstance(point_shape, Point):
#         if point_shape.is_empty:
#             return []
#         return [point_shape.wkt]
#     return []

# def find_closest_node(data: dict, node_id_geo_mapping: dict) -> Optional[str]:
#     if data["node_id"] is None:
#         return None  
#     if len(data["node_id"]) == 1:
#         return data["node_id"][0] 
#     if len(data["node_id"]) > 1:
#         return min(data["node_id"], key=lambda x: from_wkt(data["geometry"]).distance(from_wkt(node_id_geo_mapping[x])))
#     return None

# def filter_unique_nodes_from_list(node_id_list: pl.Expr)-> pl.Expr:
    
#     return (
#         pl.when(node_id_list.list.len() == 1)
#         .then(node_id_list.list.get(0, null_on_oob=True))
#         .otherwise(pl.lit(None))
#     )

# def explode_multipolygon(geometry_str: str) -> list[Polygon]:
#     geometry_shape: Geometry = from_wkt(geometry_str)
#     if isinstance(geometry_shape, Polygon):
#         return [geometry_shape]
#     if isinstance(geometry_shape, MultiPolygon):
#         return list(geometry_shape.geoms)
#     return []


# def get_geo_multipoints(data_df: pl.DataFrame, column_name: str = "geometry") -> MultiPoint:
#     return MultiPoint(
#         data_df.with_columns(
#             c(column_name).map_elements(from_wkt, return_dtype=pl.Object)
#         )[column_name].to_list()
#     )

# def dictionary_key_filtering(dictionary: dict, key_list: list) -> dict:
#     return dict(filter(lambda x : x[0] in key_list, dictionary.items()))

# def get_closest_point(geo_str: str, multi_point: MultiPoint, max_distance: float=100) -> Optional[str]:
#     geo = from_wkt(geo_str)
#     _, closest_point = nearest_points(geo, multi_point)
#     if distance(geo, closest_point) < max_distance:
#         return closest_point.wkt
#     return None


# def generate_uuid(base_value: str, base_uuid: uuid.UUID | None = None, added_string: str = "") -> str:
#     """
#     Generate a UUID based on a base value, base UUID, and an optional added string.

#     Args:
#         base_value (str): The base value for generating the UUID.
#         base_uuid (str): The base UUID for generating the UUID.
#         added_string (str, optional): The optional added string. Defaults to "".

#     Returns:
#         str: The generated UUID.
#     """
#     if base_uuid is None:
#         base_uuid=NAMESPACE_UUID
#     return str(uuid.uuid5(base_uuid, added_string + base_value))

