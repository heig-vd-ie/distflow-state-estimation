import re
import uuid
import json
from datetime import timedelta, datetime
from math import prod
from typing import Optional

import numpy as np
import polars as pl
from polars import col as c
from itertools import batched

from pyproj import CRS, Transformer
from shapely.ops import substring, transform
from shapely.geometry import Polygon, Point
from shapely import Geometry, LineString, union_all, from_wkt

from twindigrid_sql.entries import NAMESPACE_UUID
from twindigrid_sql.schema.enum import TerminalSide, ConnectionKind, Owner

from utility.general_function import (
    modify_string, generate_log, geoalchemy2_to_shape, shape_to_geoalchemy2, point_list_to_linestring, generate_uuid)

from config import settings

# Global variable
log = generate_log(name=__name__)

def generate_uuid_col(col: pl.Expr, base_uuid: uuid.UUID  | None = None, added_string: str = "") -> pl.Expr:
    """
    Generate UUIDs for a column based on a base UUID and an optional added string.

    Args:
        col (pl.Expr): The column to generate UUIDs for.
        base_uuid (str): The base UUID for generating the UUIDs.
        added_string (str, optional): The optional added string. Defaults to "".

    Returns:
        pl.Expr: The column with generated UUIDs.
    """

    return (
        col.cast(pl.Utf8)
        .map_elements(lambda x: generate_uuid(base_value=x, base_uuid=base_uuid, added_string=added_string), pl.Utf8)
    )

def cast_float(float_str: pl.Expr) -> pl.Expr:
    format_str = {r'^,': "0.", ',': "."}
    return float_str.pipe(modify_string_col, format_str=format_str).cast(pl.Float64)

def cast_boolean(col: pl.Expr) -> pl.Expr:
    """
    Cast a column to boolean based on predefined replacements.

    Args:
        col (pl.Expr): The column to cast.

    Returns:
        pl.Expr: The casted boolean column.
    """
    format_str = {
        "1": True, "true": True , "oui": True, "0": False, "false": False, "vrai": True, "non": False, 
        "off": False, "on": True}
    return col.str.to_lowercase().replace(format_str, default=False).cast(pl.Boolean)

def modify_string_col(string_col: pl.Expr, format_str: dict) -> pl.Expr:
    """
    Modify string columns based on a given format dictionary.

    Args:
        string_col (pl.Expr): The string column to modify.
        format_str (dict): The format dictionary containing the string modifications.

    Returns:
        pl.Expr: The modified string column.
    """
    return string_col.map_elements(lambda x: modify_string(string=x, format_str=format_str), return_dtype=pl.Utf8, skip_nulls=True)

def parse_date(date_str: str|None, default_date: datetime) -> datetime:
    """
    Parse a date string and return a datetime object.

    Args:
        date_str (str|None): The date string to parse.
        default_date (datetime): The default date to return if the date string is None.

    Returns:
        datetime: The parsed datetime object.
    
    Raises:
        ValueError: If the date format is not recognized.
    """
    if date_str is None:
        return default_date
    if bool(re.match(r"[0-9]{5}", date_str)):
        return  datetime(1899, 12, 30) + timedelta(days=int(date_str))

    format_str: dict[str, str] = {r"[-:.//]": "_"}
    date_str = modify_string(date_str, format_str)
    if bool(re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}", date_str)):
        return datetime.strptime(date_str, '%Y_%m_%d')
    if bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{4}", date_str)):
        return datetime.strptime(date_str, '%d_%m_%Y')
    
    raise ValueError("Date format not recognized")


def parse_timestamp(
        timestamp_str: pl.Expr, item: Optional[str],  keep_string_format: bool= False, convert_to_utc: bool = False
    ) -> pl.Expr:
    """
    Parse a timestamp column based on a given item.

    Args:
        timestamp (pl.Expr): The timestamp column.
        item (str): The item to parse.

    Returns:
        pl.Expr: The parsed timestamp column.
    
    Raises:
        ValueError: If the timestamp format is not recognized.
    """
    format_str: dict[str, str] = {r"[-:\.//]": "_"}
    
    if item is None:
        return pl.lit(None)
    item = modify_string(item, format_str)
    if bool(re.match(r"[0-9]{5}", item)):
        timestamp: pl.Expr =  (3.6e6*24*timestamp_str.cast(pl.Int32)).cast(pl.Duration("ms")) +  datetime(1899, 12, 30)
    else:
        if bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{4}\s[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{3}", item)):
            format_str: dict[str, str] = { r"[-:.//]": "_", r"_[0-9]{3}$": ""}
            format_timestamp: str = "%d_%m_%Y %H_%M_%S"
        elif bool(re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}\s[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{3}", item)):
            format_str = { r"[-:.//]": "_", r"_[0-9]{3}$": ""}
            format_timestamp: str = "%Y_%m_%d %H_%M_%S"
        elif bool(re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}\s[0-9]{2}_[0-9]{2}_[0-9]{2}", item)):
            format_timestamp: str = "%Y_%m_%d %H_%M_%S"
        elif bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{4}\s[0-9]{2}_[0-9]{2}_[0-9]{2}", item)):
            format_timestamp: str ="%d_%m_%Y %H_%M_%S"
        elif bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{2}\s[0-9]{2}_[0-9]{2}_[0-9]{2}", item)):
            format_timestamp: str ="%d_%m_%y %H_%M_%S"
        elif bool(re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}", item)):
            timestamp_str = timestamp_str + " 00_00_00"
            format_timestamp: str ="%Y_%m_%d %H_%M_%S"
        elif bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{4}", item)):
            timestamp_str = timestamp_str + " 00_00_00"
            format_timestamp: str ="%d_%m_%Y %H_%M_%S"
        else:
            raise ValueError("Timestamp format not recognized")
    
        timestamp: pl.Expr = modify_string_col(timestamp_str, format_str).str.strptime(pl.Datetime, format_timestamp)   

    if keep_string_format:
        return timestamp.dt.strftime("%Y/%m/%d %H:%M:%S")
    elif convert_to_utc:
        return timestamp.dt.replace_time_zone("UTC", ambiguous='earliest').dt.cast_time_unit(time_unit="us")
    return timestamp.dt.cast_time_unit(time_unit="us")

def cast_to_utc_timestamp(timestamp: pl.Expr, first_occurrence: pl.Expr) -> pl.Expr:
    return (
        pl.when(first_occurrence)
        .then(timestamp.dt.replace_time_zone("Europe/Zurich", ambiguous='earliest'))
        .otherwise(timestamp.dt.replace_time_zone("Europe/Zurich", ambiguous='latest'))
        .dt.convert_time_zone("UTC")
    )

def generate_random_uuid(col: pl.Expr) -> pl.Expr:
    """
    Generate a random UUID.

    Returns:
        str: The generated UUID.
    """
    return col.map_elements(lambda x: str(uuid.uuid4()), return_dtype=pl.Utf8, skip_nulls=False)

def get_transfo_impedance(rated_v: pl.Expr, rated_s: pl.Expr, voltage_ratio: pl.Expr) -> pl.Expr:
    """
    Get the transformer impedance reported to its secondary (or resistance if real part) based on the short-circuit tests.

    Args:
        rated_v (pl.Expr): The rated voltage column [V].
        rated_s (pl.Expr): The rated power column [VA].
        voltage_ratio (pl.Expr): The ratio between the applied input voltage to get rated current when transformer 
        secondary is short-circuited and the rated voltage [%].

    Returns:
        pl.Expr: The transformer impedance column [Ohm].
    """
    return voltage_ratio  / 100 * (rated_v**2)/ rated_s

def get_transfo_admittance(voltage_level: pl.Expr, rated_s: pl.Expr, oc_current_ratio: pl.Expr) -> pl.Expr:
    """
    Get the transformer admittance reported to its secondary based on the open circuit test
    Args:
        voltage_level (pl.Expr): The voltage level column [V].
        rated_s (pl.Expr): The rated power column [VA].
        oc_current_ratio (pl.Expr): The ratio between the measured current when transformer secondary is opened and the
        rated current [%].

    Returns:
        pl.Expr: The transformer admittance column [Simens].
    """
    return oc_current_ratio / 100 * rated_s / (voltage_level **2)

def get_transfo_conductance(voltage_level: pl.Expr, iron_losses: pl.Expr) -> pl.Expr:
    """
    Get the transformer conductance reported to its secondary based on iron losses measurement.

    Args:
        voltage_level (pl.Expr): The voltage level column [V].
        iron_losses (pl.Expr): The iron losses column [W].

    Returns:
        pl.Expr: The transformer conductance column [Simens].
    """
    return  iron_losses /(voltage_level**2)

def get_transfo_resistance(voltage_level: pl.Expr, rated_s: pl.Expr, copper_losses: pl.Expr) -> pl.Expr:
    """
    Get the transformer resistance reported to its secondary based on copper losses measurement.

    Args:
        voltage_level (pl.Expr): The voltage level column [V].
        rated_s (pl.Expr): The rated power column [VA].
        copper_losses (pl.Expr): The copper losses column [W].

    Returns:
        pl.Expr: The transformer resistance column [Ohm].
    """
    return  copper_losses * ((voltage_level/rated_s)**2)

def get_transfo_imaginary_component(module: pl.Expr, real: pl.Expr) -> pl.Expr:
    """
    Get the transformer imaginary component based on the module and real component.

    Args:
        module (pl.Expr): The module column [Ohm or Simens].
        real (pl.Expr): The real component column [Ohm or Simens].

    Returns:
        pl.Expr: The transformer imaginary component column [Ohm or Simens].
    """
    return (np.sqrt(module ** 2 - real ** 2))

def get_transfo_vector_group(vector_group: pl.Expr, side: pl.Expr) -> pl.Expr:
    """
    Get the transformer vector group parameters based on a vector group string.

    Args:
        vector_group (str): The vector group string.

    Returns:
        dict: The transformer vector group dictionary (HV connection kind, LV connection kind and phase angle).
    """
    vector_group = vector_group.str.replace(" ", "").str.to_uppercase()
    ck = pl.when(side==TerminalSide.T1.value)\
            .then(vector_group.map_elements(
                lambda x: getattr(ConnectionKind, re.findall(r"^\wN?", x)[0]).value, return_dtype=pl.Utf8)
            ).otherwise(vector_group.map_elements(
                lambda x: getattr(ConnectionKind, re.sub(r"\d+$", "", re.findall(r"\wN?\d+$", x)[0])).value, 
                return_dtype=pl.Utf8)
            )
    angle = pl.when(side==TerminalSide.T1.value)\
            .then(pl.lit(0).cast(pl.Int32))\
            .otherwise(vector_group.map_elements(lambda x: int(re.findall(r"\d+$", x)[0]), return_dtype=pl.Int32))
    return pl.struct(angle, ck).struct.rename_fields(names=["phase_angle", "ck"])



def get_meta_data_string(metadata: pl.Expr)-> pl.Expr:
    return (
        metadata.map_elements(
            lambda x: json.dumps({key: value for key, value in x.items() if value is not None}, ensure_ascii=False), 
        return_dtype=pl.Utf8)
    )


def shape_to_geoalchemy2_col(geo: pl.Expr) -> pl.Expr:
    return geo.map_elements(shape_to_geoalchemy2, return_dtype=pl.Utf8)

def geoalchemy2_to_shape_col(geo_str: pl.Expr) -> pl.Expr:
    return geo_str.map_elements(geoalchemy2_to_shape, return_dtype=pl.Object)

def shape_coordinate_transformer_col(shape_col: pl.Expr, crs_from: str, crs_to: str) -> pl.Expr:
    transformer = Transformer.from_crs(crs_from=CRS(crs_from), crs_to=CRS(crs_to) , always_xy=True).transform
    return shape_col.map_elements(lambda x: transform(transformer, x), return_dtype=pl.Object)

def generate_geo_point(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return (
        pl.concat_list([x, y]).map_elements(lambda coord: Point(*coord), return_dtype=pl.Object)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.SWISS_SRID, crs_to=settings.GPS_SRID)
        .pipe(shape_to_geoalchemy2_col)
    )

def wkt_to_geoalchemy(geo_str: pl.Expr) -> pl.Expr:
    return (
        geo_str.map_elements(from_wkt, return_dtype=pl.Object)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.SWISS_SRID, crs_to=settings.GPS_SRID)
        .pipe(shape_to_geoalchemy2_col)
    )

def geoalchemy2_to_wkt(geo_str: pl.Expr) -> pl.Expr:
    return (
        geo_str.pipe(geoalchemy2_to_shape_col)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.GPS_SRID, crs_to=settings.SWISS_SRID)
        .map_elements(lambda x: x.wkt, return_dtype=pl.Utf8)
    )


def generate_geo_linestring(coord_list: pl.Expr) -> pl.Expr:

    return (
        coord_list.map_elements(lambda x: LineString(batched(x, 2)), return_dtype=pl.Object)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.SWISS_SRID, crs_to=settings.GPS_SRID)
        .pipe(shape_to_geoalchemy2_col)
    )


def combine_shape(geo_shape: pl.Expr) ->  pl.Expr:
    return geo_shape.map_elements(lambda x: union_all(list(map(from_wkt, x))).wkt, return_dtype=pl.Utf8)

def point_list_to_linestring_col(point_list: pl.Expr) -> pl.Expr:
    return point_list.map_elements(point_list_to_linestring, return_dtype=pl.Utf8)
