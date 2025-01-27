import patito as pt
import polars as pl
from typing_extensions import Literal
from distflow_schema._constraints import literal_constraint

TYPES = Literal["branch", "transformer", "switch"]

class EdgeData(pt.Model):
    u_of_edge: int = pt.Field(dtype=pl.Int32)
    v_of_edge: int = pt.Field(dtype=pl.Int32)
    r_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    x_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    b_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    g_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    n_transfo: float = pt.Field(dtype=pl.Float64, default=1.0)
    type: TYPES = pt.Field(dtype=pl.Utf8, constraints=literal_constraint(pt.field, TYPES))