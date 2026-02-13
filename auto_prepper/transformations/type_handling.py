from abc import ABC

import polars as pl

from auto_prepper.core.transformation import (
    Transformation,
    TransformationNoColParam,
)
from auto_prepper.utils.data_types import SUPPORTED_TYPES
from auto_prepper.utils.helpers import safe_json_decode
from auto_prepper.utils.str_subtype import StrSubtype


class DataTypes(ABC):
    pass


class CastFloat(DataTypes, Transformation):
    _exclude_target = False
    _data_types = [pl.Decimal, pl.Float32]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.cast(pl.Float64)
        return df


class CastInt(DataTypes, Transformation):
    _exclude_target = False
    _data_types = [pl.Int8, pl.Int16, pl.Int32]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.cast(pl.Int64)
        return df


class CastUInt(DataTypes, Transformation):
    _exclude_target = False
    _data_types = [pl.UInt8, pl.UInt16, pl.UInt32]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.cast(pl.UInt64)
        return df


class CastStr(DataTypes, Transformation):
    _exclude_target = False
    _data_types = [pl.Categorical, pl.Enum]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.cast(pl.Utf8)
        return df


class BooleanToCategorical(DataTypes, Transformation):
    _exclude_target = False
    _data_types = [pl.Boolean]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.cast(pl.Categorical)
        return df


class BooleanToStr(DataTypes, Transformation):
    _exclude_target = False
    _data_types = [pl.Boolean]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.cast(pl.Utf8)
        return df


class BooleanToInt(DataTypes, Transformation):
    _exclude_target = False
    _data_types = [pl.Boolean]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.cast(pl.Int64)
        return df


class TemporalToInt(DataTypes, Transformation):
    _exclude_target = False
    _data_types = [pl.Date, pl.Datetime, pl.Duration, pl.Time]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.cast(pl.Int64)
        return df


class StrToTemporal(DataTypes, Transformation):
    _exclude_target = False
    _str_subtype = StrSubtype.TEMPORAL

    def _fit_df(self, df):
        self.time_columns = []
        self.date_columns = []
        self.datetime_columns = []
        string_columns = df.select(pl.col(pl.Utf8)).columns
        df = df.head(100)
        for col in string_columns:
            try:
                df[col].str.to_time(strict=False)
                self.time_columns.append(col)
                continue
            except Exception:
                pass
            try:
                df[col].str.to_date(strict=False)
                self.date_columns.append(col)
                continue
            except Exception:
                pass
            try:
                df[col].str.to_datetime(strict=False)
                self.datetime_columns.append(col)
            except Exception:
                pass

    def _transform_df(self, df):
        for col in self.time_columns:
            df = df.with_columns(pl.col(col).str.to_time(strict=False))
        for col in self.date_columns:
            df = df.with_columns(pl.col(col).str.to_date(strict=False))
        for col in self.datetime_columns:
            df = df.with_columns(pl.col(col).str.to_datetime(strict=False))
        return df


class StrDecode(DataTypes, Transformation):
    _exclude_target = False
    _str_subtype = StrSubtype.DECODABLE

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.with_columns(safe_json_decode(pl.all()))
        return df


class ListExplode(DataTypes, Transformation):
    _changing_row_count = True
    _data_types = [pl.List, pl.Array]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        for col in self._columns_input:
            df = df.explode(col)
        return df


class DictUnnest(DataTypes, Transformation):
    _data_types = [pl.Struct]

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.unnest(self._columns_input)
        return df


class DropUnsupportedTypes(DataTypes, TransformationNoColParam):

    def _fit_df(self, df):
        self.columns_to_keep = df.select(
            pl.col(t) for t in SUPPORTED_TYPES
        ).columns

    def _transform_df(self, df):
        df = df.select(self.columns_to_keep)
        return df
