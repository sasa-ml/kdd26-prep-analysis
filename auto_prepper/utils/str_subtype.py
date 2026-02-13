from enum import Enum

import polars as pl

from auto_prepper.utils.helpers import safe_json_decode


class StrSubtype(Enum):
    TEMPORAL = 0
    DECODABLE = 1

    @staticmethod
    def select(df, str_subtype, n_rows_to_check=0):
        if str_subtype == StrSubtype.TEMPORAL:
            return StrSubtype.select_temporal(df, n_rows_to_check)
        if str_subtype == StrSubtype.DECODABLE:
            return StrSubtype.select_decodable(df, n_rows_to_check)
        return pl.DataFrame()

    @staticmethod
    def select_temporal(df, n_rows_to_check=0):
        # TODO doing double work transforming here and then in transform, fix
        string_columns = df.select(pl.col(pl.Utf8)).columns
        df_check = None
        if n_rows_to_check == 0 or n_rows_to_check > df.height:
            df_check = df
        else:
            df_check = df.sample(n_rows_to_check)

        str_temporal_columns = []
        for col in string_columns:
            try:
                df_check[col].str.to_time(strict=False)
                str_temporal_columns.append(col)
                continue
            except Exception:
                pass
            try:
                df_check[col].str.to_date(strict=False)
                str_temporal_columns.append(col)
                continue
            except Exception:
                pass
            try:
                df_check[col].str.to_datetime(strict=False)
                str_temporal_columns.append(col)
            except Exception:
                pass

        return df.select(str_temporal_columns)

    @staticmethod
    def select_decodable(df, n_rows_to_check=0):
        # TODO doing double work transforming here and then in transform, fix
        string_columns = df.select(pl.col(pl.Utf8)).columns
        df_check = None
        if n_rows_to_check == 0 or n_rows_to_check > df.height:
            df_check = df
        else:
            df_check = df.sample(n_rows_to_check)

        str_decodable_columns = []
        for col in string_columns:
            try:
                df_check.select(safe_json_decode(pl.col(col)))
                str_decodable_columns.append(col)
            except Exception:
                pass

        return df.select(str_decodable_columns)

    def __str__(self):
        if self == StrSubtype.TEMPORAL:
            return 'temporal'
        if self == StrSubtype.DECODABLE:
            return 'decodable'
        return 'None'
