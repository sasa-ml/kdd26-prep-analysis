from abc import ABC

import polars as pl

from auto_prepper.core.transformation import TransformationNoColParam
from auto_prepper.utils.dataset_type import DatasetType


class Duplicates(ABC):
    pass


class DropDuplicates(Duplicates, TransformationNoColParam):
    _dataset_type_threshold = DatasetType.TEST
    _changing_row_count = True

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.unique()
        return df


class AggregateDuplicates(Duplicates, TransformationNoColParam):
    _changing_row_count = True

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.group_by(df.columns).agg(pl.len().alias('duplicate_count'))
        return df
