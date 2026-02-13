from abc import ABC

import polars as pl

from auto_prepper.core.transformation import (
    Transformation,
    TransformationTargetOnly,
)
from auto_prepper.utils.data_types import STRING_TYPES
from auto_prepper.utils.feature_type import FeatureType


class Encoding(ABC):
    pass


class EncodeOneHot(Encoding, Transformation):
    _feature_type = FeatureType.CATEGORICAL_NOMINAL
    _data_types = STRING_TYPES

    def _fit_df(self, df):
        self._onehot_column_names = []
        for col in df.columns:
            uniques = df[col].unique().to_list()
            self._onehot_column_names.extend(
                f'{col}_{value}' for value in uniques
            )

    def _transform_df(self, df):
        df = df.to_dummies()
        fit_columns = set(self._onehot_column_names)
        df_columns = set(df.columns)
        missing_columns = list(fit_columns - df_columns)
        if missing_columns:
            df = df.with_columns(
                [pl.lit(0).alias(col) for col in missing_columns]
            )
        extra_columns = df_columns - fit_columns
        if extra_columns:
            df = df.drop(list(extra_columns))
        return df


class EncodeHash(Encoding, Transformation):
    _feature_type = FeatureType.CATEGORICAL_NOMINAL
    _data_types = STRING_TYPES

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.with_columns(pl.all().hash())
        return df


class EncodeOneHotOrHash(Encoding, Transformation):
    _feature_type = FeatureType.CATEGORICAL_NOMINAL
    _data_types = STRING_TYPES

    def __init__(self, max_values_1hot=20):
        super().__init__()
        self.max_values_1hot = max_values_1hot
        self.one_hot_encoder = EncodeOneHot()
        self.hash_encoder = EncodeHash()

    def _fit_df(self, df):
        self.one_hot_columns = []
        self.hash_columns = []
        for col in df.columns:
            if df[col].n_unique() <= self.max_values_1hot:
                self.one_hot_columns.append(col)
            else:
                self.hash_columns.append(col)
        if self.one_hot_columns:
            self.one_hot_encoder._fit_df(df.select(self.one_hot_columns))

    def _transform_df(self, df):
        if self.hash_columns:
            df = df.with_columns(
                self.hash_encoder._transform_df(df.select(self.hash_columns))
            )
        if self.one_hot_columns:
            df = df.with_columns(
                self.one_hot_encoder._transform_df(
                    df.select(self.one_hot_columns)
                )
            )
            df = df.drop(self.one_hot_columns)
        return df


class EncodeInt(Encoding, Transformation):
    _feature_type = FeatureType.CATEGORICAL
    _data_types = STRING_TYPES
    _exclude_target = False

    def _fit_df(self, df):
        self.mapping = {}
        self.inverse_mapping = {}
        for col in df:
            self.mapping[col.name] = {
                val: i for i, val in enumerate(col.unique())
            }
            self.inverse_mapping[col.name] = {
                i: val for val, i in self.mapping[col.name].items()
            }

    def _transform_df(self, df):
        df = df.with_columns(
            col.map_elements(
                lambda x: self.mapping[col.name].get(x, -1),
                return_dtype=pl.Int64,
            )
            for col in df
        )
        return df

    def _inverse_transform_df(self, df):
        df = df.with_columns(
            col.map_elements(
                lambda x: self.inverse_mapping[col.name].get(
                    x, 'unknown_label'
                ),
                return_dtype=pl.Utf8,
            )
            for col in df
        )
        return df


class EncodeIntTarget(EncodeInt, Encoding, TransformationTargetOnly):
    pass
