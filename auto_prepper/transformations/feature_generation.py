from abc import ABC

import numpy as np
import polars.selectors as cs
from sklearn.preprocessing import KBinsDiscretizer

from auto_prepper.core.hyperparameter import HPDomainDiscrete
from auto_prepper.core.transformation import (
    Transformation,
    TransformationSklearn,
)
from auto_prepper.utils.data_types import TEMPORAL_TYPES
from auto_prepper.utils.feature_type import FeatureType


class FeatureGeneration(ABC):
    pass


class TemporalExpand(FeatureGeneration, Transformation):
    _data_types = TEMPORAL_TYPES

    def _fit_df(self, df):
        self.time_columns = df.select(cs.time()).columns
        self.date_columns = df.select(cs.date()).columns
        self.datetime_columns = df.select(cs.datetime()).columns

    def _transform_df(self, df):
        df_time = df.select(self.time_columns)
        for col in df_time:
            df = df.with_columns(
                col.dt.hour().alias(f'{col.name}_hour'),
                col.dt.minute().alias(f'{col.name}_minute'),
                col.dt.second().alias(f'{col.name}_second'),
                col.dt.millisecond().alias(f'{col.name}_millisecond'),
            )
        df_date = df.select(self.date_columns)
        for col in df_date:
            df = df.with_columns(
                col.dt.year().alias(f'{col.name}_year'),
                col.dt.month().alias(f'{col.name}_month'),
                col.dt.day().alias(f'{col.name}_day'),
                col.dt.weekday().alias(f'{col.name}_weekday'),
                col.dt.ordinal_day().alias(f'{col.name}_ordinal_day'),
            )
        df_datetime = df.select(self.datetime_columns)
        for col in df_datetime:
            df = df.with_columns(
                col.dt.year().alias(f'{col.name}_year'),
                col.dt.month().alias(f'{col.name}_month'),
                col.dt.day().alias(f'{col.name}_day'),
                col.dt.weekday().alias(f'{col.name}_weekday'),
                col.dt.ordinal_day().alias(f'{col.name}_ordinal_day'),
                col.dt.hour().alias(f'{col.name}_hour'),
                col.dt.minute().alias(f'{col.name}_minute'),
                col.dt.second().alias(f'{col.name}_second'),
                col.dt.millisecond().alias(f'{col.name}_millisecond'),
            )
        return df


class DiscretizeUniform(FeatureGeneration, Transformation):
    _feature_type = FeatureType.NUMERIC
    _hyperparameter_space = {
        'bin_count': HPDomainDiscrete(
            values=[2, 100],
            default_value=10,
        ),
    }

    def __init__(self, bin_count=None):
        super().__init__(bin_count=bin_count)

    def _fit_df(self, df):
        self._bin_breaks = {}
        for col in df:
            self._bin_breaks[col.name] = np.linspace(
                start=col.min(), stop=col.max(), num=self._hp_bin_count
            )

    def _transform_df(self, df):
        df_discrete = df.with_columns(
            col.cut(breaks=self._bin_breaks[col.name], include_breaks=True)
            for col in df
        )
        df = df.with_columns(
            df_discrete.unnest(col.name)['break_point'].alias(
                f'{col.name}_discrete'
            )
            for col in df
        )
        return df


class DiscretizeQuantile(FeatureGeneration, Transformation):
    _feature_type = FeatureType.NUMERIC
    _hyperparameter_space = {
        'bin_count': HPDomainDiscrete(
            values=[2, 100],
            default_value=10,
        ),
    }

    def __init__(self, bin_count=None):
        super().__init__(bin_count=bin_count)

    def _fit_df(self, df):
        df_quantile = df.with_columns(
            col.qcut(
                quantiles=self._hp_bin_count,
                allow_duplicates=True,
                include_breaks=True,
            )
            for col in df
        )
        df_quantile = df.with_columns(
            df_quantile.unnest(col.name)['break_point'].alias(
                f'{col.name}_quantile'
            )
            for col in df
        )
        self._quantiles = {}
        for col in df:
            self._quantiles[col.name] = df_quantile[
                f'{col.name}_quantile'
            ].unique()[
                :-1
            ]  # [:-1] to exclude inf

    def _transform_df(self, df):
        df_discrete = df.with_columns(
            col.cut(breaks=self._quantiles[col.name], include_breaks=True)
            for col in df
        )
        df = df.with_columns(
            df_discrete.unnest(col.name)['break_point'].alias(
                f'{col.name}_quantile'
            )
            for col in df
        )
        return df


class DiscretizeKMeans(FeatureGeneration, TransformationSklearn):
    _feature_type = FeatureType.NUMERIC
    _hyperparameter_space = {
        'bin_count': HPDomainDiscrete(
            values=[2, 100],
            default_value=10,
        ),
    }

    def __init__(self, bin_count=None):
        super().__init__(bin_count=bin_count)
        self.transformer = KBinsDiscretizer(
            n_bins=self._hp_bin_count, strategy='kmeans'
        )

    def _fit_df(self, df):
        super()._fit_df(df)
        self._bin_breaks = {
            col: self.transformer.bin_edges_[i]
            for i, col in enumerate(df.columns)
        }

    def _transform_df(self, df):
        df_discrete = df.with_columns(
            col.cut(breaks=self._bin_breaks[col.name], include_breaks=True)
            for col in df
        )
        df = df.with_columns(
            df_discrete.unnest(col.name)['break_point'].alias(
                f'{col.name}_kmeans'
            )
            for col in df
        )
        return df


class TransformSquare(FeatureGeneration, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.with_columns(
            (col**2).alias(f'{col.name}_square') for col in df
        )
        return df


class TransformRoot(FeatureGeneration, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.with_columns(
            col.sqrt().alias(f'{col.name}_root') for col in df
        )
        return df


class TransformLog(FeatureGeneration, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.with_columns(
            col.log1p().alias(f'{col.name}_log') for col in df
        )
        return df


class TransformExp(FeatureGeneration, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.with_columns(col.exp().alias(f'{col.name}_exp') for col in df)
        return df
