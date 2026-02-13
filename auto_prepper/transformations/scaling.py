from abc import ABC

import polars as pl
from sklearn.preprocessing import QuantileTransformer

from auto_prepper.core.hyperparameter import (
    HPDomainCategorical,
    HPDomainDiscrete,
)
from auto_prepper.core.transformation import (
    Transformation,
    TransformationSklearn,
)
from auto_prepper.utils.feature_type import FeatureType


class Scaling(ABC):
    pass


class Normalize(Scaling, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        self.mins = df.min()
        maxes = df.max()
        self.spread = maxes - self.mins

    def _transform_df(self, df):
        df = df.with_columns(
            (pl.col(col.name) - self.mins[col.name]) / self.spread[col.name]
            for col in df
        )
        return df

    def _inverse_transform_df(self, df):
        df = df.with_columns(
            pl.col(col.name) * self.spread[col.name] + self.mins[col.name]
            for col in df
        )
        return df


class Standardize(Scaling, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        self.means = df.mean()
        self.stds = df.std()

    def _transform_df(self, df):
        df = df.with_columns(
            (pl.col(col.name) - self.means[col.name]) / self.stds[col.name]
            for col in df
        )
        return df

    def _inverse_transform_df(self, df):
        df = df.with_columns(
            pl.col(col.name) * self.stds[col.name] + self.means[col.name]
            for col in df
        )
        return df


class ScaleRobust(Scaling, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        self.medians = df.median()
        self.iqrs = df.quantile(0.75) - df.quantile(0.25)

    def _transform_df(self, df):
        df = df.with_columns(
            (pl.col(col.name) - self.medians[col.name]) / self.iqrs[col.name]
            for col in df
        )
        return df

    def _inverse_transform_df(self, df):
        df = df.with_columns(
            pl.col(col.name) * self.iqrs[col.name] + self.medians[col.name]
            for col in df
        )
        return df


class QuantileTransform(Scaling, TransformationSklearn):
    _feature_type = FeatureType.NUMERIC
    _hyperparameter_space = {
        'n_quantiles': HPDomainDiscrete(
            values=[10, 10000],
            default_value=1000,
        ),
        'output_distribution': HPDomainCategorical(
            values=[
                'uniform',
                'normal',
            ],
            default_value='uniform',
        ),
    }

    def __init__(self, n_quantiles=None, output_distribution=None):
        super().__init__(
            n_quantiles=n_quantiles, output_distribution=output_distribution
        )
        self.transformer = QuantileTransformer(
            n_quantiles=self._hp_n_quantiles,
            output_distribution=self._hp_output_distribution,
        )
