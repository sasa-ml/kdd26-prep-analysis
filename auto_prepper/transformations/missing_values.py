from abc import ABC

import polars as pl
import polars.selectors as cs
from sklearn.impute import KNNImputer

from auto_prepper.core.hyperparameter import HPDomainDiscrete
from auto_prepper.core.transformation import (
    Transformation,
    TransformationNoColParam,
    TransformationSklearn,
)
from auto_prepper.utils.data_types import FLOAT_TYPES
from auto_prepper.utils.dataset_type import DatasetType
from auto_prepper.utils.feature_type import FeatureType


class MissingValues(ABC):
    pass


class DropUnnamedColumns(MissingValues, TransformationNoColParam):
    # unnamed pattern matches polars unnamed column names:
    # '' or '_duplicated_N' where N is an integer
    _unnamed_pattern = r'^$|^_duplicated_\d+$'

    def _fit_df(self, df):
        self.columns_to_drop = df.select(
            cs.matches(self._unnamed_pattern)
        ).columns

    def _transform_df(self, df):
        df = df.drop(self.columns_to_drop)
        return df


class DropNoneColumns(MissingValues, TransformationNoColParam):

    def _fit_df(self, df):
        self.columns_to_drop = [
            col.name for col in df if (col.null_count() == df.height)
        ]

    def _transform_df(self, df):
        df = df.drop(self.columns_to_drop)
        return df


class DropNoneRows(MissingValues, TransformationNoColParam):
    _changing_row_count = True

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.filter(~pl.all_horizontal(pl.all().is_null()))
        return df


class FillNaNWithNone(MissingValues, Transformation):
    _data_types = FLOAT_TYPES

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.fill_nan(None)
        return df


class EncodeNone(MissingValues, Transformation):

    def _fit_df(self, df):
        self.df_columns_with_none = df[
            [col.name for col in df if (col.null_count() > 0)]
        ]

    def _transform_df(self, df):
        df_encoded_none = df.select(
            pl.col(self.df_columns_with_none.columns).is_null().cast(pl.Utf8)
        )
        df = df.with_columns(
            col.alias(f'{col.name}_missing') for col in df_encoded_none
        )
        return df


class DropNone(MissingValues, Transformation):
    _dataset_type_threshold = DatasetType.TRAIN
    _changing_row_count = True

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.drop_nulls(subset=self._columns_selected)
        return df


class FillNoneNumericValue(MissingValues, Transformation):
    _feature_type = FeatureType.NUMERIC

    def __init__(self, value=-1):
        super().__init__()
        self.value = value

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.fill_null(value=self.value)
        return df


class FillNoneCategoricalValue(MissingValues, Transformation):
    _feature_type = FeatureType.CATEGORICAL

    def __init__(self, value='_missing_'):
        super().__init__()
        self.value = value

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.fill_null(value=self.value)
        return df


class FillNoneBackward(MissingValues, Transformation):

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.fill_null(strategy='backward')
        return df


class FillNoneForward(MissingValues, Transformation):

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df = df.fill_null(strategy='forward')
        return df


class FillNoneMode(MissingValues, Transformation):

    def _fit_df(self, df):
        self.modes = {col.name: col.drop_nulls().mode()[0] for col in df}

    def _transform_df(self, df):
        df = df.with_columns(
            pl.col(col.name).fill_null(value=self.modes[col.name])
            for col in df
        )
        return df


class FillNoneMean(MissingValues, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        self.means = {col.name: col.mean() for col in df}

    def _transform_df(self, df):
        df = df.with_columns(
            pl.col(col.name).fill_null(value=self.means[col.name])
            for col in df
        )
        return df


class FillNoneInterpolation(MissingValues, Transformation):
    _feature_type = FeatureType.NUMERIC

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        # filling by interpolation and completing backward
        df = df.with_columns(pl.col(col.name).interpolate() for col in df)
        return df


class FillNoneKNN(MissingValues, TransformationSklearn):
    _feature_type = FeatureType.NUMERIC
    _hyperparameter_space = {
        'n_neighbors': HPDomainDiscrete(
            values=[1, 100],
            default_value=5,
        ),
    }

    def __init__(self, n_neighbors=None):
        super().__init__(n_neighbors=n_neighbors)
        self.transformer = KNNImputer(n_neighbors=self._hp_n_neighbors)


class FillNoneKNNOrMean(MissingValues, Transformation):
    _feature_type = FeatureType.NUMERIC

    def __init__(
        self,
        n_neighbors=5,
        max_rows=50000,
        max_columns=100,
    ):
        super().__init__()
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.imputer_KNN = FillNoneKNN(n_neighbors=n_neighbors)
        self.imputer_mean = FillNoneMean()
        self.imputer = None

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        pass

    def fit(self, ds, columns=None):
        if ds.df.height <= self.max_rows and ds.df.width <= self.max_columns:
            self.imputer = self.imputer_KNN
        else:
            self.imputer = self.imputer_mean
        return self.imputer.fit(ds, columns)

    def transform(self, ds):
        return self.imputer.transform(ds)
