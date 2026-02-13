from abc import ABC

import polars as pl
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from auto_prepper.core.hyperparameter import HPDomainDiscrete
from auto_prepper.core.transformation import Transformation
from auto_prepper.utils.dataset_type import DatasetType
from auto_prepper.utils.feature_type import FeatureType
from auto_prepper.utils.helpers import pl_to_numpy


class Outliers(ABC):
    pass


class RemoveOutliersSTD(Outliers, Transformation):
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _changing_row_count = True
    _hyperparameter_space = {
        'cutoff_coef': HPDomainDiscrete(
            values=[2, 5],
            default_value=3,
        ),
    }

    def __init__(self, cutoff_coef=None):
        super().__init__(cutoff_coef=cutoff_coef)

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df_selection = self._select_columns(df)
        means = df_selection.mean()
        stds = df_selection.std()
        lower_bounds = means - self._hp_cutoff_coef * stds
        upper_bounds = means + self._hp_cutoff_coef * stds
        for col in df_selection:
            df = df.filter(
                pl.col(col.name).is_between(
                    lower_bounds[col.name], upper_bounds[col.name]
                )
            )
        return df


class RemoveOutliersIQR(Outliers, Transformation):
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _changing_row_count = True
    _hyperparameter_space = {
        'cutoff_coef': HPDomainDiscrete(
            values=[2, 5],
            default_value=2,
        ),
    }

    def __init__(self, cutoff_coef=None):
        super().__init__(cutoff_coef=cutoff_coef)

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df_selection = self._select_columns(df)
        q25 = df_selection.quantile(0.25)
        q75 = df_selection.quantile(0.75)
        iqr = q75 - q25
        lower_bounds = q25 - self._hp_cutoff_coef * iqr
        upper_bounds = q75 + self._hp_cutoff_coef * iqr
        for col in df_selection:
            df = df.filter(
                pl.col(col.name).is_between(
                    lower_bounds[col.name], upper_bounds[col.name]
                )
            )
        return df


class RemoveOutliersIsolationForest(Outliers, Transformation):
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _changing_row_count = True

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df_selection = self._select_columns(df)
        X = pl_to_numpy(df_selection)
        inoutliers = IsolationForest(n_jobs=self._sklearn_n_jobs).fit_predict(
            X
        )
        rows_to_drop = [
            i for i in range(len(inoutliers)) if inoutliers[i] == -1
        ]
        df = df.with_row_index().filter(~pl.col('index').is_in(rows_to_drop))
        df = df.drop('index')
        return df


class RemoveOutliersLocalOutlierFactor(Outliers, Transformation):
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _changing_row_count = True

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df_selection = self._select_columns(df)
        X = pl_to_numpy(df_selection)
        inoutliers = LocalOutlierFactor(
            n_neighbors=2, n_jobs=self._sklearn_n_jobs
        ).fit_predict(X)
        rows_to_drop = [
            i for i in range(len(inoutliers)) if inoutliers[i] == -1
        ]
        df = df.with_row_index().filter(~pl.col('index').is_in(rows_to_drop))
        df = df.drop('index')
        return df


class RemoveOutliersOneClassSVM(Outliers, Transformation):
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _changing_row_count = True

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df_selection = self._select_columns(df)
        X = pl_to_numpy(df_selection)
        inoutliers = OneClassSVM().fit_predict(X)
        rows_to_drop = [
            i for i in range(len(inoutliers)) if inoutliers[i] == -1
        ]
        df = df.with_row_index().filter(~pl.col('index').is_in(rows_to_drop))
        df = df.drop('index')
        return df


class RemoveOutliersEllipticEnvelope(Outliers, Transformation):
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _changing_row_count = True

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        df_selection = self._select_columns(df)
        X = pl_to_numpy(df_selection)
        inoutliers = EllipticEnvelope().fit_predict(X)
        rows_to_drop = [
            i for i in range(len(inoutliers)) if inoutliers[i] == -1
        ]
        df = df.with_row_index().filter(~pl.col('index').is_in(rows_to_drop))
        df = df.drop('index')
        return df
