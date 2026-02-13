import math
from abc import ABC

import polars as pl
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SMOTENC
from sklearn.neighbors import NearestNeighbors

from auto_prepper.core.hyperparameter import (
    HPDomainContinuous,
    HPDomainDiscrete,
)
from auto_prepper.core.transformation import (
    TransformationImblearn,
    TransformationNoColParam,
)
from auto_prepper.utils.dataset_type import DatasetType
from auto_prepper.utils.feature_type import FeatureType
from auto_prepper.utils.helpers import exclude_columns


class Oversampling(ABC):
    pass


class OversampleRandom(Oversampling, TransformationNoColParam):
    _target_type = FeatureType.CATEGORICAL
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns
    _hyperparameter_space = {
        'ratio_to_majority': HPDomainContinuous(
            values=[0.0, 1.0],
            default_value=0.5,
        ),
    }

    def __init__(self, ratio_to_majority=None):
        super().__init__(ratio_to_majority=ratio_to_majority)

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        value_counts = df[self._target_column].value_counts()
        target_count = math.ceil(
            value_counts['count'].max() * self._hp_ratio_to_majority
        )
        df_sample = None
        for value, count in value_counts.rows():
            sample_size = target_count - count
            if sample_size <= 0:
                continue
            df_value_sample = df.filter(
                pl.col(self._target_column) == value
            ).sample(sample_size, with_replacement=True)
            if df_sample is None:
                df_sample = df_value_sample
            else:
                df_sample = df_sample.vstack(df_value_sample)
        if df_sample is not None:
            df = pl.concat([df, df_sample], how='vertical', rechunk=True)
        return df


class OversampleSMOTE(
    Oversampling, TransformationImblearn, TransformationNoColParam
):
    _target_type = FeatureType.CATEGORICAL
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns
    _hyperparameter_space = {
        'n_neighbors': HPDomainDiscrete(
            values=[1, 100],
            default_value=5,
        ),
    }

    def __init__(self, n_neighbors=None):
        super().__init__(n_neighbors=n_neighbors)
        self.resampler = SMOTE(
            sampling_strategy='not majority',
            k_neighbors=NearestNeighbors(
                n_neighbors=self._hp_n_neighbors, n_jobs=self._sklearn_n_jobs
            ),
        )


class OversampleSMOTEOrRandom(Oversampling, TransformationNoColParam):
    _target_type = FeatureType.CATEGORICAL
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns

    def __init__(
        self,
        n_neighbors=5,
        ratio_to_majority=0.5,
        max_rows=50000,
        max_columns=100,
    ):
        super().__init__()
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.oversampler_SMOTE = OversampleSMOTE(n_neighbors=n_neighbors)
        self.oversampler_random = OversampleRandom(
            ratio_to_majority=ratio_to_majority
        )
        self.oversampler = None

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        pass

    def fit(self, ds):
        if ds.df.height <= self.max_rows and ds.df.width <= self.max_columns:
            self.oversampler = self.oversampler_SMOTE
        else:
            self.oversampler = self.oversampler_random
        return self.oversampler.fit(ds)

    def transform(self, ds):
        return self.oversampler.transform(ds)


class OversampleSMOTEN(
    Oversampling, TransformationImblearn, TransformationNoColParam
):
    _target_type = FeatureType.CATEGORICAL
    _feature_type = FeatureType.CATEGORICAL
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns
    _hyperparameter_space = {
        'n_neighbors': HPDomainDiscrete(
            values=[1, 100],
            default_value=5,
        ),
    }

    def __init__(self, n_neighbors=None):
        super().__init__(n_neighbors=n_neighbors)
        self.resampler = SMOTEN(
            sampling_strategy='not majority',
            k_neighbors=NearestNeighbors(
                n_neighbors=self._hp_n_neighbors, n_jobs=self._sklearn_n_jobs
            ),
        )


class OversampleSMOTENC(
    Oversampling, TransformationImblearn, TransformationNoColParam
):
    _target_type = FeatureType.CATEGORICAL
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns
    _hyperparameter_space = {
        'n_neighbors': HPDomainDiscrete(
            values=[1, 100],
            default_value=5,
        ),
    }

    def __init__(self, n_neighbors=None):
        super().__init__(n_neighbors=n_neighbors)
        self.resampler = SMOTENC(
            categorical_features=[],
            sampling_strategy='not majority',
            k_neighbors=NearestNeighbors(
                n_neighbors=self._hp_n_neighbors, n_jobs=self._sklearn_n_jobs
            ),
        )

    def _fit_df(self, df):
        categorical_indices = []
        for i, col in enumerate(exclude_columns(df, self._target_column)):
            if FeatureType.get_feature_type(col).is_categorical():
                categorical_indices.append(i)
        self.resampler.categorical_features = categorical_indices


class OversampleADASYN(
    Oversampling, TransformationImblearn, TransformationNoColParam
):
    _target_type = FeatureType.CATEGORICAL
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns
    _hyperparameter_space = {
        'n_neighbors': HPDomainDiscrete(
            values=[1, 100],
            default_value=5,
        ),
    }

    def __init__(self, n_neighbors=None):
        super().__init__(n_neighbors=n_neighbors)
        self.resampler = ADASYN(
            sampling_strategy='not majority',
            n_neighbors=NearestNeighbors(
                n_neighbors=self._hp_n_neighbors, n_jobs=self._sklearn_n_jobs
            ),
        )

    def _transform_df(self, df):
        try:
            df = super()._transform_df(df=df)
        except ValueError as e:
            if (
                'No samples will be generated with the provided ratio '
                + 'settings.'
                in str(e)
            ):
                print('No samples generated by ADASYN.')
                # TODO raise or pass?
            else:
                raise e
        return df
