import math
from abc import ABC

import polars as pl
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours as ENN
from imblearn.under_sampling import InstanceHardnessThreshold as IHT
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import OneSidedSelection as OSS
from imblearn.under_sampling import TomekLinks
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from auto_prepper.core.hyperparameter import (
    HPDomainContinuous,
    HPDomainDiscrete,
)
from auto_prepper.core.transformation import (
    TransformationImblearn,
    TransformationNoColParam,
)
from auto_prepper.transformations.encoding import EncodeInt
from auto_prepper.utils.data_types import STRING_TYPES
from auto_prepper.utils.dataset_type import DatasetType
from auto_prepper.utils.feature_type import FeatureType


class Undersampling(ABC):
    pass


class UndersampleRandom(Undersampling, TransformationNoColParam):
    _target_type = FeatureType.CATEGORICAL
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns
    _hyperparameter_space = {
        'ratio_to_minority': HPDomainContinuous(
            values=[1.0, 2.0],
            default_value=2,
        ),
    }

    def __init__(self, ratio_to_minority=None):
        super().__init__(ratio_to_minority=ratio_to_minority)

    def _fit_df(self, df):
        pass

    def _transform_df(self, df):
        value_counts = df[self._target_column].value_counts()
        target_count = math.ceil(
            value_counts['count'].min() * self._hp_ratio_to_minority
        )
        df_sample = None
        for value, count in value_counts.rows():
            sample_size = count - target_count
            if sample_size <= 0:
                continue
            df_value_sample = df.filter(
                pl.col(self._target_column) == value
            ).sample(sample_size)
            if df_sample is None:
                df_sample = df_value_sample
            else:
                df_sample = df_sample.vstack(df_value_sample)
        if df_sample is not None:
            df = df.join(df_sample, on=df.columns, how='anti')
        return df


class UndersampleENN(
    Undersampling, TransformationImblearn, TransformationNoColParam
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
        self.resampler = ENN(
            sampling_strategy='not minority',
            n_neighbors=NearestNeighbors(
                n_neighbors=self._hp_n_neighbors, n_jobs=self._sklearn_n_jobs
            ),
            n_jobs=self._sklearn_n_jobs,
        )


class UndersampleTomekLinks(
    Undersampling, TransformationImblearn, TransformationNoColParam
):
    _target_type = FeatureType.CATEGORICAL
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns

    def __init__(self):
        super().__init__()
        self.resampler = TomekLinks(
            sampling_strategy='not minority', n_jobs=self._sklearn_n_jobs
        )


class UndersampleNearMiss(
    Undersampling, TransformationImblearn, TransformationNoColParam
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
        n_neighbors = NearestNeighbors(
            n_neighbors=self._hp_n_neighbors, n_jobs=self._sklearn_n_jobs
        )
        self.resampler = NearMiss(
            sampling_strategy='not minority',
            n_neighbors=n_neighbors,
            n_neighbors_ver3=n_neighbors,
            n_jobs=self._sklearn_n_jobs,
        )


class UndersampleClusterCentroids(
    Undersampling, TransformationImblearn, TransformationNoColParam
):
    _target_type = FeatureType.CATEGORICAL
    _feature_type = FeatureType.NUMERIC
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns

    def __init__(self):
        super().__init__()
        self.resampler = ClusterCentroids(
            sampling_strategy='not minority',
            estimator=KMeans(n_init='auto'),
        )


class UndersampleOSS(
    Undersampling, TransformationImblearn, TransformationNoColParam
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
        self.resampler = OSS(
            sampling_strategy='not minority',
            n_neighbors=KNeighborsClassifier(
                n_neighbors=self._hp_n_neighbors, n_jobs=self._sklearn_n_jobs
            ),
            n_jobs=self._sklearn_n_jobs,
        )


class UndersampleIHT(
    Undersampling, TransformationImblearn, TransformationNoColParam
):
    _target_type = FeatureType.CATEGORICAL
    _dataset_type_threshold = DatasetType.TRAIN
    _req_target_column = True
    _changing_row_count = True
    # TODO multiple target columns

    def __init__(self):
        super().__init__()
        self.resampler = IHT(
            sampling_strategy='not minority', n_jobs=self._sklearn_n_jobs
        )

    def transform(self, ds):
        if ds.df[ds.target_column].dtype in STRING_TYPES:
            encoder_int = EncodeInt()
            ds = encoder_int.fit(ds, columns=[ds.target_column]).transform(ds)
            ds = super().transform(ds)
            ds = encoder_int.inverse_transform(ds)
        else:
            ds = super().transform(ds)
        return ds
