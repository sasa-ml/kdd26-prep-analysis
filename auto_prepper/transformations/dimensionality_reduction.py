from abc import ABC

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from umap import UMAP

from auto_prepper.core.hyperparameter import (
    HPDomainCategorical,
    HPDomainContinuous,
    HPDomainDiscrete,
)
from auto_prepper.core.transformation import TransformationSklearn
from auto_prepper.utils.data_types import NUMERIC_TYPES
from auto_prepper.utils.feature_type import FeatureType


class DimensionalityReduction(ABC):
    pass


class DimReducePCA(DimensionalityReduction, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_components': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)
        self.transformer = PCA(n_components=self._hp_n_components)


class DimReduceTruncatedSVD(DimensionalityReduction, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _data_types = NUMERIC_TYPES
    _min_features = 2
    _hyperparameter_space = {
        'n_components': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)
        self.transformer = TruncatedSVD(n_components=self._hp_n_components)


class DimReduceLDA(DimensionalityReduction, TransformationSklearn):
    # TODO multiple target columns
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.CATEGORICAL
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_components': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)
        self.transformer = LDA(n_components=self._hp_n_components)

    def _fit_numpy(self, X, y=None):
        y = y.reshape(
            len(y),
        )
        return super()._fit_numpy(X, y)


class DimReduceKernelPCA(DimensionalityReduction, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_components': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
        'kernel': HPDomainCategorical(
            values=[
                'linear',
                'poly',
                'rbf',
                'sigmoid',
                'cosine',
                'precomputed',
            ],
            default_value='linear',
        ),
    }

    def __init__(self, n_components=None, kernel=None):
        super().__init__(n_components=n_components, kernel=kernel)
        self.transformer = KernelPCA(
            n_components=self._hp_n_components,
            kernel=self._hp_kernel,
            n_jobs=self._sklearn_n_jobs,
        )


class DimReduceTSNE(DimensionalityReduction, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_components': HPDomainDiscrete(
            values=[1, 4],
            default_value=2,
        ),
    }

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)
        self.transformer = TSNE(
            n_components=self._hp_n_components, n_jobs=self._sklearn_n_jobs
        )

    def _transform_numpy(self, X, y=None):
        X_transformed = self.transformer.fit_transform(X)
        return X_transformed, y


class DimReduceUMAP(DimensionalityReduction, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_neighbors': HPDomainDiscrete(
            values=[1, 100],
            default_value=5,
        ),
        'n_components': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
        'min_dist': HPDomainContinuous(
            values=[0.0, 0.99],
            default_value=0.1,
        ),
        'metric': HPDomainCategorical(
            values=[
                'euclidean',
                'manhattan',
                'chebyshev',
                'minkowski',
                'canberra',
                'braycurtis',
                'haversine',
                'mahalanobis',
                'wminkowski',
                'seuclidean',
                'cosine',
                'correlation',
                'hamming',
                'jaccard',
                'dice',
                'russellrao',
                'kulsinski',
                'rogerstanimoto',
                'sokalmichener',
                'sokalsneath',
                'yule',
            ],
            default_value='euclidean',
        ),
    }

    def __init__(
        self,
        n_neighbors=None,
        n_components=None,
        min_dist=None,
        metric=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
        )
        self.transformer = UMAP(
            n_neighbors=self._hp_n_neighbors,
            n_components=self._hp_n_components,
            min_dist=self._hp_min_dist,
            metric=self._hp_metric,
            n_jobs=self._sklearn_n_jobs,
        )


class DimReduceIsomap(DimensionalityReduction, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_components': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)
        self.transformer = Isomap(
            n_components=self._hp_n_components, n_jobs=self._sklearn_n_jobs
        )


class DimReduceLLE(DimensionalityReduction, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_neighbors': HPDomainDiscrete(
            values=[1, 100],
            default_value=5,
        ),
        'n_components': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_neighbors=None, n_components=None):
        super().__init__(n_neighbors=n_neighbors, n_components=n_components)
        self.transformer = LLE(
            n_neighbors=self._hp_n_neighbors,
            n_components=self._hp_n_components,
            n_jobs=self._sklearn_n_jobs,
        )
