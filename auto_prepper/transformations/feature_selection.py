from abc import ABC

import numpy as np
import polars as pl
from scipy.stats import kendalltau, spearmanr
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from auto_prepper.core.hyperparameter import (
    HPDomainContinuous,
    HPDomainDiscrete,
)
from auto_prepper.core.transformation import (
    Transformation,
    TransformationNoColParam,
    TransformationSklearn,
)
from auto_prepper.utils.data_types import NUMERIC_TYPES
from auto_prepper.utils.feature_type import FeatureType


class FeatureSelection(ABC):
    pass


class DropInvariantColumns(FeatureSelection, TransformationNoColParam):

    def _fit_df(self, df):
        self.columns_to_drop = [
            col.name for col in df if df[col.name].drop_nulls().n_unique() == 1
        ]

    def _transform_df(self, df):
        df = df.drop(self.columns_to_drop)
        return df


class SelectVariationThreshold(FeatureSelection, Transformation):
    _feature_type = FeatureType.NUMERIC
    _min_features = 2
    _hyperparameter_space = {
        'coef_var_threshold': HPDomainContinuous(
            values=[0.01, 0.05],
            default_value=0.05,
        ),
    }

    def __init__(self, coef_var_threshold=None):
        super().__init__(coef_var_threshold=coef_var_threshold)

    def _fit_df(self, df):
        coef_of_variation = df.std().with_columns(
            pl.col(col) / df.mean()[col] for col in df.columns
        )
        self.columns_to_drop = [
            col.name
            for col in df
            if coef_of_variation[col.name][0] < self._hp_coef_var_threshold
        ]

    def _transform_df(self, df):
        df = df.drop(self.columns_to_drop)
        # TODO categorical?
        return df


class SelectRFEDecisionTreeReg(FeatureSelection, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.NUMERIC
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = RFE(
            estimator=DecisionTreeRegressor(),
            n_features_to_select=self._hp_n_features,
        )


class SelectRFEDecisionTreeCls(FeatureSelection, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.CATEGORICAL
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = RFE(
            estimator=DecisionTreeClassifier(),
            n_features_to_select=self._hp_n_features,
        )


class SelectUnivariatePearson(FeatureSelection, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.NUMERIC
    _feature_type = FeatureType.NUMERIC
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = SelectKBest(
            score_func=r_regression, k=self._hp_n_features
        )


class SelectUnivariateSpearman(FeatureSelection, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.NUMERIC_OR_ORDINAL
    _feature_type = FeatureType.NUMERIC_OR_ORDINAL
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = SelectKBest(
            score_func=self._spearmanr_score, k=self._hp_n_features
        )

    def _spearmanr_score(self, X, y):
        n = X.shape[1]
        statistics = np.empty(n)
        p_values = np.empty(n)
        for i in range(n):
            res = spearmanr(X[:, i], y)
            statistics[i] = res.statistic
            p_values[i] = res.pvalue
        return statistics, p_values


class SelectUnivariateANOVA(FeatureSelection, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.NUMERIC
    # _feature_type = FeatureType.CATEGORICAL
    # TODO revise feature/data types, uncomment when fixed
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = SelectKBest(
            score_func=f_classif, k=self._hp_n_features
        )


class SelectUnivariateKendall(FeatureSelection, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.NUMERIC_OR_ORDINAL
    _feature_type = FeatureType.NUMERIC_OR_ORDINAL
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = SelectKBest(
            score_func=self._kendalltau_score, k=self._hp_n_features
        )

    def _kendalltau_score(self, X, y):
        n = X.shape[1]
        statistics = np.empty(n)
        p_values = np.empty(n)
        for i in range(n):
            res = kendalltau(X[:, i], y)
            statistics[i] = res.statistic
            p_values[i] = res.pvalue
        return statistics, p_values


class SelectUnivariateChi2(FeatureSelection, TransformationSklearn):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.CATEGORICAL
    # _feature_type = FeatureType.CATEGORICAL
    # TODO revise feature/data types, uncomment when fixed
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = SelectKBest(score_func=chi2, k=self._hp_n_features)


class SelectUnivariateMutualInformationReg(
    FeatureSelection, TransformationSklearn
):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.NUMERIC
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    # TODO see discrete_features param
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = SelectKBest(
            score_func=mutual_info_regression, k=self._hp_n_features
        )


class SelectUnivariateMutualInformationCls(
    FeatureSelection, TransformationSklearn
):
    _numpy_reducing_dimensionality = True
    _target_type = FeatureType.CATEGORICAL
    _data_types = NUMERIC_TYPES
    _req_target_column = True
    _min_features = 2
    # TODO see discrete_features param
    _hyperparameter_space = {
        'n_features': HPDomainDiscrete(
            values=[1, 10000],
            default_value=2,
        ),
    }

    def __init__(self, n_features=None):
        super().__init__(n_features=n_features)
        self.transformer = SelectKBest(
            score_func=mutual_info_classif, k=self._hp_n_features
        )
