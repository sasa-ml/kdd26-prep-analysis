import numpy as np
import tabdpt.estimator
import tabdpt.utils
from autogluon.tabular.models import TabDPTModel
from sklearn.base import BaseEstimator, TransformerMixin


def disable_TabDPT_prep():
    tabdpt.utils.normalize_data = normalize_data
    tabdpt.utils.clip_outliers = clip_outliers
    tabdpt.estimator.SimpleImputer = NoOpImputer

    TabDPTModel._original_get_model_params = TabDPTModel._get_model_params
    TabDPTModel._get_model_params = _get_model_params

    TabDPTModel.preprocess = preprocess


def normalize_data(data, eval_pos=-1, dim=0, return_mean_std: bool = False):
    return data


def clip_outliers(data, eval_pos=-1, n_sigma=4, dim=0):
    return data


class NoOpImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
        add_indicator=False,
    ):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy
        self.add_indicator = add_indicator
        self.statistics_ = None
        self.indicator_ = None

    def fit(self, X, y=None):
        self.statistics_ = np.zeros(X.shape[1]) if hasattr(X, "shape") else []
        if self.add_indicator:
            self.indicator_ = (
                np.zeros(X.shape[1], dtype=bool) if hasattr(X, "shape") else []
            )
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def _get_model_params(
    self, convert_search_spaces_to_default: bool = False
) -> dict:
    params = self._original_get_model_params(convert_search_spaces_to_default)
    params["missing_indicators"] = False
    params["normalizer"] = None
    return params


def preprocess(
    self,
    X,
    preprocess_nonadaptive=True,
    preprocess_stateful=True,
    **kwargs,
):
    return X.to_numpy()
