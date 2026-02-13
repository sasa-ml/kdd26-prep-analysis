from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import tabpfn.classifier
import tabpfn.regressor
from autogluon.tabular.models import RealTabPFNv25Model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils._encode import _unique
from tabpfn.constants import NA_PLACEHOLDER


def disable_TabPFN_prep():
    RealTabPFNv25Model._original_get_model_params = (
        RealTabPFNv25Model._get_model_params
    )
    RealTabPFNv25Model._get_model_params = _get_model_params

    tabpfn.classifier.LabelEncoder = IdentityLabelEncoder
    tabpfn.classifier.get_ordinal_encoder = get_identity_column_transformer
    tabpfn.regressor.get_ordinal_encoder = get_identity_column_transformer
    tabpfn.classifier.fix_dtypes = fix_dtypes
    tabpfn.regressor.fix_dtypes = fix_dtypes
    tabpfn.classifier.process_text_na_dataframe = process_text_na_dataframe
    tabpfn.regressor.process_text_na_dataframe = process_text_na_dataframe


def _get_model_params(
    self, convert_search_spaces_to_default: bool = False
) -> dict:
    params = self._original_get_model_params(convert_search_spaces_to_default)
    params["inference_config/PREPROCESS_TRANSFORMS"] = ['none']
    params["inference_config/OUTLIER_REMOVAL_STD"] = None
    params["inference_config/FINGERPRINT_FEATURE"] = False
    params["inference_config/POLYNOMIAL_FEATURES"] = 'no'
    params["inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS"] = [None]
    params["preprocessing/scaling"] = ['none']
    params["preprocessing/categoricals"] = 'none'
    params["preprocessing/append_original"] = False
    params["preprocessing/global"] = None
    params["balance_probabilities"] = False
    return params


def get_identity_column_transformer():
    return ColumnTransformer(transformers=[], remainder='passthrough')


class IdentityLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_ = _unique(y)
        return self

    def transform(self, y):
        return self._as_array(y)

    def fit_transform(self, y):
        self.classes_ = _unique(y)
        return self._as_array(y)

    def inverse_transform(self, y):
        return self._as_array(y)

    @staticmethod
    def _as_array(y):
        if isinstance(y, np.ndarray):
            return y
        return np.asarray(y)


def fix_dtypes(  # noqa: D103
    X: pd.DataFrame | np.ndarray,
    cat_indices: Sequence[int | str] | None,
    numeric_dtype: Literal["float32", "float64"] = "float64",
) -> pd.DataFrame:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    return X


def process_text_na_dataframe(
    X: pd.DataFrame,
    placeholder: str = NA_PLACEHOLDER,
    ord_encoder: ColumnTransformer | None = None,
    *,
    fit_encoder: bool = False,
) -> np.ndarray:
    return X.to_numpy()
