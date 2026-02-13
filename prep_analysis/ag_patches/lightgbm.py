import lightgbm.compat
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._encode import _unique


def disable_LightGBM_prep():
    lightgbm.compat.LabelEncoder = IdentityLabelEncoder
    lightgbm.compat._LGBMLabelEncoder = IdentityLabelEncoder


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
