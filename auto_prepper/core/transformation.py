import copy
from abc import ABC, abstractmethod

import numpy as np

from auto_prepper.core.dataset import Dataset
from auto_prepper.io.integration import feature_target_join
from auto_prepper.io.separation import feature_target_split
from auto_prepper.utils.data_types import data_types_match, select_data_types
from auto_prepper.utils.dataset_type import DatasetType
from auto_prepper.utils.exceptions import (
    AlreadyFittedError,
    DatasetEmptyError,
    DataTypeError,
    FeatureTypeError,
    FitError,
    HyperparameterNameError,
    HyperparameterValueError,
    InsufficientFeaturesError,
    InverseTransformError,
    NoTargetError,
    NotFittedError,
    NotInversibleError,
    TransformError,
    TransformTargetError,
)
from auto_prepper.utils.feature_type import FeatureType
from auto_prepper.utils.helpers import (
    exclude_columns,
    get_class_name,
    get_object_class_name,
    numpy_to_pl,
    pl_to_numpy,
    select_columns,
)
from auto_prepper.utils.str_subtype import StrSubtype


class Transformation(ABC):
    _target_type = FeatureType.ANY
    _feature_type = FeatureType.ANY
    _dataset_type_threshold = DatasetType.INFERENCE
    _changing_row_count = False
    _req_target_column = False
    _exclude_target = None
    _min_features = None
    _data_types = []
    _str_subtype = None
    _hyperparameter_space = {}
    _sklearn_n_jobs = -1

    def __init__(self, **hyperparameters):
        if self._exclude_target is None:
            self._exclude_target = self.get_exclude_target()
        if self._min_features is None:
            self._min_features = self.get_min_features()
        self._target_column = None
        self._columns_input = None
        # column selection for the transformation,
        # all possible columns if None
        self._columns_selected = None
        self._all_columns_selected = True
        self._fitted = False
        self._set_hyperparameters(hyperparameters)
        self._hyperparameters = locals()['hyperparameters']

    @classmethod
    def get_target_type(cls):
        return cls._target_type

    @classmethod
    def get_feature_type(cls):
        return cls._feature_type

    @classmethod
    def get_req_target_column(cls):
        return cls._req_target_column

    @classmethod
    def get_exclude_target(cls):
        if cls._exclude_target is not None:
            return cls._exclude_target
        return True if not cls._changing_row_count else False

    @classmethod
    def get_min_features(cls):
        if cls._min_features is not None:
            return cls._min_features
        return (
            1
            if cls._req_target_column and cls._feature_type != FeatureType.ANY
            else 0
        )

    @classmethod
    def get_data_types(cls):
        return cls._data_types

    @classmethod
    def get_str_subtype(cls):
        return cls._str_subtype

    @classmethod
    def get_hyperparameter_space(cls):
        return cls._hyperparameter_space

    @classmethod
    def get_class_name(cls):
        return get_class_name(cls)

    @property
    def target_type(self):
        return self._target_type

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def dataset_type_threshold(self):
        return self._dataset_type_threshold

    @property
    def req_target_column(self):
        return self._req_target_column

    @property
    def exclude_target(self):
        return self._exclude_target

    @property
    def data_types(self):
        return self._data_types

    @property
    def str_subtype(self):
        return self._str_subtype

    @property
    def hyperparameter_space(self):
        return self._hyperparameter_space

    @property
    def target_column(self):
        return self._target_column

    @property
    def columns_input(self):
        return self._columns_input

    @property
    def columns_selected(self):
        return self._columns_selected

    @property
    def is_fitted(self):
        return self._fitted

    @property
    def class_name(self):
        return get_object_class_name(self)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    def copy(self):
        t = copy.deepcopy(self)
        t._fitted = False
        t._columns_selected = None
        return t

    def _set_hyperparameters(self, hyperparameters):
        for hp_name, hp_domain in self._hyperparameter_space.items():
            value = hyperparameters.get(hp_name, None)
            if value is None:
                value = hp_domain.default_value
            elif value != hp_domain.default_value:
                if not hp_domain.check_value_in_domain(value):
                    raise HyperparameterValueError(
                        value=value, hp_name=hp_name
                    )
            setattr(self, f'_hp_{hp_name}', value)
        for hp_name in hyperparameters:
            if hp_name not in self._hyperparameter_space:
                raise HyperparameterNameError(
                    caller_instance=self, hp_name=hp_name
                )

    def _select_target_column(self, df):
        df = select_columns(df, self._target_column)
        return df

    def _select_columns(self, df):
        if self._columns_selected:
            df = select_columns(df, self._columns_selected)
        elif self._columns_input:
            df = select_columns(df, self._columns_input)
        else:
            if self._data_types:
                df = select_data_types(df, self._data_types)
            df = FeatureType.select(df, self._feature_type)
            if self._exclude_target:
                df = exclude_columns(df, self._target_column)
            if self._str_subtype:
                df = StrSubtype.select(df, self._str_subtype)
        return df

    def _check_already_fitted(self):
        if self._fitted:
            raise AlreadyFittedError(caller_instance=self)

    def _check_not_fitted(self):
        if not self._fitted:
            raise NotFittedError(caller_instance=self)

    def _check_req_target(self, ds):
        if self._req_target_column and not ds.target_column:
            raise NoTargetError(caller_instance=self)

    def _check_target_excluded(self):
        if (
            self._exclude_target
            and self._target_column
            and self._columns_input
            and self._target_column in self._columns_input
        ):
            raise TransformTargetError(caller_instance=self)

    def _check_target_type(self, df):
        if not self._target_type.match_or_supertype_df(
            self._select_target_column(df)
        ):
            raise FeatureTypeError(expected_type=self._target_type)

    def _check_feature_type(self, df):
        if not self._feature_type.match_or_supertype_df(df):
            raise FeatureTypeError(expected_type=self._feature_type)

    def _check_data_types(self, df):
        if self._data_types and not data_types_match(
            exclude_columns(df, self._target_column), self._data_types
        ):
            raise DataTypeError(expected_types=self._data_types)

    def _check_df_empty(self, df):
        if df.is_empty():
            raise DatasetEmptyError(caller_instance=self)

    def _check_min_features(self, df):
        n_features = exclude_columns(df, self._target_column).width
        if n_features < self._min_features:
            raise InsufficientFeaturesError(
                caller_instance=self, min_features=self._min_features
            )

    @abstractmethod
    def _fit_df(self, df):
        pass

    @abstractmethod
    def _transform_df(self, df):
        pass

    def _inverse_transform_df(self, df):
        raise NotInversibleError(caller_instance=self)

    def fit(self, ds, columns=None):
        try:
            self._check_already_fitted()
            self._check_req_target(ds)
            df = ds.df
            if ds.target_column:
                self._target_column = ds.target_column
                self._check_target_type(df)
            if columns:
                self._columns_input = columns
            df = self._select_columns(df)
            self._check_df_empty(df)
            self._check_target_excluded()
            self._check_min_features(df)
            self._check_feature_type(df)
            self._check_data_types(df)
            self._columns_selected = df.columns
            if set(self._columns_selected) != set(ds.df.columns):
                self._all_columns_selected = False
            self._fit_df(df)
            self._fitted = True
            return self
        except Exception as e:
            # TODO temporary prints for debugging
            print('DEBUG:')
            print('ds', ds)
            print('fit df', df)
            raise FitError(caller_instance=self) from e

    def transform(self, ds):
        try:
            self._check_not_fitted()
            df = ds.df
            if not self._all_columns_selected and not self._changing_row_count:
                df_selection = self._select_columns(df)
                df = df.drop(df_selection.columns)
                df_selection = self._transform_df(df_selection)
                df = df.with_columns(df_selection)
            else:
                df = self._transform_df(df)
            ds = Dataset(
                df,
                dataset_type=ds.dataset_type,
                target_column=ds.target_column,
            )
            return ds
        except Exception as e:
            # TODO temporary prints for debugging
            print('DEBUG:')
            print('ds', ds)
            print('transform df', df)
            raise TransformError(caller_instance=self) from e

    def inverse_transform(self, ds):
        try:
            self._check_not_fitted()
            df = ds.df
            if not self._all_columns_selected and not self._changing_row_count:
                df_selection = self._select_columns(df)
                df = df.drop(df_selection.columns)
                df_selection = self._inverse_transform_df(df_selection)
                df = df.with_columns(df_selection)
            else:
                df = self._inverse_transform_df(df)
            ds = Dataset(
                df,
                dataset_type=ds.dataset_type,
                target_column=ds.target_column,
            )
            return ds
        except Exception as e:
            raise InverseTransformError(caller_instance=self) from e

    def fit_transform(self, ds, columns=None):
        ds = self.fit(ds, columns).transform(ds)
        return ds

    def __str__(self):
        str_params = ', '.join(
            f'{name}={value}' for name, value in self.hyperparameters.items()
        )
        return f'{self.class_name}({str_params})'

    def __repr__(self):
        return self.__str__()


class TransformationNoColParam(Transformation):

    def _select_columns(self, df):
        return df

    def fit(self, ds):
        return super().fit(ds, columns=None)

    def fit_transform(self, ds):
        ds = self.fit(ds).transform(ds)
        return ds


class TransformationTargetOnly(Transformation):
    _req_target_column = True
    _min_features = 0

    def fit(self, ds):
        return super().fit(ds, columns=ds.target_column)

    def fit_transform(self, ds):
        ds = self.fit(ds).transform(ds)
        return ds


class TransformationNumpy(Transformation, ABC):
    # TODO multiple target columns
    _numpy_reducing_dimensionality = False

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        self._restored_column_names = None

    def _select_columns(self, df):
        df_selection = super()._select_columns(df)
        if (
            self._target_column
            and self._target_column not in df_selection.columns
        ):
            df_selection = df_selection.with_columns(df[self._target_column])
        return df_selection

    def _check_feature_type(self, df):
        if self._target_column:
            df = exclude_columns(df, self._target_column)
        if not self._feature_type.match_or_supertype_df(df):
            raise FeatureTypeError(expected_type=self._feature_type)

    @abstractmethod
    def _fit_numpy(self, X, y=None):
        # numpy fitting eg. self.transformer.fit(X, y)
        pass

    @abstractmethod
    def _transform_numpy(self, X, y=None):
        # numpy transformation eg. self.transformer.transform(X, y)
        pass

    def _restore_column_names(self, df, X, X_selected):
        if self._restored_column_names:
            return self._restored_column_names
        column_names = []
        already_added = set()
        n = X.shape[1]
        m = X_selected.shape[1]
        for j in range(m):
            found_name = False
            for i in range(n):
                if i in already_added:
                    continue
                if np.all(X_selected[:, j] == X[:, i]):
                    column_names.append(df.columns[i])
                    already_added.add(i)
                    found_name = True
                    break
            if not found_name:
                column_names.append(f'column_{j}')
        self._restored_column_names = column_names
        return column_names

    def _numpy_transformed_to_pl(self, df, X, X_transformed):
        if not self._numpy_reducing_dimensionality:
            df = numpy_to_pl(X_transformed, columns=df.columns)
        else:
            columns = self._restore_column_names(df, X, X_transformed)
            df = numpy_to_pl(X_transformed, columns=columns)
        return df

    def _fit_df(self, df):
        if self._target_column:
            df_feature, df_target = feature_target_split(
                df, self._target_column
            )
            X = pl_to_numpy(df_feature)
            y = pl_to_numpy(
                df_target
            ).ravel()  # TODO no ravel if target columnS
        else:
            X = pl_to_numpy(df)
            y = None
        self._fit_numpy(X, y)

    def _transform_df(self, df):
        if self._target_column:
            df_feature, df_target = feature_target_split(
                df, self._target_column
            )
            X = pl_to_numpy(df_feature)
            y = pl_to_numpy(
                df_target
            ).ravel()  # TODO no ravel if target columnS
            X_transformed, y_transformed = self._transform_numpy(X, y)
            df_feature = self._numpy_transformed_to_pl(
                df_feature, X, X_transformed
            )
            df_target = numpy_to_pl(y_transformed, columns=df_target.columns)
            df = feature_target_join(df_feature, df_target)
        else:
            X = pl_to_numpy(df)
            X_transformed, _ = self._transform_numpy(X)
            df = self._numpy_transformed_to_pl(df, X, X_transformed)
        return df

    def _inverse_transform_df(self, df):
        if self._target_column:
            df_feature, df_target = feature_target_split(
                df, self._target_column
            )
            X = pl_to_numpy(df_feature)
            y = pl_to_numpy(
                df_target
            ).ravel()  # TODO no ravel if target columnS
            X_transformed, y_transformed = self._inverse_transform_numpy(X, y)
            df_feature = self._numpy_transformed_to_pl(
                df_feature, X, X_transformed
            )
            df_target = numpy_to_pl(y_transformed, columns=df_target.columns)
            df = feature_target_join(df_feature, df_target)
        else:
            X = pl_to_numpy(df)
            X_transformed, _ = self._inverse_transform_numpy(X)
            df = self._numpy_transformed_to_pl(df, X, X_transformed)
        return df


class TransformationSklearn(TransformationNumpy, ABC):
    # TODO multiple target columns

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        self.transformer = None

    def _fit_numpy(self, X, y=None):
        self.transformer.fit(X, y)

    def _transform_numpy(self, X, y=None):
        X_transformed = self.transformer.transform(X)
        return X_transformed, y

    def _inverse_transform_numpy(self, X, y=None):
        if not hasattr(self.transformer, 'inverse_transform'):
            raise NotInversibleError(caller_instance=self)
        X_transformed = self.transformer.inverse_transform(X)
        return X_transformed, y


class TransformationImblearn(TransformationNumpy, ABC):
    # TODO multiple target columns

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        self.resampler = None

    def _fit_numpy(self, X, y=None):
        pass

    def _transform_numpy(self, X, y=None):
        X_transformed, y_transformed = self.resampler.fit_resample(X, y)
        return X_transformed, y_transformed
