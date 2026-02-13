import pickle

from auto_prepper.utils.dataset_type import DatasetType
from auto_prepper.utils.exceptions import (
    AlreadyFittedError,
    DatasetTypeError,
    NotFittedError,
    NotInversibleError,
)


class Pipeline:

    def __init__(self, *transformations):
        self._transformations = [t.copy() for t in transformations]
        self._fitted = False

    @property
    def transformations(self):
        return self._transformations

    @property
    def is_fitted(self):
        return self._fitted

    @property
    def length(self):
        return len(self._transformations)

    def set_fitted(self, fitted):
        self._fitted = fitted

    def _check_already_fitted(self):
        if self._fitted:
            raise AlreadyFittedError(caller_instance=self)

    def _check_not_fitted(self):
        if not self._fitted:
            raise NotFittedError(caller_instance=self)

    def _check_dataset_train(self, ds):
        if ds.dataset_type != DatasetType.TRAIN:
            raise DatasetTypeError(expected_type='train')

    def add(self, transformation):
        self._transformations.append(transformation.copy())

    def add_as_is(self, transformation):
        self._transformations.append(transformation)

    def copy(self):
        p = Pipeline(t.copy() for t in self._transformations)
        return p

    def transform(self, ds):
        self._check_not_fitted()
        for t in self._transformations:
            if ds.dataset_type.is_within_threshold(t.dataset_type_threshold):
                ds = t.transform(ds)
        return ds

    def fit_transform(self, ds):
        self._check_already_fitted()
        self._check_dataset_train(ds)
        for t in self._transformations:
            t.fit(ds)
            if ds.dataset_type.is_within_threshold(t.dataset_type_threshold):
                ds = t.transform(ds)
        self._fitted = True
        return ds

    def inverse_transform(self, ds):
        self._check_not_fitted()
        for t in self._transformations[::-1]:
            if ds.target_column in t.selected_columns:
                # inversing only if target was transformed
                try:
                    ds = t.inverse_transform(ds)
                except NotInversibleError:
                    pass
        return ds

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def report(self):
        r = [{t.class_name: t.hyperparameters} for t in self._transformations]
        return r

    def extend(self, pipeline):
        # TODO check fitted?
        self._transformations.extend(pipeline.transformations)
        return self

    def __str__(self):
        s = 'Pipeline ['
        for t in self._transformations:
            s += f'\n    {str(t)},'
        s += '\n]'
        return s

    def __repr__(self):
        return self.__str__()
