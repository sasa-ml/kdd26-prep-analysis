from abc import ABC, abstractmethod
from math import inf

from auto_prepper.utils.dataset_type import DatasetType
from auto_prepper.utils.exceptions import DatasetTypeError, OptimizerEvalError


class Optimizer(ABC):

    def __init__(self, ds, eval_func=None):
        if ds.dataset_type != DatasetType.TRAIN:
            raise DatasetTypeError(expected_type='train')
        self._ds = ds
        self._eval_func = eval_func

    def _evaluate(self, pipeline):
        if not self._eval_func:
            raise OptimizerEvalError()
        try:
            return self._eval_func(pipeline.fit_transform(self._ds))
        except Exception as e:
            # TODO log exception somewhere?
            print(e)  # placeholder
            return -inf

    @abstractmethod
    def optimize_pipeline(self):
        pass
