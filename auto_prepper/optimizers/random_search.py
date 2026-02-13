import random
from math import inf

from auto_prepper.core.optimizer import Optimizer
from auto_prepper.core.pipeline import Pipeline
from auto_prepper.core.pool import TransformationPool


class RandomSearch(Optimizer):
    _no_eval = False

    def __init__(
        self,
        ds,
        eval_func=None,
        min_length=1,
        max_length=10,
        max_iterations=10,
        default_hyperparameters=True,
    ):
        super().__init__(ds, eval_func)
        self._min_length = min_length
        self._max_length = max_length
        self._max_iterations = max_iterations
        self._default_hyperparameters = default_hyperparameters
        if not self._eval_func:
            self._eval_func = self._pipeline_check_helper
            self._no_eval = True

    def _pipeline_check_helper(self, p):
        return 0

    def _random_pipeline(self):
        pipeline = Pipeline()
        if self._max_length >= 0:
            length = random.randint(self._min_length, self._max_length)
            for _ in range(length):
                t_pool = TransformationPool(self._ds).pool
                if not t_pool:
                    break
                transformations = sum(
                    [list(category.keys()) for category in t_pool.values()], []
                )
                transformation = random.choice(transformations)
                hyperparameter_space = (
                    transformation.get_hyperparameter_space()
                )
                if hyperparameter_space and not self._default_hyperparameters:
                    hyperparameters = {
                        hp_name: hp_domain.random_value()
                        for hp_name, hp_domain in hyperparameter_space.items()
                    }
                    pipeline.add(transformation(**hyperparameters))
                else:
                    pipeline.add(transformation())
        return pipeline

    def optimize_pipeline(self):
        pipeline = Pipeline()
        pipeline.fit_transform(self._ds)
        score = -inf
        for _ in range(self._max_iterations):
            p = self._random_pipeline()
            s = self._evaluate(p)
            if s > score:
                pipeline = p
                score = s
                if self._no_eval:
                    break
        return pipeline
