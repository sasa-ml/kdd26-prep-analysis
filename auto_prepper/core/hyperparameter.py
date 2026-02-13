import random
from abc import ABC, abstractmethod
from enum import Enum

from auto_prepper.utils.exceptions import (
    HyperparameterRangeError,
    HyperparameterStepError,
    HyperparameterValueError,
)


class HyperparameterType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    CATEGORICAL = 2

    def is_numeric(self):
        return self in {
            HyperparameterType.DISCRETE,
            HyperparameterType.CONTINUOUS,
        }

    def is_discrete(self):
        return self == HyperparameterType.DISCRETE

    def is_continuous(self):
        return self == HyperparameterType.CONTINUOUS

    def is_categorical(self):
        return self == HyperparameterType.CATEGORICAL

    def __str__(self):
        if self == HyperparameterType.DISCRETE:
            return 'discrete'
        if self == HyperparameterType.CONTINUOUS:
            return 'continuous'
        return 'categorical'

    def __repr__(self):
        return self.__str__()


class HyperparameterDomain(ABC):

    def __init__(self, values, default_value):
        self._values = values
        self._default_value = default_value
        if not self.check_value_in_domain(self._default_value):
            raise HyperparameterValueError(value=self._default_value)

    @property
    def hp_type(self):
        return self._hp_type

    @property
    def values(self):
        return self._values

    @property
    def default_value(self):
        return self._default_value

    @abstractmethod
    def random_value(self):
        pass

    @abstractmethod
    def iter_values(self):
        pass

    @abstractmethod
    def check_value_in_domain(self, val):
        pass


class HPDomainCategorical(HyperparameterDomain):
    _hp_type = HyperparameterType.CATEGORICAL

    def random_value(self):
        return random.choice(self._values)

    def iter_values(self):
        for val in self._values:
            yield val

    def check_value_in_domain(self, val):
        return val in self._values


class HPDomainDiscrete(HyperparameterDomain):
    _hp_type = HyperparameterType.DISCRETE

    def __init__(self, values, default_value):
        if (
            len(values) != 2
            or not isinstance(values[0], int)
            or not isinstance(values[1], int)
            or values[0] > values[1]
        ):
            raise HyperparameterRangeError(hp_values=values)
        super().__init__(values, default_value)

    def random_value(self):
        return random.randint(*self._values)

    def iter_values(self, step=1):
        if not step or step != round(step):
            raise HyperparameterStepError(
                hp_iter_step=step,
                hp_values=self._values,
                hp_type=self._hp_type,
            )
        for i in range(self._values[0], self._values[1] + step, step):
            yield i

    def check_value_in_domain(self, val):
        return self._values[0] <= val <= self._values[1] and round(val) == val


class HPDomainContinuous(HyperparameterDomain):
    _hp_type = HyperparameterType.CONTINUOUS

    def __init__(self, values, default_value):
        if (
            len(values) != 2
            or not isinstance(values[0], (int, float))
            or not isinstance(values[1], (int, float))
            or values[0] > values[1]
        ):
            raise HyperparameterRangeError(hp_values=values)
        super().__init__(values, default_value)

    def random_value(self):
        return random.uniform(*self._values)

    def iter_values(self, step=0.1):
        if not step:
            raise HyperparameterStepError(
                hp_iter_step=step,
                hp_values=self._values,
                hp_type=self._hp_type,
            )
        for i in range(self._values[0], self._values[1] + step, step):
            yield i

    def check_value_in_domain(self, val):
        return self._values[0] <= val <= self._values[1]
