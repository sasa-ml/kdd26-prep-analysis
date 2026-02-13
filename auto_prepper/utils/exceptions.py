from auto_prepper.utils.helpers import get_object_class_name


class NotFittedError(Exception):

    def __init__(self, caller_instance):
        message = (
            f'{get_object_class_name(caller_instance)} not fitted yet. Call '
            + 'fit first or fit_transform instead.'
        )
        super().__init__(message)


class AlreadyFittedError(Exception):

    def __init__(self, caller_instance):
        message = (
            f'{get_object_class_name(caller_instance)} is already fitted.'
        )
        super().__init__(message)


class FitError(Exception):

    def __init__(self, caller_instance):
        message = f'Error fitting {get_object_class_name(caller_instance)}.'
        super().__init__(message)


class TransformError(Exception):

    def __init__(self, caller_instance):
        message = (
            'Error applying transformation '
            + f'{get_object_class_name(caller_instance)}.'
        )
        super().__init__(message)


class InverseTransformError(Exception):

    def __init__(self, caller_instance):
        message = (
            'Error inversing transformation '
            + f'{get_object_class_name(caller_instance)}.'
        )
        super().__init__(message)


class NotInversibleError(Exception):

    def __init__(self, caller_instance):
        message = (
            f'{get_object_class_name(caller_instance)} transformation '
            + 'not inversible.'
        )
        super().__init__(message)


class DatasetTypeError(Exception):

    def __init__(self, expected_type):
        message = f'Expecting dataset of type {expected_type}.'
        super().__init__(message)


class TargetColumnError(Exception):

    def __init__(self, column):
        message = (
            f'Column {column} not found. Target column must be an '
            + 'existing dataframe column.'
        )
        super().__init__(message)


class TransformTargetError(Exception):

    def __init__(self, caller_instance):
        message = (
            f'{get_object_class_name(caller_instance)} cannot be used to '
            + 'transform the target column.'
        )
        super().__init__(message)


class FeatureTypeError(Exception):

    def __init__(self, expected_type):
        message = f'Expecting column(s) of feature type {expected_type}.'
        super().__init__(message)


class DataTypeError(Exception):

    def __init__(self, expected_types):
        message = f'Expecting column(s) of data types {expected_types}.'
        super().__init__(message)


class NoTargetError(Exception):

    def __init__(self, caller_instance):
        message = f'{caller_instance} requires a dataset with a target column.'
        super().__init__(message)


class DatasetEmptyError(Exception):

    def __init__(self, caller_instance):
        message = (
            f'Cannot perform {caller_instance} transformation on an '
            + 'empty dataset selection.'
        )
        super().__init__(message)


class InsufficientFeaturesError(Exception):

    def __init__(self, caller_instance, min_features):
        message = (
            f'Cannot perform {caller_instance} transformation with fewer than '
            + f'{min_features} features.'
        )
        super().__init__(message)


class UnsupportedFormatError(Exception):

    def __init__(self, message='Unsupported input or output format.'):
        super().__init__(message)


class OptimizerEvalError(Exception):

    def __init__(self, message='Optimizer evaluation function is undefined.'):
        super().__init__(message)


class HyperparameterValueError(Exception):

    def __init__(self, value, hp_name=None):
        message = f'Value {value} is not in'
        if hp_name:
            message += f' {hp_name}'
        message += ' hyperparameter domain.'
        super().__init__(message)


class HyperparameterNameError(Exception):

    def __init__(self, caller_instance, hp_name):
        message = (
            f'{hp_name} is not a valid hyperparameter for {caller_instance}.'
        )
        super().__init__(message)


class HyperparameterRangeError(Exception):

    def __init__(self, hp_values):
        message = (
            f'Invalid hyperparmeter range {hp_values}. Range should be '
            + 'in format [min_value, max_value].'
        )
        super().__init__(message)


class HyperparameterStepError(Exception):

    def __init__(self, hp_iter_step, hp_values, hp_type):
        message = (
            f'Invalid iteration step {hp_iter_step} for {hp_type} '
            + f'hyperparameter value range {hp_values}.'
        )
        super().__init__(message)
