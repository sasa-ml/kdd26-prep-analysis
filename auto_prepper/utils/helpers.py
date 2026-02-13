import inspect
from pathlib import Path

import polars as pl


def pl_to_numpy(df):
    return df.to_numpy()


def numpy_to_pl(data, columns=None):
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    return pl.from_numpy(data, schema=columns, orient='row')


def select_columns(df, columns):
    if columns:
        df = df.select(columns)
    return df


def exclude_columns(df, columns):
    if columns:
        df = df.select(pl.exclude(columns))
    return df


def safe_json_decode(expr):
    return (
        expr.str.replace(r'(?i)\bnan\b', 'null', literal=False)
        .str.replace(r'(?i)\binfinity\b', 'null', literal=False)
        .str.replace(r'(?i)\b-infinity\b', 'null', literal=False)
        .str.json_decode()
        .cast(pl.Float64, strict=False)
    )


def make_dir_if_not_exists(path):
    p = Path(path)
    target = p if p.suffix == '' else p.parent
    target.mkdir(parents=True, exist_ok=True)


def get_class_name(instance):
    return instance.__name__


def get_object_class_name(instance):
    return instance.__class__.__name__


def get_class_parameter_names(cls):
    signature = inspect.signature(cls.__init__)
    param_names = [
        p.name for p in signature.parameters.values() if p.name != 'self'
    ]
    return param_names


def get_class_parameters(instance):
    param_names = get_class_parameter_names(get_class_name(instance))
    params = {p: getattr(instance, p, None) for p in param_names}
    return params


def get_method_parameter_names(cls, method_name):
    method = getattr(cls, method_name, None)
    if method is None:
        raise ValueError(
            f'Method {method_name} not found in class {cls.__name__}'
        )
    signature = inspect.signature(method)
    param_names = [
        p.name for p in signature.parameters.values() if p.name != 'self'
    ]
    return param_names


def get_class_subclasses(cls, recursion=False):
    subclasses = cls.__subclasses__()
    if recursion:
        for subclass in subclasses:
            subclasses += get_class_subclasses(subclass)
    return subclasses


def get_class_superclasses(cls, recursion=False):
    if not recursion:
        superclasses = [c for c in cls.__bases__ if c != object]
    else:
        superclasses = [
            c for c in inspect.getmro(cls) if c not in {cls, object}
        ]
    return superclasses
