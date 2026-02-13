import polars as pl
import polars.selectors as cs


def select_data_types(df, data_types):
    columns = df.select(cs.by_dtype(*data_types)).columns
    for typ in data_types:
        if typ in NESTED_TYPES:
            for col in df:
                if col.dtype.base_type() == typ:
                    columns.append(col.name)
    df = df.select(columns)
    return df


def data_types_match(df, data_types):
    return set(select_data_types(df, data_types).columns) == set(df.columns)


SUPPORTED_TYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Decimal,
    pl.Float32,
    pl.Float64,
    pl.Utf8,
    pl.Categorical,
    pl.Enum,
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
]

NUMERIC_TYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Decimal,
    pl.Float32,
    pl.Float64,
]

INTEGER_TYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
]

UINTEGER_TYPES = [
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
]

FLOAT_TYPES = [
    pl.Decimal,
    pl.Float32,
    pl.Float64,
]

STRING_TYPES = [
    pl.Utf8,
    pl.Categorical,
    pl.Enum,
]

TEMPORAL_TYPES = [
    pl.Date,
    pl.Datetime,
    pl.Duration,
    pl.Time,
]

NESTED_TYPES = [
    pl.List,
    pl.Array,
    pl.Struct,
]
