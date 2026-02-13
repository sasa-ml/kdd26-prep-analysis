import polars as pl

from auto_prepper.utils.exceptions import UnsupportedFormatError
from auto_prepper.utils.feature_type import FeatureType
from auto_prepper.utils.helpers import pl_to_numpy


def save_df(df, format, path):
    match format:
        case 'csv':
            return to_csv(df)
        case 'parquet':
            return to_parquet(df, path)
        case 'json':
            return to_json(df, path)
        case _:
            raise UnsupportedFormatError()


def to_polars(df):
    return df


def to_dict(df):
    return df.to_dict(as_series=False)


def to_csv(df, file):
    return df.write_csv(file=file)


def to_parquet(df, file):
    return df.write_parquet(file=file)


def to_json(df, file):
    return df.write_json(file=file)


def to_pandas(df):
    categorical_columns = FeatureType.select(df, FeatureType.CATEGORICAL)
    df = df.with_columns(
        [pl.col(col).cast(pl.Categorical) for col in categorical_columns]
    )
    return df.to_pandas()


def to_numpy(df):
    return pl_to_numpy(df)
