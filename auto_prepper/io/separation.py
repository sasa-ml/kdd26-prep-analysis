import polars as pl
from sklearn.model_selection import (
    train_test_split as sklearn_train_test_split,
)

from auto_prepper.utils.helpers import numpy_to_pl, pl_to_numpy


def train_test_split(df, test_fraction=0.2, shuffle=True):
    if shuffle:
        df = df.sample(fraction=1, shuffle=shuffle)
    n = df.height
    test_size = round(test_fraction * n)
    df_train = df.head(-test_size)
    df_test = df.tail(test_size)
    return df_train, df_test


def train_test_split_stratified(df, columns, test_fraction=0.2, shuffle=True):
    df_strat = df[columns]
    data = pl_to_numpy(df)
    strat = pl_to_numpy(df_strat)
    train, test = sklearn_train_test_split(
        data, test_size=test_fraction, shuffle=shuffle, stratify=strat
    )
    df_train = numpy_to_pl(train, columns=df.columns)
    df_test = numpy_to_pl(test, columns=df.columns)
    return df_train, df_test


def feature_target_split(df, target_columns):
    df_target = df.select(target_columns)
    df_feature = df.select(pl.exclude(target_columns))
    return df_feature, df_target
