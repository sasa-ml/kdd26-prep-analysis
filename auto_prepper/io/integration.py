import polars as pl


def feature_target_join(df_feature, df_target):
    df = df_feature.with_columns(df_target)
    return df


def df_join_vertical(*dfs):
    df = pl.concat(list(dfs), how='vertical_relaxed', rechunk=True)
    return df


def df_join_horizontal(*dfs):
    df = pl.concat(list(dfs), how='horizontal', rechunk=True)
    return df


def df_join_diagonal(*dfs):
    df = pl.concat(list(dfs), how='diagonal_relaxed', rechunk=True)
    return df
