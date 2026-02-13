import polars as pl


def df_filter_perf(df):
    df = df.select(
        'i_dataset',
        'task',
        'metric',
        'extra_metric',
        'pipeline',
        'model',
        'score',
        'score_std',
        'extra_score',
        'extra_score_std',
        'accuracy_score',
        'balanced_accuracy_score',
    ).sort(
        by=[
            'i_dataset',
            'pipeline',
            'model',
        ]
    )

    return df


def df_add_levels(df):
    df_minimal = (
        df.filter(pl.col('pipeline').is_in(['1_No_prep', '2_Basic']))
        .sort('i_dataset', 'model', 'pipeline')
        .fill_nan(None)
        .fill_null(strategy='backward')
        .filter(pl.col('pipeline') == '1_No_prep')
    )
    df_minimal = df_minimal.with_columns(pl.lit('2_Minimal').alias('pipeline'))

    df_best = (
        df.fill_nan(0)
        .sort('extra_score', descending=True)
        .group_by(['i_dataset', 'model'], maintain_order=True)
        .first()
    )
    df_best = df_best.with_columns(pl.lit('3_Best').alias('pipeline'))
    df_best = df_best.select(df.columns)

    df = pl.concat([df, df_minimal, df_best])

    return df


def df_add_level_diffs(df):
    df_no_prep = df.filter(pl.col('pipeline') == '1_No_prep')
    df_minimal = df.filter(pl.col('pipeline') == '2_Minimal')
    df_best = df.filter(pl.col('pipeline') == '3_Best')

    df_no_prep = df_no_prep.sort(['i_dataset', 'model', 'pipeline'])
    df_minimal = df_minimal.sort(['i_dataset', 'model', 'pipeline'])
    df_best = df_best.sort(['i_dataset', 'model', 'pipeline'])

    df_diff_minimal = df_minimal.with_columns(
        (pl.col('extra_score') - df_no_prep['extra_score'].fill_nan(0))
        .clip(lower_bound=0)
        .alias('extra_score')
    )
    df_diff_minimal = df_diff_minimal.with_columns(
        pl.lit('2_diff_minimal_no_prep').alias('pipeline')
    )
    df_diff_best = df_best.with_columns(
        (pl.col('extra_score') - df_minimal['extra_score'])
        .clip(lower_bound=0)
        .alias('extra_score')
    )
    df_diff_best = df_diff_best.with_columns(
        pl.lit('3_diff_best_minimal').alias('pipeline')
    )

    df = pl.concat([df, df_diff_minimal, df_diff_best])

    return df


def df_get_wins_by_pipeline(df):
    df = (
        df.filter(pl.col('extra_score').is_not_nan())
        .filter(pl.col('extra_score') > 0)
        .sort('extra_score', descending=True)
        .group_by(['i_dataset', 'pipeline'], maintain_order=True)
        .first()
    )

    wins = {'model': [], 'pipeline': [], 'win_count': []}

    for model in sorted(list(df['model'].unique())):
        for pipeline in sorted(list(df['pipeline'].unique())):
            win_count = df.filter(
                (pl.col('model') == model) & (pl.col('pipeline') == pipeline)
            ).height
            wins['model'].append(model)
            wins['pipeline'].append(pipeline)
            wins['win_count'].append(win_count)

    df = pl.DataFrame(wins)

    return df


def df_get_wins_overall(df):
    df = (
        df.filter(pl.col('extra_score').is_not_nan())
        .filter(pl.col('extra_score') > 0)
        .sort('extra_score', descending=True)
        .group_by('i_dataset', maintain_order=True)
        .first()
    )

    wins = {'model': [], 'pipeline': [], 'win_count': []}

    for model in sorted(list(df['model'].unique())):
        for pipeline in sorted(list(df['pipeline'].unique())):
            win_count = df.filter(
                (pl.col('model') == model) & (pl.col('pipeline') == pipeline)
            ).height
            wins['model'].append(model)
            wins['pipeline'].append(pipeline)
            wins['win_count'].append(win_count)

    df = pl.DataFrame(wins)

    return df
