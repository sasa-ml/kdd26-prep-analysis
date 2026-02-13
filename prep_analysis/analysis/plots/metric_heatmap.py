import gc

import plotly.graph_objects as go
import polars as pl

from prep_analysis.analysis.helpers_df import df_filter_perf
from prep_analysis.experiments.parse_results import parse_results


def transform(df):
    all_models = sorted(list(df['model'].unique()))
    all_pipelines = sorted(list(df['pipeline'].unique()))

    df = df.filter(pl.col('task').is_in(['binary', 'multiclass']))

    df = df.rename(
        {
            'accuracy_score': 'accuracy',
            'balanced_accuracy_score': 'balanced_accuracy',
        }
    )

    df = df.melt(
        id_vars=['i_dataset', 'model', 'pipeline'],
        value_vars=[
            'accuracy',
            'balanced_accuracy',
        ],
        variable_name='acc_metric',
        value_name='acc_score',
    )

    df_acc = (
        df.filter(pl.col('acc_score').is_not_nan())
        .filter(pl.col('acc_metric') == 'accuracy')
        .sort('acc_score', descending=True)
        .group_by('i_dataset', maintain_order=True)
        .first()
    )

    df_bal = (
        df.filter(pl.col('acc_score').is_not_nan())
        .filter(pl.col('acc_metric') == 'balanced_accuracy')
        .sort('acc_score', descending=True)
        .group_by('i_dataset', maintain_order=True)
        .first()
    )

    df = pl.concat([df_acc, df_bal])

    wins = {'acc_metric': [], 'model': [], 'pipeline': [], 'win_count': []}

    for acc_metric in sorted(list(df['acc_metric'].unique())):
        for model in all_models:
            for pipeline in all_pipelines:
                win_count = df.filter(
                    (pl.col('acc_metric') == acc_metric)
                    & (pl.col('model') == model)
                    & (pl.col('pipeline') == pipeline)
                ).height
                wins['acc_metric'].append(acc_metric)
                wins['model'].append(model)
                wins['pipeline'].append(pipeline)
                wins['win_count'].append(win_count)

    df = pl.DataFrame(wins)

    df_acc = df.filter(pl.col('acc_metric') == 'accuracy')
    df_bal = df.filter(pl.col('acc_metric') == 'balanced_accuracy')

    df_acc = df_acc.sort(['model', 'pipeline'])
    df_bal = df_bal.sort(['model', 'pipeline'])

    df = df_bal.with_columns(
        (pl.col('win_count') - df_acc['win_count']).alias('diff_win_count')
    ).drop('acc_metric', 'win_count')

    return df


def plot(df, output_dir=None):
    fig = go.Figure()

    mat = df.pivot(
        index='model',
        columns='pipeline',
        values='diff_win_count',
    ).sort('model')

    values = mat.select(pl.exclude('model')).to_numpy()
    pipelines = [c for c in mat.columns if c != 'model']
    models = sorted(mat['model'].to_list())

    fig.add_trace(
        go.Heatmap(
            z=values,
            x=pipelines,
            y=models,
            colorscale='PiYG',
            zmid=0,
            text=values,
            texttemplate='%{text}',
            showscale=False,
            hovertemplate=(
                'Model: %{y}<br>'
                'Pipeline: %{x}<br>'
                'Win count diff: %{text}<extra></extra>'
            ),
        ),
    )

    fig.update_yaxes(autorange='reversed')

    fig.update_layout(
        width=500,
        height=600,
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=60, b=0),
        title=(
            20 * '&nbsp;'
            + 'Win count difference between balanced<br>'
            + 32 * '&nbsp;'
            + 'accuracy and accuracy'
        ),
    )

    fig.show()
    if output_dir:
        fig.write_image(f'{output_dir}metric_heatmap.png')
    del fig
    gc.collect()


def transform_and_plot(df, output_dir=None):
    df = df_filter_perf(df)
    df = transform(df)
    # print(df)
    plot(df, output_dir)


if __name__ == '__main__':

    F_PATH = '../../../Data/prep-analysis/results.txt'
    df = parse_results(F_PATH)
    print(df)
    transform_and_plot(df)
