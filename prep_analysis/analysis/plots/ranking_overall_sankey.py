import gc

import numpy as np
import plotly.graph_objects as go
import polars as pl

from prep_analysis.analysis.helpers_df import (
    df_filter_perf,
    df_get_wins_overall,
)
from prep_analysis.analysis.helpers_plot import model_color_map
from prep_analysis.experiments.parse_results import parse_results


def transform(df):
    df = df_get_wins_overall(df)

    return df


def plot(df, output_dir=None):
    fig = go.Figure()

    df = df.sort(by=['model', 'pipeline'])

    models = sorted(df['model'].unique().to_list())
    pipelines = sorted(df['pipeline'].unique().to_list())

    model_win_count = [
        df.filter(pl.col('model') == model)['win_count'].sum()
        for model in models
    ]
    pipeline_win_count = [
        df.filter(pl.col('pipeline') == pipeline)['win_count'].sum()
        for pipeline in pipelines
    ]

    # nodes
    nodes = models + pipelines
    labels = [f'{models[i]}: {model_win_count[i]}' for i in range(len(models))]
    labels.extend(
        [
            f'{pipelines[i]}: {pipeline_win_count[i]}'
            for i in range(len(pipelines))
        ]
    )
    colors = [model_color_map[model] for model in models] + ['white'] * len(
        pipelines
    )

    # links
    sources = []
    targets = []
    values = []
    link_colors = []

    node_index = {node: i for i, node in enumerate(nodes)}
    x = [0.001 for i in range(len(models))]
    x.append([0.999 for i in range(len(pipelines))])
    y_left = list(np.linspace(1, 0, len(models)))
    y_right = list(np.linspace(1, 0, len(pipelines)) + 0.05)
    y = y_left + y_right

    for row in df.iter_rows(named=True):
        model = row['model']
        pipeline = row['pipeline']
        win_count = row['win_count']

        sources.append(node_index[model])
        targets.append(node_index[pipeline])
        values.append(win_count)
        link_colors.append(model_color_map[model])

    fig.add_trace(
        go.Sankey(
            arrangement='snap',
            node=dict(
                pad=20,
                thickness=1,
                label=labels,
                color=colors,
                line=dict(width=0),
                y=y,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate=(
                    'Model: %{source.label}<br>'
                    'Pipeline: %{target.label}<br>'
                    'Win count: %{value}<extra></extra>'
                ),
            ),
        )
    )

    fig.update_layout(
        height=600,  # 800,
        width=600,  # 800,
        title=(30 * '&nbsp;' + 'Winning model and pipeline pairs'),
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=30, b=0),
    )

    fig.show()
    if output_dir:
        fig.write_image(f'{output_dir}ranking_overall_sankey.png')
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
