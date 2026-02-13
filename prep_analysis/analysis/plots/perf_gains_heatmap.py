import gc

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl

from prep_analysis.analysis.helpers_df import (
    df_add_level_diffs,
    df_add_levels,
    df_filter_perf,
)
from prep_analysis.experiments.parse_results import parse_results


def transform(df):
    df = df_add_levels(df)

    df = df_add_level_diffs(df)

    df = df.filter(
        pl.col('pipeline').is_in(
            ['1_No_prep', '2_diff_minimal_no_prep', '3_diff_best_minimal']
        )
    )

    return df


def plot(df, output_dir=None):
    fig = go.Figure()

    pipelines = sorted(list(df['pipeline'].unique()))

    all_models = df.select('model').unique().sort('model')
    all_datasets = df.select('i_dataset').unique().sort('i_dataset')
    full_index = all_models.join(all_datasets, how='cross')

    matrices = {
        p: (
            full_index.join(
                df.filter(pl.col('pipeline') == p),
                on=['model', 'i_dataset'],
                how='left',
            )
            .with_columns(pl.col('extra_score'))
            .pivot(index='model', columns='i_dataset', values='extra_score')
            .sort('model', descending=True)
        )
        for p in pipelines
    }

    fig = sp.make_subplots(
        rows=len(pipelines),
        cols=1,
        subplot_titles=[
            'No data preparation',
            'Difference between minimal and no preparation',
            'Difference between best and minimal preparation',
        ],
        vertical_spacing=0.06,
    )

    colorscales = {
        '1_No_prep': 'Purples',
        '2_diff_minimal_no_prep': 'Blues',
        '3_diff_best_minimal': 'Greens',
    }

    for i, p in enumerate(pipelines, start=1):
        mat = matrices[p]

        models = mat['model'].to_list()
        datasets = [c for c in mat.columns if c != 'model']
        values = mat.select(datasets).to_numpy()
        values_text = np.round(values, 2)
        values_text = np.where(values_text == 1.00, 0.99, values_text)
        values_text = np.char.lstrip(np.char.mod('%.2f', values_text), '0')
        value_colors = values
        if p in ['2_diff_minimal_no_prep', '3_diff_best_minimal']:
            values_text = np.char.add('+', values_text)
        if p == '3_diff_best_minimal':
            value_colors = np.sqrt(np.sqrt(values))

        fig.add_trace(
            go.Heatmap(
                z=value_colors,
                x=datasets,
                y=models,
                colorscale=colorscales[p],
                text=values_text,
                texttemplate='%{text}',
                showscale=False,
                hovertemplate=(
                    'Index: %{x}<br>'
                    'Score: %{text}<br>'
                    'Model: %{y}<br>'
                    f'Pipeline: {p}<extra></extra>'
                ),
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        width=1600,
        height=300 * (len(pipelines) + 1),
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=20, b=0),
        # title=(
        #    'Model performance improvement between data preparation levels'
        # ),
    )

    fig.show()
    if output_dir:
        fig.write_image(f'{output_dir}perf_gains_heatmap.png')
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
