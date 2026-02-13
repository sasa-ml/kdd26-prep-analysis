import gc

import plotly.graph_objects as go
import polars as pl

from prep_analysis.analysis.helpers_df import (
    df_add_levels,
    df_filter_perf,
    df_get_wins_by_pipeline,
)
from prep_analysis.analysis.helpers_plot import model_color_map
from prep_analysis.experiments.parse_results import parse_results


def transform(df):
    df = df_add_levels(df)

    df = df.filter(
        pl.col('pipeline').is_in(['1_No_prep', '2_Minimal', '3_Best'])
    )

    df = df_get_wins_by_pipeline(df)

    return df


def plot(df, output_dir=None):
    fig = go.Figure()

    # df = df.sort(by=['win_count', 'model'])
    df = df.sort(by='model')
    for row in df.iter_rows(named=True):
        model = row['model']
        fig.add_trace(
            go.Bar(
                y=[row['pipeline']],
                x=[row['win_count']],
                name=model,
                opacity=1,
                orientation='h',
                marker=dict(
                    color=model_color_map[model],
                ),
                text=[row['win_count']],
                textposition='inside',
                insidetextanchor='middle',
                textangle=0,
                showlegend=False,
                hovertemplate=(
                    f'Model: {model}<br>'
                    'DPrep level: %{x}<br>'
                    'Win count: %{y}<extra></extra>'
                ),
                width=0.6,
            )
        )

    models = sorted(list(df['model'].unique()))
    for model in models:
        # dummy trace for model legend
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=model,
                marker=dict(
                    color=model_color_map[model],
                ),
            )
        )

    sorted_pipelines = sorted(list(df['pipeline'].unique()), reverse=True)

    fig.update_layout(
        barmode='relative',
        height=600,  # 800,
        width=600,  # 800,
        # width=1600,
        # title=('Model ranking by data preparation level'),
        yaxis=dict(
            title='Level of data preparation',
            type='category',
            categoryorder='array',
            categoryarray=sorted_pipelines,
            tickmode='array',
            tickvals=sorted_pipelines,
            ticktext=sorted_pipelines,
        ),
        xaxis=dict(
            title='Win count across datasets',
            # type='log',
        ),
        legend=dict(
            orientation='h',
            yanchor='top',
            # y=-0.15,
            xanchor='center',
            x=0.5,
        ),
        legend_title_text='Model',
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.show()
    if output_dir:
        fig.write_image(f'{output_dir}ranking_level_bar_h.png')
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
