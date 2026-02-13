import gc

import plotly.graph_objects as go

from prep_analysis.analysis.helpers_df import df_filter_perf
from prep_analysis.analysis.helpers_plot import (
    category_orders,
    model_color_map,
    pipeline_pattern_map,
)
from prep_analysis.experiments.parse_results import parse_results


def transform(df):
    df = df.fill_nan(0)

    return df


def plot(df, output_dir=None):
    fig = go.Figure()

    df = df.sort(by='extra_score', descending=True)
    for row in df.iter_rows(named=True):
        model = row['model']
        pipeline = row['pipeline']
        fig.add_trace(
            go.Bar(
                x=[row['i_dataset']],
                y=[row['extra_score']],
                name=f'{model}, {pipeline}',
                opacity=1,
                marker=dict(
                    color=model_color_map[model],
                    pattern=dict(
                        shape=pipeline_pattern_map[pipeline],
                        # fgcolor='white',
                        size=9,
                    ),
                ),
                showlegend=False,
                hovertemplate=(
                    'Index: %{x}<br>'
                    'Score: %{y}<br>'
                    f'Model: {model}<br>'
                    f'Pipeline: {pipeline}<extra></extra>'
                ),
            )
        )

    models = [m for m in category_orders['model'] if m in df['model'].unique()]
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

    pipelines = [
        p for p in category_orders['pipeline'] if p in df['pipeline'].unique()
    ]
    for pipeline in pipelines:
        # dummy trace for pipeline legend
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=pipeline,
                marker=dict(
                    color='white',
                    pattern=dict(
                        shape=pipeline_pattern_map[pipeline], fgcolor='black'
                    ),
                ),
            )
        )

    fig.update_layout(
        barmode='overlay',
        height=500,  # 800,
        width=1200,  # 600,
        # title=(
        #    'Performance of different model and pipeline combinations '
        #    + 'across datasets'
        # ),
        xaxis=dict(
            title='Dataset index',
            type='category',
            categoryorder='array',
            categoryarray=category_orders['i_dataset'],
            tickmode='array',
            tickvals=category_orders['i_dataset'],
            ticktext=category_orders['i_dataset'],
        ),
        yaxis=dict(
            title='Evaluation metric score',
            # type='log',
        ),
        legend=dict(
            orientation='h',
            yanchor='top',
            # y=-0.15,
            xanchor='center',
            x=0.5,
        ),
        legend_title_text='Model & Pipeline',
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.show()
    if output_dir:
        fig.write_image(f'{output_dir}perf_bar.png')
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
