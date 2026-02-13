from prep_analysis.analysis.helpers_df import df_filter_perf
from prep_analysis.analysis.numbers_analysis import analyze_numbers
from prep_analysis.analysis.plots.metric_heatmap import (
    transform_and_plot as metric_heatmap,
)
from prep_analysis.analysis.plots.perf_bar import (
    transform_and_plot as perf_bar,
)
from prep_analysis.analysis.plots.perf_gains_heatmap import (
    transform_and_plot as perf_gains_heatmap,
)
from prep_analysis.analysis.plots.ranking_level_bar_h import (
    transform_and_plot as ranking_level_bar_h,
)
from prep_analysis.analysis.plots.ranking_overall_sankey import (
    transform_and_plot as ranking_overall_sankey,
)
from prep_analysis.experiments.parse_results import parse_results_multiple
from prep_analysis.helpers import make_dir_if_not_exists


def analyze(df, output_dir=None):
    if output_dir:
        make_dir_if_not_exists(output_dir)

    df = df_filter_perf(df)

    # uncomment below to exclude Catboost and/or LightGBM
    # df = df.filter(pl.col('model') != 'g_CatBoost')
    # df = df.filter(pl.col('model') != 'f_LightGBM')

    perf_bar(df, output_dir=output_dir)
    perf_gains_heatmap(df, output_dir=output_dir)
    ranking_overall_sankey(df, output_dir=output_dir)
    ranking_level_bar_h(df, output_dir=output_dir)
    metric_heatmap(df, output_dir=output_dir)


if __name__ == '__main__':

    OUTPUT_DIR = '../../../Data/prep-analysis/plots/'

    RESULT_DIR = '../../../Data/prep-analysis/'
    F_PATHS = [
        f'{RESULT_DIR}results_cpu.txt',
        f'{RESULT_DIR}results_catboost.txt',
        f'{RESULT_DIR}results_gpu.txt',
    ]

    df = parse_results_multiple(F_PATHS)

    analyze_numbers(df)
    analyze(df, OUTPUT_DIR)
