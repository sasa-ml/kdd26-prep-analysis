import polars as pl

from prep_analysis.analysis.helpers_df import (
    df_add_level_diffs,
    df_add_levels,
    df_filter_perf,
    df_get_wins_by_pipeline,
)
from prep_analysis.experiments.parse_results import parse_results_multiple


def analyze_numbers(df):
    df = df_filter_perf(df)
    df = df_add_levels(df)
    df = df_add_level_diffs(df)
    df_wins_pipe = df_get_wins_by_pipeline(df)

    no_prep_nan_pct = (
        df.filter(
            (pl.col('pipeline') == '1_No_prep')
            & pl.col('extra_score').is_nan()
        ).height
        / df.filter(pl.col('pipeline') == '1_No_prep').height
        * 100
    )
    print(f'No prep nan dataset %: {round(no_prep_nan_pct)}')

    best_inc_pct = round(
        df.filter(
            (pl.col('pipeline') == '3_diff_best_minimal')
            & (pl.col('extra_score') > 0)
        ).height
        / df.filter(pl.col('pipeline') == '3_diff_best_minimal').height
        * 100
    )
    print(f'Best > Minimal dataset %: {round(best_inc_pct)}')

    n_datasets = df.group_by('i_dataset').first().height

    tree_win_pct = (
        df_wins_pipe.filter(
            (pl.col('pipeline') == '1_No_prep')
            & pl.col('model').is_in(
                [
                    'c_RandomForest',
                    'd_ExtraTrees',
                    'e_XGBoost',
                    'f_LightGBM',
                    'g_CatBoost',
                ]
            )
        )['win_count'].sum()
        / n_datasets
        * 100
    )
    print(f'No prep tree win %: {round(tree_win_pct)}')

    catboost_win_pct = (
        df_wins_pipe.filter(
            (pl.col('pipeline') == '1_No_prep')
            & (pl.col('model') == 'g_CatBoost')
        )['win_count'].item()
        / n_datasets
        * 100
    )
    print(f'No prep CatBoost win %: {round(catboost_win_pct)}')

    foundation_win_pct = (
        df_wins_pipe.filter(
            (pl.col('pipeline') == '3_Best')
            & pl.col('model').is_in(
                [
                    'j_TabDPT',
                    'k_RealTabPFN-v2.5',
                ]
            )
        )['win_count'].sum()
        / n_datasets
        * 100
    )
    print(f'Best prep foundation win %: {round(foundation_win_pct)}')

    tabpfn_win_pct = (
        df_wins_pipe.filter(
            (pl.col('pipeline') == '3_Best')
            & (pl.col('model') == 'k_RealTabPFN-v2.5')
        )['win_count'].item()
        / n_datasets
        * 100
    )
    print(f'Best prep RealTabPFN-v2.5 win %: {round(tabpfn_win_pct)}')

    nn_win_pct = (
        df_wins_pipe.filter(
            (pl.col('pipeline') == '3_Best')
            & pl.col('model').is_in(
                [
                    'h_NeuralNetFastAI',
                    'i_TabM',
                ]
            )
        )['win_count'].sum()
        / n_datasets
        * 100
    )
    print(f'Best prep NN win %: {round(nn_win_pct)}')


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
