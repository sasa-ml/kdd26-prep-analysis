from prep_analysis.analysis.analysis import analyze
from prep_analysis.analysis.numbers_analysis import analyze_numbers
from prep_analysis.experiments.parse_results import parse_results_multiple
from prep_analysis.run_scripts.config_output_dir import OUTPUT_DIR

if __name__ == '__main__':

    PLOT_DIR = f'{OUTPUT_DIR}original/plots/'

    F_PATHS = [
        f'{OUTPUT_DIR}original/results_cpu_original.txt',
        f'{OUTPUT_DIR}original/results_catboost_original.txt',
        f'{OUTPUT_DIR}original/results_gpu_original.txt',
    ]

    df = parse_results_multiple(F_PATHS)

    analyze_numbers(df)
    analyze(df, PLOT_DIR)
    print(f'Plot directory path: {PLOT_DIR}')
