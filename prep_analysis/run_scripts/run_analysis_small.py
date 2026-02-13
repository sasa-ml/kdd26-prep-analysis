from prep_analysis.analysis.analysis import analyze
from prep_analysis.experiments.parse_results import parse_results_multiple
from prep_analysis.run_scripts.config_output_dir import OUTPUT_DIR

if __name__ == '__main__':

    PLOT_DIR = f'{OUTPUT_DIR}small/plots/'

    F_PATHS = [
        f'{OUTPUT_DIR}small/results_small.txt',
    ]

    df = parse_results_multiple(F_PATHS)

    analyze(df, PLOT_DIR)
    print(f'Plot directory path: {PLOT_DIR}')
