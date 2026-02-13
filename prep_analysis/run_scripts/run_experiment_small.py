from prep_analysis.experiments.experiment import Experiment
from prep_analysis.run_scripts.config_output_dir import OUTPUT_DIR

AG_PATH = f'{OUTPUT_DIR}AGModels/AutoGluonModels_small'

F_PATH = f'{OUTPUT_DIR}small/results_small.txt'

PIPELINES = [
    '1_No_prep',
    '2_Basic',
    '3_Tree',
    '5_NN',
]

MODELS = [
    'LR',
    'KNN',
    'RF',
    'XT',
    'XGB',
]

Experiment(
    output_filepath=F_PATH,
    output_append=False,
    subsample=False,
    lite=True,
    task_indices=list(range(46, 51)),
    pipeline_names=PIPELINES,
    model_names=MODELS,
    ag_models_path=AG_PATH,
)
