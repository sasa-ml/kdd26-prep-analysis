from prep_analysis.experiments.experiment import Experiment
from prep_analysis.run_scripts.config_output_dir import OUTPUT_DIR

AG_PATH = f'{OUTPUT_DIR}AGModels/AutoGluonModels_gpu'

F_PATH = f'{OUTPUT_DIR}experiment/results_gpu.txt'

PIPELINES = [
    '1_No_prep',
    '2_Basic',
    '3_Tree',
    '4_Tree_imb',
    '5_NN',
    '6_NN_imb',
]

MODELS = [
    'FASTAI',
    'TABM',
    'TABDPT',
    'REALTABPFN-V2.5',
]

Experiment(
    output_filepath=F_PATH,
    output_append=False,
    subsample=False,
    lite=False,
    task_indices=None,
    pipeline_names=PIPELINES,
    model_names=MODELS,
    ag_models_path=AG_PATH,
)
