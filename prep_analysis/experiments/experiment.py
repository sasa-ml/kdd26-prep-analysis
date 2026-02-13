import time
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from auto_prepper.core.dataset import Dataset  # type: ignore
from auto_prepper.core.profiling import Profile  # type: ignore
from auto_prepper.io.output import to_pandas  # type: ignore
from auto_prepper.optimizers.preset_adaptable import (  # type: ignore
    AdaptablePresetPipeline,
)
from autogluon.common import FeatureMetadata
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from tabarena.benchmark.task.openml import OpenMLTaskWrapper

from prep_analysis.ag_patches.patch_all import disable_AG_model_prep
from prep_analysis.helpers import clear_dir, make_dir_if_not_exists, remove_dir

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

TABARENA_SUITE = 457

NUM_GPUS = ResourceManager.get_gpu_count()

AG_HP_CONFIG_ALL = {
    'LR': {},
    'KNN': [
        {
            'weights': 'distance',
        },
    ],
    'RF': [
        {
            'criterion': 'entropy',
            'ag_args': {
                'problem_types': ['binary', 'multiclass'],
            },
        },
        {
            'criterion': 'squared_error',
            'ag_args': {
                'problem_types': ['regression', 'quantile'],
            },
        },
    ],
    'XT': [
        {
            'criterion': 'entropy',
            'ag_args': {
                'problem_types': ['binary', 'multiclass'],
            },
        },
        {
            'criterion': 'squared_error',
            'ag_args': {
                'problem_types': ['regression', 'quantile'],
            },
        },
    ],
    'XGB': {},
    'GBM': {},
    'CAT': {},
    'FASTAI': {
        'ag.num_gpus': NUM_GPUS,
    },
    'NN_TORCH': {
        'ag.num_gpus': NUM_GPUS,
    },
    'REALMLP': {
        'ag.num_gpus': NUM_GPUS,
    },
    'TABM': {
        'ag.num_gpus': NUM_GPUS,
        'ag.max_rows': None,
        'ag.max_features': None,
        'ag.max_classes': None,
    },
    'TABICL': {
        'ag.num_gpus': NUM_GPUS,
        'ag.max_rows': None,
        'ag.max_features': None,
        'ag.max_classes': None,
    },
    'TABDPT': {
        'ag.num_gpus': NUM_GPUS,
        'ag.max_rows': None,
        'ag.max_features': None,
        'ag.max_classes': None,
    },
    'REALTABPFN-V2.5': {
        'ag.num_gpus': NUM_GPUS,
        'ag.max_rows': None,
        'ag.max_features': None,
        'ag.max_classes': None,
    },
}


class Experiment:
    def __init__(
        self,
        output_filepath=None,
        output_append=False,
        subsample=False,
        lite=False,
        task_indices=None,
        pipeline_names=[],
        model_names=[],
        ag_models_path=None,
    ):
        self.output_filepath = (
            str(Path(output_filepath).expanduser())
            if output_filepath
            else None
        )
        self.output_append = output_append
        self.subsample = subsample
        self.lite = lite
        self.task_indices = task_indices
        self.pipeline_names = pipeline_names
        self.model_hp_configs = {m: AG_HP_CONFIG_ALL[m] for m in model_names}
        self.ag_models_path = (
            str(Path(ag_models_path).expanduser()) if ag_models_path else None
        )

        self.suite = openml.study.get_suite(TABARENA_SUITE)
        if not self.task_indices:
            self.task_indices = list(range(len(self.suite.tasks)))
        self.logger = None
        self._current_task = None
        self._current_pipeline_name = None

        self._launch()

    def _launch(self):
        make_dir_if_not_exists(self.ag_models_path)
        make_dir_if_not_exists(self.output_filepath)

        self.logger = ExperimentLogger(
            output_filepath=self.output_filepath, append=self.output_append
        )
        t = time.time()

        try:
            self.logger.start()

            self._run()

        finally:
            t = time.time() - t
            self.logger.log_time('total experiment', t)
            self.logger.stop()
            remove_dir(dir_path=self.ag_models_path)
            remove_dir(dir_path='lightning_logs')
            print(f'Output file path: {self.output_filepath}')

    def _run(self):
        disable_AG_model_prep()
        self.logger.log('Model prep disabled.')

        self.logger.log(self.suite, no_indent=True, top_line='=')

        for task_index in self.task_indices:
            self._current_task = self._get_openml_task(task_index=task_index)
            self.logger.log(f'task: {task_index}', top_line='=')
            self.logger.log(
                self._current_task.task,
                no_indent=True,
                top_line='=',
            )
            self.logger.log(
                f'task type: {self._current_task.problem_type}',
                top_line='-',
            )

            n_rows = self._preview_task_data()
            if self.lite:
                n_repeats = 1
                n_folds = 1
            else:
                # n_repeats, n_folds, _ = (
                #    self._current_task.get_split_dimensions()
                # )
                # if n_rows >= 2500:
                #    n_repeats = 3
                n_repeats = 8  # 5
                n_folds = 3
                if n_rows >= 2000:
                    n_repeats = 4  # 3
                if n_rows >= 10000:
                    n_repeats = 2  # 1
                if n_rows >= 50000:
                    n_repeats = 1
                    # n_folds = 1

            for repeat in range(n_repeats):
                for fold in range(n_folds):
                    self.logger.log(f'task: {task_index}', top_line='-')
                    self.logger.log(f'repeat: {repeat}')
                    self.logger.log(f'fold: {fold}')

                    for pipeline_name in self.pipeline_names:
                        self._current_pipeline_name = pipeline_name
                        self.logger.log(
                            f'pipeline: {self._current_pipeline_name}',
                            top_line='-',
                        )

                        df_train, df_test = self._get_train_test_data(
                            repeat=repeat, fold=fold
                        )
                        df_train, pipeline = self._prep_data_train(
                            df_train=df_train
                        )
                        predictor = self._fit_models(df_train=df_train)
                        df_test = self._prep_data_test(
                            df_test=df_test,
                            pipeline=pipeline,
                        )
                        self._eval_models(
                            df_test=df_test,
                            predictor=predictor,
                        )

                        self.logger.log(f'pipeline {pipeline}', no_indent=True)

                        self.logger.flush()
                        clear_dir(dir_path=self.ag_models_path)
                        clear_dir(dir_path='lightning_logs')

    def _get_openml_task(self, task_index):
        task_id = self.suite.tasks[task_index]
        task = OpenMLTaskWrapper.from_task_id(task_id=task_id)
        return task

    def _preview_task_data(self):
        df_train, df_test = self._current_task.get_train_test_split_combined(
            repeat=0, fold=0
        )
        n_rows = len(df_train) + len(df_test)
        ds_train = Dataset(
            df_train,
            dataset_type='train',
            target_column=self._current_task.label,
        )

        self.logger.log(f'ds train\n{ds_train}')
        profile = Profile(ds_train)
        self.logger.log(f'profile summary\n{profile.summary_table()}')

        return n_rows  # TODO or profile?

    def _stratified_sample(
        self,
        df,
        target_column,
        sample_size,
        samples_per_class,
    ):
        class_samples = df.groupby(target_column, group_keys=False).apply(
            lambda c: c.sample(n=min(samples_per_class, len(c)))
        )
        remaining_size = max(sample_size - len(class_samples), 0)
        if remaining_size > 0:
            excluded = df.drop(class_samples.index)
            random_samples = excluded.sample(
                n=min(remaining_size, len(excluded))
            )
            df_stratified = pd.concat([class_samples, random_samples])
        else:
            df_stratified = class_samples
        return df_stratified

    def _subsample_data(
        self,
        df_train,
        df_test,
        train_size=300,
        test_size=100,
    ):
        target_column = self._current_task.label
        try:
            _, df_train = train_test_split(
                df_train,
                test_size=train_size,
                stratify=df_train[target_column],
                random_state=42,
            )
            _, df_test = train_test_split(
                df_test,
                test_size=test_size,
                stratify=df_test[target_column],
                random_state=42,
            )
        except Exception as e:
            print(e)
            print('>> warning: sklearn stratified sampling failed, using ours')
            try:
                df_train = self._stratified_sample(
                    df_train,
                    target_column=target_column,
                    sample_size=train_size,
                    samples_per_class=2,
                )
                df_test = self._stratified_sample(
                    df_test,
                    target_column=target_column,
                    sample_size=test_size,
                    samples_per_class=2,
                )
            except Exception as e:
                print(e)
                print(
                    '>> warning: stratified sampling failed, '
                    + 'using random sampling'
                )
                if len(df_train) > train_size:
                    df_train = df_train.sample(train_size)
                if len(df_test) > test_size:
                    df_test = df_test.sample(test_size)
        return df_train, df_test

    def _get_train_test_data(self, repeat=0, fold=0):
        df_train, df_test = self._current_task.get_train_test_split_combined(
            repeat=repeat,
            fold=fold,
        )

        if self.subsample:
            df_train, df_test = self._subsample_data(
                df_train=df_train,
                df_test=df_test,
                train_size=400,
                test_size=100,
            )

        return df_train, df_test

    def _is_current_pipeline_custom(self):
        return self._current_pipeline_name not in [
            '1_No_prep',
        ]

    def _get_feature_metadata(self, df):
        df = df.convert_dtypes()
        dtype_dict = df.dtypes.apply(lambda x: x.name).to_dict()
        feature_metadata = FeatureMetadata(type_map_raw=dtype_dict)
        return feature_metadata

    def _prep_data_train(self, df_train):
        if not self._is_current_pipeline_custom():
            self.logger.log_time('train prep', 0.0)
            return df_train, None

        t = time.time()

        ds_train = Dataset(
            df_train,
            dataset_type='train',
            target_column=self._current_task.label,
        )
        pipeline = AdaptablePresetPipeline(
            ds=ds_train,
            preset_type=self._current_pipeline_name,
        ).optimize_pipeline()
        ds_train = pipeline.transform(ds_train)
        df_train = to_pandas(ds_train.df)

        t = time.time() - t
        self.logger.log_time('train prep', t)

        return df_train, pipeline

    def _prep_data_test(self, df_test, pipeline):
        if not self._is_current_pipeline_custom():
            self.logger.log_time('test prep', 0.0)
            return df_test

        t = time.time()

        ds_test = Dataset(
            df_test,
            dataset_type='test',
            target_column=self._current_task.label,
        )
        ds_test = pipeline.transform(ds_test)
        df_test = to_pandas(ds_test.df)

        t = time.time() - t
        self.logger.log_time('test prep', t)

        return df_test

    def _fit_models(self, df_train):
        t = time.time()

        predictor = TabularPredictor(
            label=self._current_task.label,
            problem_type=self._current_task.problem_type,
            eval_metric=self._current_task.eval_metric,
            path=self.ag_models_path,
            verbosity=0,
        )

        predictor = predictor.fit(
            train_data=df_train,
            # presets='best',
            feature_generator=None,
            hyperparameters=self.model_hp_configs,
            fit_weighted_ensemble=False,
            raise_on_no_models_fitted=False,
            feature_metadata=self._get_feature_metadata(df_train),
            # time_limit=120,
            # infer_limit=0.05,
        )

        t = time.time() - t
        self.logger.log_time('fit models', t)

        # predictor.save_space()

        return predictor

    def _eval_models(self, df_test, predictor):
        t = time.time()

        if self._current_task.problem_type == 'regression':
            extra_metrics = [
                'r2',
                # 'mean_absolute_percentage_error',
                # 'symmetric_mean_absolute_percentage_error',
            ]
        elif self._current_task.problem_type == 'multiclass':
            extra_metrics = [
                'accuracy',
                'balanced_accuracy',
                # 'f1_macro',
                'roc_auc_ovr',  # macro or weighted?
            ]
        else:  # self._current_task.problem_type == 'binary'
            extra_metrics = [
                'accuracy',
                'balanced_accuracy',
                # 'f1',
                'roc_auc',
            ]

        columns_to_track = [
            'model',
            'score_test',
            'eval_metric',
            'pred_time_test',
            'fit_time',
        ]
        columns_to_track.extend(extra_metrics)

        leaderboard = predictor.leaderboard(
            data=df_test,
            extra_metrics=extra_metrics,
        )

        if not leaderboard.empty:
            leaderboard = leaderboard[columns_to_track]
        leaderboard = self._add_failed_to_leaderboard(
            predictor=predictor,
            leaderboard=leaderboard,
            extra_metrics=extra_metrics,
        )

        t = time.time() - t
        self.logger.log_time('eval models', t)
        self.logger.log(f'leaderboard\n{leaderboard}')

        return leaderboard

    def _add_failed_to_leaderboard(
        self,
        predictor,
        leaderboard,
        extra_metrics,
    ):
        model_failures = predictor.model_failures()
        if not model_failures.empty:
            failed_models = list(model_failures['model'])
            nan_list = [np.nan for i in range(len(failed_models))]
            failed_dict = {
                'model': failed_models,
                'score_test': nan_list,
                'eval_metric': [
                    self._current_task.eval_metric
                    for i in range(len(failed_models))
                ],
                'pred_time_test': nan_list,
                'fit_time': nan_list,
            }
            for extra_metric in extra_metrics:
                failed_dict[extra_metric] = nan_list
            failed_leaderboard = pd.DataFrame(failed_dict)
            if leaderboard.empty:
                leaderboard = failed_leaderboard
            else:
                leaderboard = pd.concat(
                    [leaderboard, failed_leaderboard],
                    ignore_index=True,
                )

        return leaderboard


class ExperimentLogger:
    def __init__(self, output_filepath=None, append=False):
        self.output_filepath = output_filepath
        self.f_results = None
        self.mode = 'a' if append else 'w'

    def start(self):
        if self.output_filepath:
            self.f_results = open(self.output_filepath, self.mode)

    def stop(self):
        if self.f_results:
            self.f_results.close()

    def flush(self):
        if self.f_results:
            self.f_results.flush()

    def log(self, s, no_indent=False, top_line=None, bottom_line=None):
        indent = '' if no_indent else '>> '
        if top_line:
            self.log_line(top_line)
        print(f'{indent}{s}')
        if self.f_results:
            self.f_results.write(f'{indent}{s}\n')
        if bottom_line:
            self.log_line(bottom_line)

    def log_time(self, s, t):
        self.log(f'{s} time: {t} s')

    def log_line(self, c):
        s = 36 * c
        print(s)
        if self.f_results:
            self.f_results.write(f'{s}\n')


if __name__ == '__main__':

    AG_PATH = '../../../Data/prep-analysis/AGModels/AutoGluonModels_gpu'
    # AG_PATH = '../../../Data/prep-analysis/AGModels/AutoGluonModels_test'

    F_PATH = '../../../Data/prep-analysis/results_gpu.txt'
    # F_PATH = '../../../Data/prep-analysis/test.txt'

    PIPELINES = [
        '1_No_prep',
        '2_Basic',
        '3_Tree',
        '4_Tree_imb',
        '5_NN',
        '6_NN_imb',
    ]

    MODELS = [
        'LR',
        'KNN',
        'RF',
        'XT',
        'XGB',
        'GBM',
        'CAT',
        'FASTAI',
        'TABM',
        'TABDPT',
        'REALTABPFN-V2.5',
    ]
    # 'NN_TORCH',
    # 'REALMLP',

    Experiment(
        output_filepath=F_PATH,
        output_append=True,
        subsample=False,
        lite=False,
        task_indices=None,
        pipeline_names=PIPELINES,
        model_names=MODELS,
        ag_models_path=AG_PATH,
    )
