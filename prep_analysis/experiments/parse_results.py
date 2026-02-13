import polars as pl

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)


def parse_results(file_path):
    results = {
        'i_dataset': [],
        'n_rows': [],
        'n_cols': [],
        'task': [],
        'repeat': [],
        'fold': [],
        'metric': [],
        'pipeline': [],
        'train_prep_time': [],
        'fit_models_time': [],
        'test_prep_time': [],
        'eval_models_time': [],
        'model': [],
        'score': [],
        'pred_time': [],
        'fit_time': [],
        'extra_metric': [],
        'extra_score': [],
        'accuracy_score': [],
        'balanced_accuracy_score': [],
    }

    def update_results(
        i_dataset,
        n_rows,
        n_cols,
        task,
        repeat,
        fold,
        metric,
        pipeline,
        train_prep_time,
        fit_models_time,
        test_prep_time,
        eval_models_time,
        model,
        score,
        pred_time,
        fit_time,
        extra_metric,
        extra_score,
        accuracy_score,
        balanced_accuracy_score,
    ):
        results['i_dataset'].append(i_dataset)
        results['n_rows'].append(n_rows)
        results['n_cols'].append(n_cols)
        results['task'].append(task)
        results['repeat'].append(repeat)
        results['fold'].append(fold)
        results['metric'].append(metric)
        results['pipeline'].append(pipeline)
        results['train_prep_time'].append(train_prep_time)
        results['fit_models_time'].append(fit_models_time)
        results['test_prep_time'].append(test_prep_time)
        results['eval_models_time'].append(eval_models_time)
        results['model'].append(model)
        results['score'].append(score)
        results['pred_time'].append(pred_time)
        results['fit_time'].append(fit_time)
        results['extra_metric'].append(extra_metric)
        results['extra_score'].append(extra_score)
        results['accuracy_score'].append(accuracy_score)
        results['balanced_accuracy_score'].append(balanced_accuracy_score)

    # parsing
    with open(file_path, 'r') as f:
        lines = f.readlines()
        in_leaderboard = False

        for i, line in enumerate(lines):
            # print(i)

            if line.startswith('>> task: '):
                split = line.split()
                i_dataset = int(split[-1])
                continue

            if line.startswith('>> task type: '):
                split = line.split()
                task = split[3]
                continue

            if line.startswith('>> repeat: '):
                split = line.split()
                repeat = int(split[-1])
                continue

            if line.startswith('>> fold: '):
                split = line.split()
                fold = int(split[-1])
                continue

            if line.startswith('>> pipeline: '):
                split = line.split()
                pipeline = split[2]
                continue

            if line.startswith('>> train prep time: '):
                split = line.split()
                train_prep_time = float(split[-2])
                continue
            if line.startswith('>> fit models time: '):
                split = line.split()
                fit_models_time = float(split[-2])
                continue
            if line.startswith('>> test prep time: '):
                split = line.split()
                test_prep_time = float(split[-2])
                continue
            if line.startswith('>> eval models time: '):
                split = line.split()
                eval_models_time = float(split[-2])
                continue

            if line.startswith('>> leaderboard'):
                in_leaderboard = True
                continue

            if line.startswith('│ shape_n_rows'):
                split = line.split()
                n_rows = int(split[3])
                continue

            if line.startswith('│ shape_n_columns'):
                split = line.split()
                n_cols = int(split[3])
                continue

            if in_leaderboard:
                split = line.split()
                if split[0] == 'model':
                    measuring_accuracy = False
                    if split[-1] != 'fit_time':
                        extra_metric = split[-1]
                        if (
                            split[-3] == 'accuracy'
                            and split[-2] == 'balanced_accuracy'
                        ):
                            measuring_accuracy = True
                    else:
                        extra_metric = None
                    continue
                if split[0].isdigit():
                    model = split[1]
                    score = float(split[2])
                    metric = split[3]
                    pred_time = float(split[4])
                    fit_time = float(split[5])
                    extra_score = None
                    accuracy_score = None
                    balanced_accuracy_score = None
                    if extra_metric:
                        extra_score = float(split[-1])
                    if measuring_accuracy:
                        accuracy_score = float(split[-3])
                        balanced_accuracy_score = float(split[-2])
                    update_results(
                        i_dataset,
                        n_rows,
                        n_cols,
                        task,
                        repeat,
                        fold,
                        metric,
                        pipeline,
                        train_prep_time,
                        fit_models_time,
                        test_prep_time,
                        eval_models_time,
                        model,
                        score,
                        pred_time,
                        fit_time,
                        extra_metric,
                        extra_score,
                        accuracy_score,
                        balanced_accuracy_score,
                    )
                else:
                    in_leaderboard = False
                continue

    df = pl.DataFrame(results)

    # normalizing roc-auc(-ovr) / clipping
    df = df.with_columns(
        pl.when(pl.col('extra_metric').is_in(['roc_auc', 'roc_auc_ovr']))
        .then(2 * pl.col('extra_score') - 1)
        .otherwise(pl.col('extra_score'))
        .alias('extra_score')
    )
    df = df.with_columns(pl.col('extra_score').clip(lower_bound=0))

    # aggregation
    df = df.group_by(
        [
            'i_dataset',
            'n_rows',
            'n_cols',
            'task',
            'repeat',
            'metric',
            'pipeline',
            'model',
            'extra_metric',
        ],
        maintain_order=True,
    ).agg(
        [
            pl.col('score').mean(),
            pl.col('extra_score').mean(),
            pl.col('accuracy_score').mean(),
            pl.col('balanced_accuracy_score').mean(),
            pl.col('train_prep_time').mean(),
            pl.col('fit_models_time').mean(),
            pl.col('test_prep_time').mean(),
            pl.col('eval_models_time').mean(),
            pl.col('pred_time').mean(),
            pl.col('fit_time').mean(),
        ]
    )
    df = df.group_by(
        [
            'i_dataset',
            'n_rows',
            'n_cols',
            'task',
            'metric',
            'pipeline',
            'model',
            'extra_metric',
        ],
        maintain_order=True,
    ).agg(
        [
            pl.col('score').mean(),
            pl.col('score').std().alias('score_std'),
            pl.col('extra_score').mean(),
            pl.col('extra_score').std().alias('extra_score_std'),
            pl.col('accuracy_score').mean(),
            pl.col('accuracy_score').std().alias('accuracy_score_std'),
            pl.col('balanced_accuracy_score').mean(),
            pl.col('balanced_accuracy_score')
            .std()
            .alias('balanced_accuracy_score_std'),
            pl.col('train_prep_time').mean(),
            pl.col('fit_models_time').mean(),
            pl.col('test_prep_time').mean(),
            pl.col('eval_models_time').mean(),
            pl.col('pred_time').mean(),
            pl.col('fit_time').mean(),
        ]
    )

    df = df.with_columns(
        pl.col('model').replace(
            {
                'LinearModel': 'a_LinearModel',
                'KNeighbors': 'b_KNeighbors',
                'RandomForest': 'c_RandomForest',
                'ExtraTrees': 'd_ExtraTrees',
                'XGBoost': 'e_XGBoost',
                'LightGBM': 'f_LightGBM',
                'CatBoost': 'g_CatBoost',
                'NeuralNetFastAI': 'h_NeuralNetFastAI',
                'TabM': 'i_TabM',
                'TabDPT': 'j_TabDPT',
                'RealTabPFN-v2.5': 'k_RealTabPFN-v2.5',
            }
        )
    )

    df = df.sort(
        by=[
            'i_dataset',
            'pipeline',
            'model',
        ]
    )

    return df


def parse_results_multiple(file_paths):
    df = parse_results(file_paths[0])
    for fp in file_paths[1:]:
        df2 = parse_results(fp)
        df = pl.concat([df, df2])

    df = df.sort(
        by=[
            'i_dataset',
            'pipeline',
            'model',
        ]
    )

    return df


if __name__ == '__main__':

    RESULT_DIR = '../../../Data/prep-analysis/'
    F_PATHS = [
        f'{RESULT_DIR}results_cpu.txt',
        f'{RESULT_DIR}results_catboost.txt',
        f'{RESULT_DIR}results_gpu.txt',
    ]

    df = parse_results_multiple(F_PATHS)

    print(df)
