import io
import logging
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from autogluon.core.utils.early_stopping import SimpleES
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import (
    TabularNeuralNetTorchModel,
    logger,
)
from autogluon.tabular.models.tabular_nn.torch.tabular_torch_dataset import (
    TabularTorchDataset,
)

# from sklearn.compose import ColumnTransformer


def disable_NN_Torch_prep():
    TabularNeuralNetTorchModel._process_test_data = _process_test_data
    TabularNeuralNetTorchModel._process_train_data = _process_train_data

    TabularNeuralNetTorchModel._original_train_net = (
        TabularNeuralNetTorchModel._train_net
    )
    TabularNeuralNetTorchModel._train_net = _train_net


def _process_test_data(self, df, labels=None):
    """Process train or test DataFrame into a form fit for
       neural network models.
    Args:
        df (pd.DataFrame): Data to be processed (X)
        labels (pd.Series): labels to be processed (y)
    Returns:
        Dataset object
    """

    # df = self.processor.transform(df)
    df = df.to_numpy()

    return TabularTorchDataset(
        df,
        self.feature_arraycol_map,
        self.feature_type_map,
        self.problem_type,
        labels,
    )


def _process_train_data(
    self,
    df,
    impute_strategy,
    max_category_levels,
    skew_threshold,
    embed_min_categories,
    use_ngram_features,
    labels,
):

    features = df.columns.tolist()

    # OrderedDict of
    # feature-name -> list of column-indices in df
    # corresponding to this feature
    self.feature_arraycol_map = {
        feature: [i] for i, feature in enumerate(features)
    }
    self.feature_arraycol_map = OrderedDict(
        [
            (key, self.feature_arraycol_map[key])
            for key in self.feature_arraycol_map
        ]
    )

    # OrderedDict of
    # feature-name -> feature_type string (options: 'vector', 'embed')
    self.feature_type_map = {feature: "vector" for feature in features}
    self.feature_type_map = OrderedDict(
        [(key, self.feature_type_map[key]) for key in self.feature_type_map]
    )

    # self.processor = ColumnTransformer(
    #    transformers=[], remainder='passthrough'
    # )
    # df = self.processor.fit_transform(df)
    df = df.to_numpy()

    return TabularTorchDataset(
        df,
        self.feature_arraycol_map,
        self.feature_type_map,
        self.problem_type,
        labels,
    )


def _train_net(
    self,
    train_dataset: TabularTorchDataset,
    loss_kwargs: dict,
    batch_size: int,
    num_epochs: int,
    epochs_wo_improve: int,
    val_dataset: TabularTorchDataset = None,
    test_dataset: TabularTorchDataset = None,
    time_limit: float = None,
    reporter=None,
    verbosity: int = 2,
):
    import torch

    start_time = time.time()
    logging.debug("initializing neural network...")
    self.model.init_params()
    logging.debug("initialized")
    train_dataloader = train_dataset.build_loader(
        batch_size, self.num_dataloading_workers, is_test=False
    )

    if (
        isinstance(loss_kwargs.get("loss_function", "auto"), str)
        and loss_kwargs.get("loss_function", "auto") == "auto"
    ):
        loss_kwargs["loss_function"] = self._get_default_loss_function()
    if epochs_wo_improve is not None:
        early_stopping_method = SimpleES(patience=epochs_wo_improve)
    else:
        early_stopping_method = self._get_early_stopping_strategy(
            num_rows_train=len(train_dataset)
        )

    ag_params = self._get_ag_params()
    generate_curves = ag_params.get("generate_curves", False)

    if generate_curves:
        scorers = ag_params.get("curve_metrics", [self.eval_metric])
        use_curve_metric_error = ag_params.get(
            "use_error_for_curve_metrics", False
        )
        metric_names = [scorer.name for scorer in scorers]

        train_curves = {metric.name: [] for metric in scorers}
        val_curves = {metric.name: [] for metric in scorers}
        test_curves = {metric.name: [] for metric in scorers}

        # make copy of train_dataset to avoid interfering with train_dataloader
        curve_train_dataset = deepcopy(train_dataset)
        y_train = curve_train_dataset.get_labels()
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.flatten()

        if test_dataset is not None:
            y_test = test_dataset.get_labels()
            if y_test.ndim == 2 and y_test.shape[1] == 1:
                y_test = y_test.flatten()
        else:
            y_test = None

    if val_dataset is not None:
        y_val = val_dataset.get_labels()
        if y_val.ndim == 2 and y_val.shape[1] == 1:
            y_val = y_val.flatten()
    else:
        y_val = None

    if verbosity <= 1:
        verbose_eval = False
    else:
        verbose_eval = True

    logger.log(15, "Neural network architecture:")
    logger.log(15, str(self.model))

    io_buffer = None
    if num_epochs == 0:
        # use dummy training loop that stops immediately
        # useful for using NN just for data preprocessing / debugging
        logger.log(
            20, "Not training Tabular Neural Network since num_updates == 0"
        )

        # for each batch
        for batch_idx, data_batch in enumerate(train_dataloader):
            if batch_idx > 0:
                break
            loss = self.model.compute_loss(data_batch, **loss_kwargs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return

    # start training loop:
    logger.log(
        15, f"Training tabular neural network for up to {num_epochs} epochs..."
    )
    total_updates = 0
    num_updates_per_epoch = max(round(len(train_dataset) / batch_size) + 1, 1)
    update_to_check_time = min(10, max(1, int(num_updates_per_epoch / 5)))
    do_update = True
    epoch = 0
    best_epoch = 0
    best_val_metric = -np.inf  # higher = better
    best_val_update = 0
    start_fit_time = time.time()
    if time_limit is not None:
        time_limit = time_limit - (start_fit_time - start_time)
        if time_limit <= 0:
            raise TimeLimitExceeded
    while do_update:
        time_start_epoch = time.time()
        time_cur = time_start_epoch
        total_train_loss = 0.0
        total_train_size = 0.0
        for batch_idx, data_batch in enumerate(train_dataloader):
            # forward
            loss = self.model.compute_loss(data_batch, **loss_kwargs)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite train loss detected "
                    f"(epoch={epoch}, batch={batch_idx}, loss={loss.item()})"
                )
            total_train_loss += loss.item()
            total_train_size += 1

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_updates += 1

            # time limit
            if time_limit is not None:
                time_cur_tmp = time.time()
                time_elapsed_batch = time_cur_tmp - time_cur
                time_cur = time_cur_tmp
                update_cur = batch_idx + 1
                if epoch == 0 and update_cur == update_to_check_time:
                    time_elapsed_epoch = time_cur - time_start_epoch

                    # v1 estimate is sensitive to fixed cost overhead at the
                    # start of training, such as torch initialization.
                    # v2 fixes this, but we keep both and take the min to
                    # avoid potential cases where v2 is inaccurate due to an
                    # overly slow batch.
                    estimated_time_v1 = (
                        time_elapsed_epoch / update_cur * num_updates_per_epoch
                    )  # Less accurate than v2, but never underestimates time
                    estimated_time_v2 = (
                        time_elapsed_epoch
                        + time_elapsed_batch
                        * (num_updates_per_epoch - update_cur)
                    )  # Less likely to overestimate time
                    estimated_time = min(estimated_time_v1, estimated_time_v2)
                    if estimated_time > time_limit:
                        logger.log(
                            30,
                            f"\tNot enough time to train first epoch. "
                            f"(Time Required: {round(estimated_time, 2)}s, "
                            + f"Time Left: {round(time_limit, 2)}s)",
                        )
                        raise TimeLimitExceeded
                time_elapsed = time_cur - start_fit_time
                if time_limit < time_elapsed:
                    if epoch == 0:
                        logger.log(
                            30,
                            "\tNot enough time to train first epoch. Stopped "
                            + f"on Update {total_updates} (Epoch {epoch}))",
                        )
                        raise TimeLimitExceeded
                    logger.log(
                        15,
                        "\tRan out of time, stopping training early. (Stopped "
                        + f"on Update {total_updates} (Epoch {epoch}))",
                    )
                    do_update = False
                    break

        if not do_update:
            break

        epoch += 1

        # learning curve generation
        if generate_curves:
            stop = self._generate_curves(
                train_curves=train_curves,
                val_curves=val_curves,
                test_curves=test_curves,
                scorers=scorers,
                best_epoch=best_epoch,
                use_curve_metric_error=use_curve_metric_error,
                train_dataset=curve_train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
            )

            if stop:
                break

        # validation
        if val_dataset is not None:
            is_best = False
            # compute validation score
            val_metric = self.score(
                X=val_dataset,
                y=y_val,
                metric=self.stopping_metric,
                _reset_threads=False,
            )
            if not self._assert_valid_metric(
                metric=val_metric, best_epoch=best_epoch
            ):
                break

            # update best validation
            if (val_metric >= best_val_metric) or best_epoch == 0:
                if val_metric > best_val_metric:
                    is_best = True
                best_val_metric = val_metric
                io_buffer = io.BytesIO()
                torch.save(self.model.state_dict(), io_buffer)
                best_epoch = epoch
                best_val_update = total_updates
            early_stop = early_stopping_method.update(
                cur_round=epoch - 1, is_best=is_best
            )
            if verbose_eval:
                logger.log(
                    15,
                    f"Epoch {epoch} (Update {total_updates}).\t"
                    "Train loss: "
                    + f"{round(total_train_loss / total_train_size, 4)}, "
                    f"Val {self.stopping_metric.name}: "
                    + f"{round(val_metric, 4)}, "
                    f"Best Epoch: {best_epoch}",
                )

            if reporter is not None:
                reporter(
                    epoch=total_updates,
                    # Higher val_metric = better
                    validation_performance=val_metric,
                    train_loss=total_train_loss / total_train_size,
                    eval_metric=self.eval_metric.name,
                    greater_is_better=self.eval_metric.greater_is_better,
                )

            # no improvement
            if early_stop:
                break

        if epoch >= num_epochs:
            break

        if time_limit is not None:
            time_elapsed = time.time() - start_fit_time
            time_epoch_average = time_elapsed / max(
                epoch, 1
            )  # avoid divide by 0
            time_left = time_limit - time_elapsed
            if time_left < time_epoch_average:
                logger.log(
                    20,
                    "\tRan out of time, stopping training early. "
                    + f"(Stopping on epoch {epoch})",
                )
                break

    if epoch == 0:
        raise AssertionError("0 epochs trained!")

    if generate_curves:
        curves = {"train": train_curves}
        if val_dataset is not None:
            curves["val"] = val_curves
        if test_dataset is not None:
            curves["test"] = test_curves
        self.save_learning_curves(metrics=metric_names, curves=curves)

    # revert back to best model
    if val_dataset is not None:
        logger.log(
            15,
            f"Best model found on Epoch {best_epoch} "
            + f"(Update {best_val_update}). Val "
            + f"{self.stopping_metric.name}: {best_val_metric}",
        )
        if io_buffer is not None:
            io_buffer.seek(0)
            self.model.load_state_dict(
                torch.load(io_buffer, weights_only=True)
            )
    else:
        logger.log(
            15,
            f"Best model found on Epoch {best_epoch} "
            + f"(Update {best_val_update}).",
        )
    self.params_trained["batch_size"] = batch_size
    self.params_trained["num_epochs"] = best_epoch
