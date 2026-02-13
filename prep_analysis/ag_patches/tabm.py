import math
import random
import time
from typing import Any, Literal

import autogluon.tabular.models.tabm._tabm_internal
import numpy as np
import pandas as pd
import scipy
import torch
from autogluon.core.metrics import compute_metric
from autogluon.tabular.models.tabm import rtdl_num_embeddings, tabm_reference
from autogluon.tabular.models.tabm._tabm_internal import (
    RTDLQuantileTransformer,
    TabMImplementation,
    TabMOrdinalEncoder,
    get_tabm_auto_batch_size,
    logger,
)
from autogluon.tabular.models.tabm.tabm_reference import make_parameter_groups
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TaskType = Literal["regression", "binclass", "multiclass"]


def disable_TabM_prep():
    autogluon.tabular.models.tabm._tabm_internal.SimpleImputer = NoOpImputer
    TabMImplementation.fit = fit_model
    RTDLQuantileTransformer.fit = fit
    RTDLQuantileTransformer.transform = transform
    TabMOrdinalEncoder.fit = fit
    TabMOrdinalEncoder.transform = transform_encoder
    TabMOrdinalEncoder.get_cardinalities = get_cardinalities


class NoOpImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
        add_indicator=False,
    ):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy
        self.add_indicator = add_indicator
        self.statistics_ = None
        self.indicator_ = None

    def fit(self, X, y=None):
        self.statistics_ = np.zeros(X.shape[1]) if hasattr(X, "shape") else []
        if self.add_indicator:
            self.indicator_ = (
                np.zeros(X.shape[1], dtype=bool) if hasattr(X, "shape") else []
            )
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def fit(self, X, y=None):
    return self


def transform(self, X):
    return X


def transform_encoder(self, X):
    return X.to_numpy()


def get_cardinalities(self):
    return []


def fit_model(
    self,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_col_names: list[Any],
    time_to_fit_in_seconds: float | None = None,
):
    start_time = time.time()

    if X_val is None or len(X_val) == 0:
        raise ValueError(
            "Training without validation set is currently not implemented"
        )
    seed: int | None = self.config.get("random_state", None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if "n_threads" in self.config:
        torch.set_num_threads(self.config["n_threads"])

    # -- Meta parameters
    problem_type = self.config["problem_type"]
    task_type: TaskType = (
        "binclass" if problem_type == "binary" else problem_type
    )
    n_train = len(X_train)
    n_classes = None
    device = self.config["device"]
    device = torch.device(device)
    self.task_type_ = task_type
    self.device_ = device
    self.cat_col_names_ = cat_col_names

    # -- Hyperparameters
    arch_type = self.config.get("arch_type", "tabm-mini")
    num_emb_type = self.config.get("num_emb_type", "pwl")
    n_epochs = self.config.get("n_epochs", 1_000_000_000)
    patience = self.config.get("patience", 16)
    batch_size = self.config.get("batch_size", "auto")
    compile_model = self.config.get("compile_model", False)
    lr = self.config.get("lr", 2e-3)
    d_embedding = self.config.get("d_embedding", 16)
    d_block = self.config.get("d_block", 512)
    dropout = self.config.get("dropout", 0.1)
    tabm_k = self.config.get("tabm_k", 32)
    allow_amp = self.config.get("allow_amp", False)
    n_blocks = self.config.get("n_blocks", "auto")
    num_emb_n_bins = self.config.get("num_emb_n_bins", 48)
    eval_batch_size = self.config.get("eval_batch_size", 1024)
    share_training_batches = self.config.get("share_training_batches", False)
    weight_decay = self.config.get("weight_decay", 3e-4)
    # this is the search space default but not the example default
    # (which is 'none')
    gradient_clipping_norm = self.config.get("gradient_clipping_norm", 1.0)

    # -- Verify HPs
    num_emb_n_bins = min(num_emb_n_bins, n_train - 1)
    if n_train <= 2:
        # there is no valid number of bins for piecewise linear embeddings
        num_emb_type = "none"
    if batch_size == "auto":
        batch_size = get_tabm_auto_batch_size(n_train=n_train)

    # -- Preprocessing
    ds_parts = dict()
    # Unique ordinal encoder -> replaces nan and missing values with the
    # cardinality
    self.ord_enc_ = TabMOrdinalEncoder()
    self.ord_enc_.fit(X_train[self.cat_col_names_])
    # TODO: fix transformer to be able to work with empty input data like the
    # sklearn default
    self.num_prep_ = Pipeline(
        steps=[
            (
                "qt",
                RTDLQuantileTransformer(
                    random_state=self.config.get("random_state", None)
                ),
            ),
            ("imp", SimpleImputer(add_indicator=True)),
        ]
    )
    self.has_num_cols = bool(set(X_train.columns) - set(cat_col_names))
    for part, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        tensors = dict()

        tensors["x_cat"] = torch.as_tensor(
            self.ord_enc_.transform(X[cat_col_names]), dtype=torch.long
        )

        if self.has_num_cols:
            x_cont_np = X.drop(columns=cat_col_names).to_numpy(
                dtype=np.float32
            )
            if part == "train":
                self.num_prep_.fit(x_cont_np)
            tensors["x_cont"] = torch.as_tensor(
                self.num_prep_.transform(x_cont_np)
            )
        else:
            tensors["x_cont"] = torch.empty((len(X), 0), dtype=torch.float32)

        if task_type == "regression":
            tensors["y"] = torch.as_tensor(y.to_numpy(np.float32))
            if part == "train":
                n_classes = 0
        else:
            tensors["y"] = torch.as_tensor(
                y.to_numpy(np.int32), dtype=torch.long
            )
            if part == "train":
                n_classes = tensors["y"].max().item() + 1

        ds_parts[part] = tensors

    part_names = ["train", "val"]
    cat_cardinalities = self.ord_enc_.get_cardinalities()
    self.n_classes_ = n_classes

    # filter out numerical columns with only a single value
    #  -> AG also does this already but preprocessing might create constant
    # columns again
    x_cont_train = ds_parts["train"]["x_cont"]
    self.num_col_mask_ = ~torch.all(
        x_cont_train == x_cont_train[0:1, :], dim=0
    )
    for part in part_names:
        ds_parts[part]["x_cont"] = ds_parts[part]["x_cont"][
            :, self.num_col_mask_
        ]
        # tensor infos are not correct anymore, but might not be used either
    for part in part_names:
        for tens_name in ds_parts[part]:
            ds_parts[part][tens_name] = ds_parts[part][tens_name].to(device)

    # update
    n_cont_features = ds_parts["train"]["x_cont"].shape[1]

    Y_train = ds_parts["train"]["y"].clone()
    if task_type == "regression":
        self.y_mean_ = ds_parts["train"]["y"].mean().item()
        self.y_std_ = ds_parts["train"]["y"].std(correction=0).item()

        Y_train = (Y_train - self.y_mean_) / (self.y_std_ + 1e-30)

    # the | operator joins dicts (like update() but not in-place)
    data = {
        part: dict(x_cont=ds_parts[part]["x_cont"], y=ds_parts[part]["y"])
        | (
            dict(x_cat=ds_parts[part]["x_cat"])
            if ds_parts[part]["x_cat"].shape[1] > 0
            else dict()
        )
        for part in part_names
    }

    # adapted from
    # https://github.com/yandex-research/tabm/blob/main/example.ipynb

    # Automatic mixed precision (AMP)
    # torch.float16 is implemented for completeness,
    # but it was not tested in the project,
    # so torch.bfloat16 is used by default.
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available() else None
    )
    # Changing False to True will result in faster training on compatible
    # hardware.
    amp_enabled = allow_amp and amp_dtype is not None
    grad_scaler = (
        torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None
    )  # type: ignore

    # fmt: off
    logger.log(
        15, f"Device:        {device.type.upper()}"
        f"\nAMP:           {amp_enabled} (dtype: {amp_dtype})"
        f"\ntorch.compile: {compile_model}",
    )
    # fmt: on

    bins = (
        None
        if num_emb_type != "pwl" or n_cont_features == 0
        else rtdl_num_embeddings.compute_bins(
            data["train"]["x_cont"], n_bins=num_emb_n_bins
        )
    )

    model = tabm_reference.Model(
        n_num_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes if n_classes > 0 else None,
        backbone={
            "type": "MLP",
            "n_blocks": (
                n_blocks if n_blocks != "auto" else (3 if bins is None else 2)
            ),
            "d_block": d_block,
            "dropout": dropout,
        },
        bins=bins,
        num_embeddings=(
            None
            if bins is None
            else {
                "type": "PiecewiseLinearEmbeddings",
                "d_embedding": d_embedding,
                "activation": False,
                "version": "B",
            }
        ),
        arch_type=arch_type,
        k=tabm_k,
        share_training_batches=share_training_batches,
    ).to(device)
    optimizer = torch.optim.AdamW(
        make_parameter_groups(model), lr=lr, weight_decay=weight_decay
    )

    if compile_model:
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument
        # (mode="reduce-overhead" caused issues during training with
        # torch==2.0.1).
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    # type: ignore[code]
    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
    def apply_model(part: str, idx: torch.Tensor) -> torch.Tensor:
        return (
            model(
                data[part]["x_cont"][idx],
                data[part]["x_cat"][idx] if "x_cat" in data[part] else None,
            )
            .squeeze(-1)  # Remove the last dimension for regression tasks.
            .float()
        )

    # TODO: use BCELoss for binary classification
    base_loss_fn = (
        torch.nn.functional.mse_loss
        if task_type == "regression"
        else torch.nn.functional.cross_entropy
    )

    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TabM produces k predictions per object. Each of them must be trained
        # separately.
        # (regression)     y_pred.shape == (batch_size, k)
        # (classification) y_pred.shape == (batch_size, k, n_classes)
        k = y_pred.shape[1]
        return base_loss_fn(
            y_pred.flatten(0, 1),
            (
                y_true.repeat_interleave(k)
                if model.share_training_batches
                else y_true
            ),
        )

    @evaluation_mode()
    def evaluate(part: str) -> float:
        model.eval()

        # When using torch.compile, you may need to reduce the evaluation
        # batch size.
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(
                        len(data[part]["y"]), device=device
                    ).split(
                        eval_batch_size,
                    )
                ],
            )
            .cpu()
            .numpy()
        )
        if task_type == "regression":
            # Transform the predictions back to the original label space.
            y_pred = y_pred * self.y_std_ + self.y_mean_

        # Compute the mean of the k predictions.
        average_logits = self.config.get("average_logits", False)
        if average_logits:
            y_pred = y_pred.mean(1)
        if task_type != "regression":
            # For classification, the mean must be computed in the probability
            # space.
            y_pred = scipy.special.softmax(y_pred, axis=-1)
        if not average_logits:
            y_pred = y_pred.mean(1)

        return compute_metric(
            y=data[part]["y"].cpu().numpy(),
            metric=self.early_stopping_metric,
            y_pred=y_pred if task_type == "regression" else y_pred.argmax(1),
            y_pred_proba=y_pred[:, 1] if task_type == "binclass" else y_pred,
            silent=True,
        )

    math.ceil(n_train / batch_size)
    best = {
        "val": -math.inf,
        # 'test': -math.inf,
        "epoch": -1,
    }
    best_params = [p.clone() for p in model.parameters()]
    # Early stopping: the training stops when
    # there are more than `patience` consecutive bad updates.
    remaining_patience = patience

    try:
        if self.config.get("verbosity", 0) >= 1:
            from tqdm.std import tqdm
        else:
            tqdm = lambda arr, desc: arr
    except ImportError:
        tqdm = lambda arr, desc: arr

    logger.log(15, "-" * 88 + "\n")
    for epoch in range(n_epochs):
        # check time limit
        if epoch > 0 and time_to_fit_in_seconds is not None:
            pred_time_after_next_epoch = (
                (epoch + 1) / epoch * (time.time() - start_time)
            )
            if pred_time_after_next_epoch >= time_to_fit_in_seconds:
                break

        batches = (
            torch.randperm(n_train, device=device).split(batch_size)
            if model.share_training_batches
            else [
                x.transpose(0, 1).flatten()
                for x in torch.rand((model.k, n_train), device=device)
                .argsort(dim=1)
                .split(batch_size, dim=1)
            ]
        )

        for batch_idx in tqdm(batches, desc=f"Epoch {epoch}"):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model("train", batch_idx), Y_train[batch_idx])

            # added from
            # https://github.com/yandex-research/tabm/blob/main/bin/model.py
            if (
                gradient_clipping_norm is not None
                and gradient_clipping_norm != "none"
            ):
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(),
                    gradient_clipping_norm,
                )

            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()  # type: ignore
                # Ignores grad scaler might skip steps; should not break
                # anything
                grad_scaler.step(optimizer)
                grad_scaler.update()

        val_score = evaluate("val")
        logger.log(15, f"(val) {val_score:.4f}")

        if (
            not math.isfinite(val_score) or abs(val_score) > 1e10
        ):  # 1e10 is arbitrary threshold
            logger.warning(
                f"Validation score diverged ({val_score}), stopping early."
            )
            break

        if val_score > best["val"]:
            logger.log(15, "🌸 New best epoch! 🌸")
            # best = {'val': val_score, 'test': test_score, 'epoch': epoch}
            best = {"val": val_score, "epoch": epoch}
            remaining_patience = patience
            with torch.no_grad():
                for bp, p in zip(best_params, model.parameters()):
                    bp.copy_(p)
        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            break

    logger.log(15, "\n\nResult:")
    logger.log(15, str(best))

    logger.log(15, "Restoring best model")
    with torch.no_grad():
        for bp, p in zip(best_params, model.parameters()):
            p.copy_(bp)

    self.model_ = model
