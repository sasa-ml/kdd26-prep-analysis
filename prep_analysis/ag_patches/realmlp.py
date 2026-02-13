import pytabkit.models.data.conversion
import pytabkit.models.sklearn.sklearn_base
import torch
from autogluon.tabular.models.realmlp.realmlp_model import RealMLPModel
from pytabkit.models.training.lightning_modules import TabNNModule


def disable_RealMLP_prep():
    TabNNModule.training_step = training_step

    pytabkit.models.data.conversion.OrdinalEncoder.transform = transform
    pytabkit.models.data.conversion.OrdinalEncoder.fit_transform = (
        fit_transform
    )

    pytabkit.models.sklearn.sklearn_base.OrdinalEncoder.transform = transform
    pytabkit.models.sklearn.sklearn_base.OrdinalEncoder.fit_transform = (
        fit_transform
    )

    RealMLPModel._original_get_model_params = RealMLPModel._get_model_params
    RealMLPModel._get_model_params = _get_model_params


def training_step(self, batch, batch_idx):
    # x = batch["x_cont"]
    # x = x / (1e-8 + x.std(dim=-2, keepdim=True))
    # print(f'{x.mean().item()=}')
    # print(f'{list(self.model.parameters())[0].mean().item()=}')
    # print(f'{list(self.model.parameters())[-1].mean().item()=}')
    output = self.model(batch)
    opt = self.optimizers()
    # do sum() over models dimension
    loss = self.criterion(output["x_cont"], output["y"]).sum()
    # print(f'{loss.item()=}')
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite train loss detected "
            f"(epoch={self.progress.epoch}, batch={batch_idx}, "
            + f"loss={loss.item()})"
        )
    # Callbacks for regularization are called before the backward pass
    self.manual_backward(loss)
    opt.step(loss=loss)
    opt.zero_grad()

    self.progress.total_samples += batch["y"].shape[-2]
    self.progress.epoch_float = (
        self.progress.total_samples / self.train_dl.get_num_iterated_samples()
    )
    return loss


def transform(self, X):
    return X.to_numpy()


def fit_transform(self, X, y=None):
    return self.fit(X, y=None).transform(X)


def _get_model_params(
    self, convert_search_spaces_to_default: bool = False
) -> dict:
    params = self._original_get_model_params(convert_search_spaces_to_default)
    params["tfms"] = []
    return params
