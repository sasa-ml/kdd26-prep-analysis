from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_DATETIME,
    R_FLOAT,
    R_INT,
    R_OBJECT,
)
from autogluon.core.constants import QUANTILE, REGRESSION
from autogluon.tabular.models.fastainn.tabular_nn_fastai import (  # logger,
    NNFastAiTabularModel,
)


def disable_NN_FastAI_prep():
    NNFastAiTabularModel._original_fit = NNFastAiTabularModel._fit
    NNFastAiTabularModel._fit = _fit

    NNFastAiTabularModel._original_get_model_params = (
        NNFastAiTabularModel._get_model_params
    )
    NNFastAiTabularModel._get_model_params = _get_model_params
    NNFastAiTabularModel._preprocess_train = _preprocess_train


LABEL = "__label__"


def _get_model_params(
    self, convert_search_spaces_to_default: bool = False
) -> dict:
    params = self._original_get_model_params(convert_search_spaces_to_default)
    params['y_scaler'] = None
    params['clipping'] = False
    return params


def _preprocess_train(self, X, y, X_val, y_val):
    from fastai.data.block import CategoryBlock, RegressionBlock
    from fastai.data.transforms import IndexSplitter
    from fastai.tabular.core import TabularPandas
    from fastcore.basics import range_of

    # logger.log(15, f"Using {len(self.cont_columns)} cont features")
    df_train, train_idx, val_idx = self._generate_datasets(X, y, X_val, y_val)
    y_block = (
        RegressionBlock()
        if self.problem_type in [REGRESSION, QUANTILE]
        else CategoryBlock()
    )

    self.cont_columns = self._feature_metadata.get_features(
        valid_raw_types=[R_INT, R_FLOAT, R_DATETIME]
    )
    self.cat_columns = self._feature_metadata.get_features(
        valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL]
    )

    # Copy cat_columns and cont_columns
    # because TabularList is mutating the list
    data = TabularPandas(
        df_train,
        cat_names=self.cat_columns.copy(),
        cont_names=self.cont_columns.copy(),
        procs=None,
        y_block=y_block,
        y_names=LABEL,
        splits=IndexSplitter(val_idx)(range_of(df_train)),
    )
    return data


def _fit(
    self,
    X,
    y,
    X_val=None,
    y_val=None,
    time_limit=None,
    num_cpus=None,
    num_gpus=0,
    sample_weight=None,
    **kwargs,
):
    self._original_fit(
        X,
        y,
        X_val,
        y_val,
        time_limit,
        num_cpus,
        num_gpus,
        sample_weight,
        **kwargs,
    )
    self.y_scaler = None
