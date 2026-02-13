from autogluon.core.models import AbstractModel

from prep_analysis.ag_patches.lightgbm import disable_LightGBM_prep
from prep_analysis.ag_patches.nn_fastai import disable_NN_FastAI_prep
from prep_analysis.ag_patches.nn_torch import disable_NN_Torch_prep
from prep_analysis.ag_patches.realmlp import disable_RealMLP_prep
from prep_analysis.ag_patches.tabdpt import disable_TabDPT_prep
from prep_analysis.ag_patches.tabm import disable_TabM_prep
from prep_analysis.ag_patches.tabpfn import disable_TabPFN_prep

# AutoGluon 1.5.0
# TabPFN 6.2.0
# TabDPT 1.1.11

# AbstractModel -> OK
# AbstractNeuralNetworkModel -> N/A
# KNN -> OK (AbstractModel)
# Linear -> OK (AbstractModel)
# Random Forest -> OK (AbstractModel)
# Extra Trees -> OK (AbstractModel)
# XGBoost -> OK (AbstractModel)
# LightGBM -> OK (AbstractModel + internal)
# CatBoost -> OK (AbstractModel)
# EBM -> TODO (or not)
# NN FastAI -> OK (AbstractModel + specific)
# NN Torch -> OK (AbstractModel + specific)
# RealMLP -> OK (AbstractModel + specific + internal)
# TabM -> OK (AbstractModel + specific)
# TabDPT -> OK (AbstractModel + specific + internal)
# TabPFN -> OK (AbstractModel + specific + internal)

# FTTransformer? -> considered multimodal, more complicated?


def disable_AG_model_prep():
    AbstractModel.preprocess = preprocess
    disable_LightGBM_prep()
    disable_NN_FastAI_prep()
    disable_NN_Torch_prep()
    disable_RealMLP_prep()
    disable_TabM_prep()
    disable_TabDPT_prep()
    disable_TabPFN_prep()


def preprocess(
    self,
    X,
    preprocess_nonadaptive=True,
    preprocess_stateful=True,
    **kwargs,
):
    return X
