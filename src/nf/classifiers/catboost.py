from typing import Dict, Any

from catboost import CatBoostClassifier

from src.nf.classic import NormalizingFlowModel
from src.nf.utils import Vector
from .utils import make_clf_dataset


def train_catboost_clf(
    true_ds: Vector,
    model: NormalizingFlowModel,
    cb_params: Dict[str, Any]
):
    clf_ds = make_clf_dataset(true_ds, model)
    return clf_ds, CatBoostClassifier(**cb_params).fit(clf_ds[:, :-1], clf_ds[:, -1], verbose=0)
