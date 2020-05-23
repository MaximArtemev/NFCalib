from typing import Callable

import numpy as np
from scipy.special import logsumexp

from src.nf.classic import NormalizingFlowModel
from src.nf.utils import to_numpy, Vector


def make_clf_dataset(true_ds: Vector, model: NormalizingFlowModel):
    n = true_ds.shape[0]
    true_ds = to_numpy(true_ds)
    generated_ds = to_numpy(model.sample_n(n))
    return np.vstack([
        np.hstack([true_ds, np.ones(n).reshape(-1, 1)]),
        np.hstack([generated_ds, np.zeros(n).reshape(-1, 1)]),
    ]).astype(np.float32)


def deduce_calibration_constant(
        clf: Callable[[np.ndarray], np.ndarray],
        data: Vector
) -> float:
    return logsumexp(clf(to_numpy(data))) / len(data)
