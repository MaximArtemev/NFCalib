import pytest
from copy import deepcopy
from typing import Callable, Union

import torch
import torch.optim as optim
import torch.nn as nn
from catboost import CatBoostClassifier
import numpy as np
from torch.distributions.distribution import Distribution

from src.nf.classic import NormalizingFlowModel
from src.nf.classifiers import make_clf_dataset
from src.nf.calibration import CalibratedModel
from src.nf import to_numpy, neg_log_likelihood


@pytest.fixture
def trained_flow_moon(moon_flow: NormalizingFlowModel, moon_ds: torch.Tensor) -> NormalizingFlowModel:
    model = deepcopy(moon_flow)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    for _ in range(100):
        model.ll_train_step(optimizer, moon_ds)

    return model


@pytest.fixture
def clf_ds(trained_flow_moon: NormalizingFlowModel, moon_ds: torch.Tensor) -> np.ndarray:
    return make_clf_dataset(moon_ds, trained_flow_moon)


def test_clf_ds(moon_ds, clf_ds):
    assert clf_ds.shape == (moon_ds.shape[0] * 2, 3)
    assert clf_ds.dtype == np.float32


@pytest.fixture
def trained_cb(trained_flow_moon: NormalizingFlowModel, clf_ds: np.ndarray) -> CatBoostClassifier:
    clf = CatBoostClassifier(iterations=20)
    clf.fit(clf_ds[:, :-1], clf_ds[:, -1], verbose=0)
    return clf


class MockConstantModelNP(Distribution):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        return -3 * np.ones(x.shape[0], dtype=np.float32)

    def sample_n(self, n: int) -> np.ndarray:
        return np.ones((n, self.dim), dtype=np.float32) * 1.234


class MockConstantModelTorch(nn.Module, Distribution):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.Tensor(1).to(device))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -3 * torch.ones(x.shape[0])

    def sample_n(self, n: int) -> torch.Tensor:
        return torch.ones((n, self.dim)) * 1.234


@pytest.fixture
def clf_mock() -> Callable:
    return lambda x: 2 * np.ones(x.shape[0])


@pytest.fixture(params=["np", "torch"])
def moon_const_model_mock(request, device) -> Union[MockConstantModelNP, MockConstantModelTorch]:
    if request.param == 'np':
        return MockConstantModelNP(2)
    else:
        return MockConstantModelTorch(2, device)


@pytest.fixture
def calibrated_model_mock(clf_mock, moon_const_model_mock) -> CalibratedModel:
    return CalibratedModel(clf_mock, moon_const_model_mock, logit=True)


def test_rejection_constant(calibrated_model_mock: CalibratedModel):
    return calibrated_model_mock.deduce_rejection_constant(10) == 0.1 + np.exp(2)


def test_logp(calibrated_model_mock: CalibratedModel):
    logp = calibrated_model_mock.log_prob(np.ones((10, 2)))
    return np.allclose(to_numpy(logp), -3 + 2)


def test_sample(calibrated_model_mock: CalibratedModel):
    logp = calibrated_model_mock.sample_n(10)
    return np.allclose(logp, 1.234)


@pytest.fixture
def calibrated_model(trained_flow_moon: NormalizingFlowModel, trained_cb: CatBoostClassifier) -> CalibratedModel:
    return CalibratedModel(trained_cb, trained_flow_moon, logit=True)


def test_sample_moon(calibrated_model: CalibratedModel):
    sample = calibrated_model.sample_n(1234)
    assert sample.shape == (1234, 2, )


def test_ll_calibrated(
        calibrated_model: CalibratedModel,
        trained_flow_moon: NormalizingFlowModel,
        moon_ds: torch.Tensor
):
    assert \
        neg_log_likelihood(trained_flow_moon, moon_ds) > neg_log_likelihood(calibrated_model, moon_ds), \
        'Log likelihood of calibrated model should greater'
