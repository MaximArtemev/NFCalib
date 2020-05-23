import os
import pytest
from typing import Tuple

import torch
from torch.distributions import MultivariateNormal
from sklearn.datasets import make_moons
import numpy as np

from src.nf import (
    NormalizingFlowModel, ConditionalMAF, MAF, RealNVP,
    TorchDistributionWrapper, ConditionalNormalizingFlowModel, ConditionalNormal,
    one_hot_encoding, FCNN
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def moon_ds(device) -> torch.Tensor:
    X, _ = make_moons(1000)
    return torch.from_numpy(X.astype(np.float32)).to(device)


@pytest.fixture
def cond_moon_ds(device) -> Tuple[torch.Tensor, torch.Tensor]:
    X, Y = make_moons(1000)
    Y = one_hot_encoding(Y, 2)
    return torch.from_numpy(X.astype(np.float32)).to(device), torch.from_numpy(Y.astype(np.float32)).to(device)


@pytest.fixture(params=["maf", "real_nvp"])
def moon_flow(request, device: torch.device) -> NormalizingFlowModel:
    dim = 2
    prior = MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device))
    if request.param == 'maf':
        return NormalizingFlowModel(dim, TorchDistributionWrapper(prior), [MAF(dim), MAF(dim)]).to(device)
    elif request.param == 'real_nvp':
        return NormalizingFlowModel(dim, TorchDistributionWrapper(prior), [RealNVP(dim), RealNVP(dim)]).to(device)
    else:
        assert 0 == 1, f'Unknown flow: {request.param}'


@pytest.fixture(params=["maf"])
def cond_moon_flow(request, device: torch.device) -> NormalizingFlowModel:
    dim, cond_dim = 2, 2
    prior = ConditionalNormal(
        torch.zeros(dim).to(device), torch.eye(dim).to(device),
        FCNN(cond_dim, dim, 8), FCNN(cond_dim, dim, 8)
    )
    if request.param == 'maf':
        return ConditionalNormalizingFlowModel(
            dim, cond_dim, prior, [ConditionalMAF(dim, cond_dim), ConditionalMAF(dim, cond_dim)]
        ).to(device)
    else:
        assert 0 == 1, f'Unknown conditional flow: {request.param}'
