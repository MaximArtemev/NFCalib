import os
import pytest

import torch
from torch.distributions import MultivariateNormal
from sklearn.datasets import make_moons
import numpy as np

from src.nf.classic import NormalizingFlowModel, MAF, RealNVP

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def moon_ds(device) -> torch.Tensor:
    X, _ = make_moons(1000)
    return torch.from_numpy(X.astype(np.float32)).to(device)


@pytest.fixture(params=["maf", "real_nvp"])
def moon_flow(request, device: torch.device) -> NormalizingFlowModel:
    dim = 2
    prior = MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device))
    if request.param == 'maf':
        return NormalizingFlowModel(dim, prior, [MAF(dim), MAF(dim)]).to(device)
    elif request.param == 'real_nvp':
        return NormalizingFlowModel(dim, prior, [RealNVP(dim), RealNVP(dim)]).to(device)
    else:
        assert 0 == 1
