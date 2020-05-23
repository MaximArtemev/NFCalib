import pytest
from copy import deepcopy
from typing import Tuple, Optional

import numpy as np
from scipy import stats
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal

from src.nf import (
    ConditionalNormalizingFlowModel,
    cond_neg_log_likelihood, one_hot_encoding,
    ConditionalMAF, BaseConditionalFlow, BaseUnconditionalDistribution,
    ConditionalNormal, BaseConditionalDistribution
)
from src.nf.utils import Vector


def test_cond_normal():
    dist = ConditionalNormal(torch.zeros(3), torch.eye(3), lambda x: x[0], lambda x: x[1])
    x = torch.FloatTensor([[0, 0, 0], [1, 2, 3], [-4, 0, 5]])
    mu = torch.FloatTensor([[0, 0, 0], [6, 5, 4], [3, 0, 2]])
    log_sigma = torch.FloatTensor([[0, 0, 0], [3, 2, 1], [6, 0, 5]])
    n_samples = 200
    assert abs(torch.mean(
        dist.sample_n(n_samples, [torch.zeros((n_samples, 3)), torch.zeros((n_samples, 3))])
    )) < 0.15
    assert abs(torch.std(
        dist.sample_n(n_samples, [torch.zeros((n_samples, 3)), torch.zeros((n_samples, 3))])
    ) - 1) < 0.15

    for a, m, s, log_p in zip(
        x, mu, log_sigma,
        dist.log_prob(x, [mu, log_sigma])
    ):
        naive_res = MultivariateNormal(m, torch.exp(2 * s) * torch.eye(s.shape[0])).log_prob(a)
        assert torch.allclose(log_p, naive_res)


def test_probs():
    x, mu, sigma = 1., 3., 4.

    np_res = stats.norm.logpdf(x, loc=mu, scale=sigma)
    torch_res = MultivariateNormal(torch.FloatTensor([mu]), torch.FloatTensor([[sigma**2]])).log_prob(x).item()
    torch_std_res = (MultivariateNormal(torch.zeros(1), torch.eye(1)).log_prob((x - mu) / sigma) - np.log(sigma)).item()

    assert abs(np_res - torch_res) < 0.01
    assert abs(np_res - torch_std_res) < 0.01


def test_cond_maf_instance():
    maf = ConditionalMAF(1, 2)
    assert isinstance(maf, BaseConditionalFlow)
    assert not isinstance(maf, BaseUnconditionalDistribution)


def test_cond_nf_model_logp(
        cond_moon_flow: ConditionalNormalizingFlowModel,
        cond_moon_ds: Tuple[torch.Tensor, torch.Tensor]
):
    model = deepcopy(cond_moon_flow)
    log_prob = model.log_prob(cond_moon_ds[0], cond_moon_ds[1])

    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.shape == (cond_moon_ds[0].shape[0], )
    assert log_prob.requires_grad


def test_nf_model_sample(cond_moon_flow: ConditionalNormalizingFlowModel):
    model = deepcopy(cond_moon_flow)
    n_samples = 1234
    cond = np.hstack([np.zeros(n_samples // 2, dtype=np.uint8), np.ones(n_samples // 2, dtype=np.uint8)])
    generated = model.sample_n(n_samples, one_hot_encoding(cond, 2, True))

    assert isinstance(generated, torch.Tensor)
    assert generated.shape == (n_samples, 2)
    assert generated.dtype == torch.float32
    assert not generated.requires_grad


def test_cnf_log_probs():
    class MockCondDist(BaseConditionalDistribution):
        def __init__(self, dim, cond_dim):
            self.dim = dim
            self.cond_dim = cond_dim

        def _log_prob_impl(self, x: Vector, condition: Optional[Vector]) -> Vector:
            assert condition.shape[1] == self.cond_dim
            return torch.sum(x, dim=1) + torch.sum(condition, dim=1)

        def _sample_n_impl(self, n: int, condition: Optional[Vector]) -> Vector:
            assert condition.shape[0] == n
            assert condition.shape[1] == self.cond_dim
            return torch.ones((n, self.dim))

    class MockConditionalFlow(BaseConditionalFlow):
        def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return x, torch.sum(condition, dim=1)

        def backward(self, z: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return z, -torch.sum(condition, dim=1)

    dim, cond_dim = 2, 1
    model = ConditionalNormalizingFlowModel(dim, cond_dim, MockCondDist(dim, cond_dim), [MockConditionalFlow()])

    x = torch.FloatTensor([[1, 2], [3, 4]])
    condition = torch.FloatTensor([[5], [6]])
    z, prior_logprob, log_det = model.forward(x, condition)

    assert torch.allclose(z, x)
    assert torch.allclose(prior_logprob, torch.FloatTensor([1 + 2 + 5, 3 + 4 + 6]))
    assert torch.allclose(log_det, torch.FloatTensor([5, 6]))

    z, log_det = model.backward(x, condition)
    assert torch.allclose(z, x)
    assert torch.allclose(log_det, -torch.FloatTensor([5, 6]))


def test_forward_backward_test(
        cond_moon_flow: ConditionalNormalizingFlowModel,
        cond_moon_ds: Tuple[torch.Tensor, torch.Tensor]
):
    model = deepcopy(cond_moon_flow)

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    epochs = 10
    metrics = []
    for i in range(epochs):
        loss = model.ll_train_step(optimizer, *cond_moon_ds)
        metrics.append(loss)

    n_samples = 100
    condition = torch.from_numpy(np.vstack([
        one_hot_encoding(np.zeros(n_samples // 2, dtype=np.uint32), 2, True),
        one_hot_encoding(np.ones(n_samples // 2, dtype=np.uint32), 2, True)
    ]))
    z = model.prior.sample_n(n_samples, condition)
    samples, log_det = model.backward(z, condition)
    logp_z = model.prior.log_prob(z, condition)

    z_, logp_z_, log_det_ = model.forward(samples, condition)

    assert torch.allclose(z, z_)
    assert torch.allclose(logp_z, logp_z_)
    assert torch.allclose(log_det, -log_det_)


def test_maf_train(
        cond_moon_flow: ConditionalNormalizingFlowModel,
        cond_moon_ds: Tuple[torch.Tensor, torch.Tensor]
):
    model = deepcopy(cond_moon_flow)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 100
    metrics = []
    for i in range(epochs):
        loss = model.ll_train_step(optimizer, *cond_moon_ds)
        metrics.append(loss)

    assert np.mean(metrics[-10:]) < 2
    assert \
        cond_neg_log_likelihood(model, *cond_moon_ds) == pytest.approx(metrics[-1], 0.1), \
        'negative log likelihood should be approximately equal to the loss before the last update'


def test_nf_model_forward(
        cond_moon_flow: ConditionalNormalizingFlowModel,
        cond_moon_ds: Tuple[torch.Tensor, torch.Tensor]
):
    model = deepcopy(cond_moon_flow)
    z, prior_logprob, log_det = model.forward(*cond_moon_ds)

    assert isinstance(z, torch.Tensor)
    assert isinstance(prior_logprob, torch.Tensor)
    assert isinstance(log_det, torch.Tensor)

    assert z.shape == cond_moon_ds[0].shape
    assert prior_logprob.shape == (cond_moon_ds[0].shape[0], )
    assert log_det.shape == (cond_moon_ds[0].shape[0], )


def test_nf_model_backward(
        cond_moon_flow: ConditionalNormalizingFlowModel,
        cond_moon_ds: Tuple[torch.Tensor, torch.Tensor]
):
    model = deepcopy(cond_moon_flow)
    x, log_det = model.backward(*cond_moon_ds)

    assert isinstance(x, torch.Tensor)
    assert isinstance(log_det, torch.Tensor)

    assert x.shape == cond_moon_ds[0].shape
    assert log_det.shape == (cond_moon_ds[0].shape[0], )
