import pytest
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim

from src.nf.classic import NormalizingFlowModel
from src.nf.metrics import neg_log_likelihood


def test_nf_model_logp(moon_flow: NormalizingFlowModel, moon_ds: torch.Tensor):
    model = deepcopy(moon_flow)
    log_prob = model.log_prob(moon_ds)

    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.shape == (moon_ds.shape[0], )
    assert log_prob.requires_grad


def test_nf_model_sample(moon_flow: NormalizingFlowModel):
    model = deepcopy(moon_flow)
    n_samples = 1234
    generated = model.sample_n(n_samples)

    assert isinstance(generated, torch.Tensor)
    assert generated.shape == (n_samples, 2)
    assert generated.dtype == torch.float32
    assert not generated.requires_grad


def test_maf_train(moon_flow: NormalizingFlowModel, moon_ds: torch.Tensor):
    model = deepcopy(moon_flow)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 100
    metrics = []
    for i in range(epochs):
        loss = model.ll_train_step(optimizer, moon_ds)
        metrics.append(loss)

    assert np.mean(metrics[-10:]) < 2
    assert \
        neg_log_likelihood(model, moon_ds) == pytest.approx(metrics[-1], 0.1), \
        'negative log likelihood should be approximately to the loss before the last update'


def test_nf_model_forward(moon_flow: NormalizingFlowModel, moon_ds: torch.Tensor):
    model = deepcopy(moon_flow)
    z, prior_logprob, log_det = model.forward(moon_ds)

    assert isinstance(z, torch.Tensor)
    assert isinstance(prior_logprob, torch.Tensor)
    assert isinstance(log_det, torch.Tensor)

    assert z.shape == moon_ds.shape
    assert prior_logprob.shape == (moon_ds.shape[0], )
    assert log_det.shape == (moon_ds.shape[0], )


def test_nf_model_backward(moon_flow: NormalizingFlowModel, moon_ds: torch.Tensor):
    model = deepcopy(moon_flow)
    x, log_det = model.backward(moon_ds)

    assert isinstance(x, torch.Tensor)
    assert isinstance(log_det, torch.Tensor)

    assert x.shape == moon_ds.shape
    assert log_det.shape == (moon_ds.shape[0], )

