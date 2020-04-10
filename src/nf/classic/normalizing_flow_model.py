from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.distributions.distribution import Distribution


class NormalizingFlowModel(nn.Module, Distribution):
    def __init__(self, dim: int, prior: Distribution, flows: Iterable[nn.Module]):
        super().__init__()
        self.dim = dim
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def backward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m, _ = z.shape
        log_det = torch.zeros(m).to(z.device)
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
        x = z
        return x, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        _, prior_logprob, log_det = self.forward(x)
        return prior_logprob + log_det

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        n_samples = sample_shape.numel()
        z = self.prior.sample((n_samples, self.dim))
        if z.dim() != 2:
            z = self.prior.sample((n_samples,))
        x, _ = self.backward(z)
        return x

    def ll_train_step(self, optimizer: Optimizer, train_tensor: torch.Tensor) -> float:
        self.train()
        optimizer.zero_grad()
        logp_x = self.log_prob(train_tensor)
        loss = -torch.mean(logp_x)
        loss.backward()
        optimizer.step()
        return float(loss.mean().detach().cpu())

    def __repr__(self):
        return f'NFModel(dim={self.dim}, ' \
            f'prior={self.prior.__class__.__name__}, ' \
            f'flows={[f.__class__.__name__ for f in self.flows]})'
