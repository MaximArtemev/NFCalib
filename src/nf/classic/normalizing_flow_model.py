from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from src.nf.distribution import BaseUnconditionalDistribution, BaseConditionalDistribution, FakeCondDistribution
from src.nf.classic.utils import FCNN
from src.nf.utils import Vector
from src.nf.classic.base import BaseUnconditionalFlow, BaseConditionalFlow


class ConditionalNormalizingFlowModel(nn.Module, BaseConditionalDistribution):
    def __init__(
            self, dim: int, condition_dim: int,
            prior: BaseConditionalDistribution,
            flows: Iterable[Union[BaseUnconditionalFlow, BaseUnconditionalFlow]],
            mu: nn.Module = FCNN, log_sigma: nn.Module = FCNN, hidden_dim=8
    ):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.prior = prior
        self.flows = nn.ModuleList(flows)
        self.mu = mu(condition_dim, dim, hidden_dim)
        self.log_sigma = log_sigma(condition_dim, dim, hidden_dim)

    def forward(self, x: torch.Tensor, *condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = condition[0]
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        for flow in self.flows:
            if isinstance(flow, BaseConditionalFlow):
                x, ld = flow.forward(x, condition)
            else:
                x, ld = flow.forward(x)
            log_det += ld

        mu, log_sigma = self.mu(condition), self.log_sigma(condition)
        z, prior_logprob = x, self.prior.log_prob(x, mu, log_sigma)
        log_det -= self.prior.log_det(x, mu, log_sigma)
        return z, prior_logprob, log_det

    def backward(self, z: torch.Tensor, *condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        condition = condition[0]
        m, _ = z.shape
        log_det = torch.zeros(m).to(z.device)
        mu, log_sigma = self.mu(condition), self.log_sigma(condition)
        z = z * torch.exp(log_sigma) + mu
        log_det += log_sigma.sum(dim=1)
        for flow in self.flows[::-1]:
            if isinstance(flow, BaseConditionalFlow):
                z, ld = flow.backward(z, condition)
            else:
                z, ld = flow.backward(z)
            log_det += ld
        x = z
        return x, log_det

    def _log_prob_impl(self, x: torch.Tensor, *condition: torch.Tensor) -> torch.Tensor:
        _, log_p, log_det = self.forward(x, *condition)
        return log_p + log_det

    def _sample_n_impl(self, n: int, *condition: torch.Tensor) -> torch.Tensor:
        condition = condition[0]
        self.eval()
        with torch.no_grad():
            mu, log_sigma = self.mu(condition), self.log_sigma(condition)
            z = self.prior.sample_n(n, mu, log_sigma)
            x, _ = self.backward(z, condition)
        return x

    def log_det(self, x: Vector, *condition: Vector) -> Vector:
        _, __, log_det = self.forward(x, *condition)
        return log_det

    def ll_train_step(self, optimizer: Optimizer, train_tensor: torch.Tensor, condition_tensor: torch.Tensor) -> float:
        self.train()
        optimizer.zero_grad()
        logp_x = self.log_prob(train_tensor, condition_tensor)
        loss = -torch.mean(logp_x)
        loss.backward()
        optimizer.step()
        return float(loss.mean().detach().cpu())

    def __repr__(self) -> str:
        return f'NFModel(dim={self.dim}, ' \
            f'prior={self.prior.__class__.__name__}, ' \
            f'flows={[f.__class__.__name__ for f in self.flows]})'


class UnconditionalNormalizingFlowModel(nn.Module, BaseUnconditionalDistribution):
    def __init__(
            self, dim: int,
            prior: BaseUnconditionalDistribution,
            flows: Iterable[Union[BaseUnconditionalFlow, BaseUnconditionalFlow]]
    ):
        super().__init__()
        self._inner_nf = ConditionalNormalizingFlowModel(
            dim, 0, FakeCondDistribution(prior), flows, self.zero_net, self.zero_net, 0
        )

    @staticmethod
    def _empty_vec(dim: int):
        return torch.empty(dim, 0)

    def _empty_vec_like(self, x: Vector):
        return self._empty_vec(x.shape[0])

    @staticmethod
    def zero_net(_, dim: int, __):
        return lambda x: torch.zeros(x.shape[0], dim)

    def forward(self, x: torch.Tensor):
        return self._inner_nf.forward(x, self._empty_vec_like(x))

    def backward(self, z: torch.Tensor):
        return self._inner_nf.backward(z, self._empty_vec_like(z))

    def _log_prob_impl(self, x: torch.Tensor, condition=None):
        assert condition is None
        return self._inner_nf._log_prob_impl(x, self._empty_vec_like(x))

    def _sample_n_impl(self, n: int, condition=None):
        assert condition is None
        return self._inner_nf._sample_n_impl(n, self._empty_vec(n))

    def ll_train_step(self, optimizer: Optimizer, train_tensor: torch.Tensor) -> float:
        return self._inner_nf.ll_train_step(optimizer, train_tensor, self._empty_vec_like(train_tensor))


NormalizingFlowModel = UnconditionalNormalizingFlowModel

# class NormalizingFlowModel(nn.Module, Distribution):
#     def __init__(self, dim: int, prior: Distribution, flows: Iterable[nn.Module]):
#         super().__init__()
#         self.dim = dim
#         self.prior = prior
#         self.flows = nn.ModuleList(flows)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         m, _ = x.shape
#         log_det = torch.zeros(m).to(x.device)
#         for flow in self.flows:
#             x, ld = flow.forward(x)
#             log_det += ld
#         z, prior_logprob = x, self.prior.log_prob(x)
#         return z, prior_logprob, log_det
#
#     def backward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         m, _ = z.shape
#         log_det = torch.zeros(m).to(z.device)
#         for flow in self.flows[::-1]:
#             z, ld = flow.backward(z)
#             log_det += ld
#         x = z
#         return x, log_det
#
#     def log_prob(self, x: torch.Tensor) -> torch.Tensor:
#         _, prior_logprob, log_det = self.forward(x)
#         return prior_logprob + log_det
#
#     def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
#         n_samples = sample_shape.numel()
#         z = self.prior.sample((n_samples, self.dim))
#         if z.dim() != 2:
#             z = self.prior.sample((n_samples,))
#         x, _ = self.backward(z)
#         return x
#
    # def ll_train_step(self, optimizer: Optimizer, train_tensor: torch.Tensor) -> float:
    #     self.train()
    #     optimizer.zero_grad()
    #     logp_x = self.log_prob(train_tensor)
    #     loss = -torch.mean(logp_x)
    #     loss.backward()
    #     optimizer.step()
    #     return float(loss.mean().detach().cpu())
    #
    # def __repr__(self):
    #     return f'NFModel(dim={self.dim}, ' \
    #         f'prior={self.prior.__class__.__name__}, ' \
    #         f'flows={[f.__class__.__name__ for f in self.flows]})'
