from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .utils import Vector


class BaseDistribution(ABC):
    @abstractmethod
    def _log_prob_impl(self, x: Vector, condition: Optional[Vector]) -> Vector:
        pass

    @abstractmethod
    def _sample_n_impl(self, n: int, condition: Optional[Vector]) -> Vector:
        pass


class BaseConditionalDistribution(BaseDistribution, ABC):
    def log_prob(self, x: Vector, condition: Vector) -> Vector:
        return self._log_prob_impl(x, condition)

    def sample_n(self, n: int, condition: Vector) -> Vector:
        return self._sample_n_impl(n, condition)


class BaseUnconditionalDistribution(BaseDistribution, ABC):
    def log_prob(self, x: Vector) -> Vector:
        return self._log_prob_impl(x, None)

    def sample_n(self, n: int) -> Vector:
        return self._sample_n_impl(n, None)


class TorchDistributionWrapper(BaseUnconditionalDistribution):
    def __init__(self, distribution: torch.distributions.Distribution):
        self._distribution = distribution

    def _log_prob_impl(self, x: Vector, _):
        return self._distribution.log_prob(x)

    def _sample_n_impl(self, n: int, _):
        return self._distribution.sample(torch.Size((n,)))


class FakeCondDistribution(BaseConditionalDistribution):
    def __init__(self, distribution: BaseUnconditionalDistribution):
        self._distribution = distribution

    def _log_prob_impl(self, x: Vector, _):
        return self._distribution.log_prob(x)

    def _sample_n_impl(self, n: int, _):
        return self._distribution.sample_n(n)


class ConditionalNormal(nn.Module, BaseConditionalDistribution):
    def __init__(
            self,
            mu: torch.Tensor, sigma: torch.Tensor,
            mu_cond: nn.Module, log_sigma_cond: nn.Module
    ):
        super().__init__()

        self.dist = MultivariateNormal(mu, sigma)
        self.mu = mu_cond
        self.log_sigma = log_sigma_cond

    def _log_prob_impl(self, x: Vector, condition: Optional[torch.Tensor]) -> torch.Tensor:
        mu, log_sigma = self.mu(condition), self.log_sigma(condition)

        return self.dist.log_prob(
            (x - mu) * torch.exp(-log_sigma)
        ) - log_sigma.sum(dim=1)

    def _sample_n_impl(self, n: int, condition: Optional[torch.Tensor]) -> torch.Tensor:
        mu, log_sigma = self.mu(condition), self.log_sigma(condition)
        return self.dist.sample_n(n) * torch.exp(log_sigma) + mu
