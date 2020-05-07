from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np
from torch.distributions import MultivariateNormal

from .utils import Vector


class BaseDistribution(ABC):
    @abstractmethod
    def _log_prob_impl(self, x: Vector, *condition: Optional[Vector]) -> Vector:
        pass

    @abstractmethod
    def _sample_n_impl(self, n: int, *condition: Optional[Vector]) -> Vector:
        pass


class BaseConditionalDistribution(BaseDistribution, ABC):
    def log_prob(self, x: Vector, *condition: Vector) -> Vector:
        return self._log_prob_impl(x, *condition)

    def sample_n(self, n: int, *condition: Vector) -> Vector:
        return self._sample_n_impl(n, *condition)

    @abstractmethod
    def log_det(self, x: Vector, *condition: Vector) -> Vector:
        pass


class BaseUnconditionalDistribution(BaseDistribution, ABC):
    def log_prob(self, x: Vector) -> Vector:
        return self._log_prob_impl(x, None)

    def sample_n(self, n: int) -> Vector:
        return self._sample_n_impl(n, None)


class TorchDistributionWrapper(BaseUnconditionalDistribution):
    def __init__(self, distribution: torch.distributions.Distribution):
        self._distribution = distribution

    def _log_prob_impl(self, x: Vector, *_):
        return self._distribution.log_prob(x)

    def _sample_n_impl(self, n: int, *_):
        return self._distribution.sample(torch.Size((n,)))


class FakeCondDistribution(BaseConditionalDistribution):
    def __init__(self, distribution: BaseUnconditionalDistribution):
        self._distribution = distribution

    def _log_prob_impl(self, x: Vector, *_):
        return self._distribution.log_prob(x)

    def _sample_n_impl(self, n: int, *_):
        return self._distribution.sample_n(n)

    def log_det(self, x: Vector, *condition: Vector) -> Vector:
        lib = torch if isinstance(x, torch.Tensor) else np
        return lib.zeros(x.shape[0])


class ConditionalNormal(BaseConditionalDistribution, MultivariateNormal):
    def __init__(self, dim: int, device: torch.device = torch.device('cpu')):
        super().__init__(torch.zeros(dim).to(device), torch.eye(dim).to(device))

    def _log_prob_impl(self, x: Vector, *condition: Optional[torch.Tensor]) -> torch.Tensor:
        mu, log_sigma = condition

        return MultivariateNormal.log_prob(
            self,
            (x - mu) * torch.exp(-log_sigma)
        ) - self.log_det(x, mu, log_sigma)

    def _sample_n_impl(self, n: int, *condition: Optional[torch.Tensor]) -> torch.Tensor:
        mu, log_sigma = condition
        return MultivariateNormal.sample_n(self, n) * torch.exp(log_sigma) + mu

    def log_det(self, _, *condition: torch.Tensor) -> torch.Tensor:
        _, log_sigma = condition
        return log_sigma.sum(dim=1)
