from typing import Tuple

import torch
import torch.nn as nn

from .utils import FCNN
from .base import BaseFlow


class RealNVP(BaseFlow):
    def __init__(self, dim: int, hidden_dim: int = 8, base_network: nn.Module = FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + \
            torch.sum(s2_transformed, dim=1)
        return z.flip(dims=(1,)), log_det

    def backward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
            torch.sum(-s2_transformed, dim=1)
        return x, log_det
