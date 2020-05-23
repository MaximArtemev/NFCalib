from typing import Tuple

import torch
import torch.nn as nn

from src.nf.classic.utils import FCNN
from src.nf.classic.base import BaseUnconditionalFlow, BaseConditionalFlow


class ConditionalMAF(BaseConditionalFlow):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 8, base_network: torch.nn.Module = FCNN):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.initial_param = nn.Parameter(torch.empty(2), requires_grad=True)
        for i in range(1, dim):
            self.layers += [base_network(i + cond_dim, 2, hidden_dim)]
        self.reset_parameters()

    @staticmethod
    def init_linear(i):
        def f(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 1e-2 / i)
                torch.nn.init.normal_(m.bias, 0, 1e-2 / i)

        return f

    def reset_parameters(self):
        nn.init.normal_(self.initial_param, 0, 1e-2)

        for i, layer in enumerate(self.layers):
            layer.apply(self.init_linear(i + 1))

    def forward(self, x, condition):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0]).to(x.device)
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](torch.cat([x[:, :i], condition], axis=1))
                mu, alpha = out[:, 0], out[:, 1]
            z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
            log_det -= alpha
        return z.flip(dims=(1,)), log_det

    def backward(self, z, condition):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0]).to(z.device)
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](torch.cat([x[:, :i], condition], axis=1))
                mu, alpha = out[:, 0], out[:, 1]
            x[:, i] = mu + torch.exp(alpha) * z[:, i]
            log_det += alpha
        return x, log_det


class MAF(BaseUnconditionalFlow):
    def __init__(self, dim: int, hidden_dim: int = 8, base_network: torch.nn.Module = FCNN):
        super().__init__()
        self._inner_maf = ConditionalMAF(dim, 0, hidden_dim, base_network)

    def forward(self, x):
        return self._inner_maf.forward(x, torch.empty(x.shape[0], 0).to(x.device))

    def backward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._inner_maf.backward(z, torch.empty(z.shape[0], 0).to(z.device))
