from typing import List

import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Densnet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.layers = []
        x = in_dim
        for dim in hidden_dims + [out_dim]:
            self.layers.append(nn.Linear(x, dim))
            x += dim

        self.layers = nn.ModuleList(self.layers)
        self.activations = [nn.ReLU()] * (len(hidden_dims) - 1)
        self.activations.append(nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers[0](x)
        for activation, layer in zip(self.layers[1:], self.activations):
            x = torch.cat([x, out], dim=1)
            out = layer(activation(x))
        return out
