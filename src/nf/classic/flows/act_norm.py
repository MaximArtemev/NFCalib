import torch
import torch.nn as nn

from src.nf.classic.base import BaseUnconditionalFlow


class ActNorm(BaseUnconditionalFlow):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mu = nn.Parameter(torch.zeros(dim, dtype=torch.float), requires_grad=True)
        self.log_sigma = nn.Parameter(torch.zeros(dim, dtype=torch.float), requires_grad=True)

        self.init()

    def init(self):
        torch.nn.init.normal_(self.mu, 0, 2e-5)
        torch.nn.init.normal_(self.log_sigma, 0, 2e-5)

    def forward(self, x):
        z = x * torch.exp(self.log_sigma) + self.mu
        log_det = torch.sum(self.log_sigma)
        return z, log_det

    def backward(self, z):
        x = (z - self.mu) / torch.exp(self.log_sigma)
        log_det = -torch.sum(self.log_sigma)
        return x, log_det
