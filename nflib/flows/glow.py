"""
Implements various flows.
Each flow is invertible so it can be forward()ed and inverse()ed.
Each flow also outputs its log det J "regularization"

Reference:

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"""

import logging

import torch
from torch import nn

from nflib.flows.affine import AffineConstantFlow

logger = logging.getLogger('main.nflib.flows.GlowFlows')


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x, context=None):
        # first batch is used for init
        if not self.data_dep_init_done:
            if self.scale:
                self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            if self.shift:
                self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)


class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = nn.Parameter(P, requires_grad=False)  # remains fixed during optimization
        self.L = nn.Parameter(L, requires_grad=True)  # lower triangular portion
        self.S = nn.Parameter(U.diag(), requires_grad=True)  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1), requires_grad=True)  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.placeholder.device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, context=None):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z, context=None):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det
