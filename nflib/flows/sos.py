"""
Implements various flows.
Each flow is invertible so it can be forward()ed and inverse()ed.
Each flow also outputs its log det J "regularization"

Reference:
Sum-of-Squares Polynomial Flow, Priyank Jain et al. June 2019
https://arxiv.org/pdf/1905.02325.pdf
"""

import logging

import torch
from torch import nn

from nflib.nn.networks import ARMLP, MLP

logger = logging.getLogger('main.nflib.flows.SOSFlows')


class SOSFlow(nn.Module):
    """
    todo add explanation
    """

    def __init__(self, dim, k=2, r=2, base_network=ARMLP, hidden_size=24, **base_network_kwargs):
        """
        Args:
            dim: input shape
            k: number of squares (of polynomials)
            r: degree of polynomials
        """

        super().__init__()
        self.dim = dim
        self.k = k
        self.m = r + 1
        self.net = base_network(dim, self.k * self.m * self.dim + self.dim, hidden_size, **base_network_kwargs)
        self.register_buffer('placeholder', torch.randn(1))

        n = torch.arange(self.m).unsqueeze(1)
        e = torch.ones(self.m).unsqueeze(1).long()

        self.filter = ((n.mm(e.transpose(0, 1))) + (e.mm(n.transpose(0, 1))) + 1).float()

    def _transform(self, xx, cc):
        # сс: b* d * k * m * m
        # xx: b* d * 1* m * 1
        cc_xx = torch.matmul(cc, xx)  # bs x d x k x m x 1
        xx_cc_xx = torch.matmul(xx.transpose(3, 4), cc_xx)
        # bs x d x k x 1 x 1
        summed = xx_cc_xx.squeeze(-1).squeeze(-1).sum(-1)  # bs x d

        return summed

    def forward(self, x, context=None):
        """
        Args:
        x: shape[b, d]
        context: condition/context
        """
        params = self.net(x, context=None)
        i = self.k * self.m * self.dim
        # split params
        c = params[:, :i]\
            .view(x.size(0), -1, self.dim) \
            .transpose(1, 2) \
            .view(x.size(0), self.dim, self.k, self.m, 1)
        # c: b*n*k*m*1
        const = params[:, i:].view(x.size(0), self.dim)
        # const: b*n
        # C: b*n*k*m*1 X b*n*k *1 *m = b * n* k *m * m
        # every square parts (one of k summed up parts) has m*m parameters
        C = torch.matmul(c, c.transpose(3, 4))

        # C: bs x d x k x m x m
        # const: bs x d

        zz = x.unsqueeze(-1) ** (torch.arange(self.m).float().to(x.device))
        zz = zz.view(x.size(0), x.size(1), 1, self.m, 1)

        # X: bs x d x 1 x m x 1
        ss = self._transform(zz, C / self.filter) * x + const
        # S: bs x d x 1 x m x 1
        # S = T-inverse(X), T-inverse is SoS-flow,
        #  i.e., si =T_i-inverse(x_i)= c+ integral_0^{x_i}
        # (sum of squares dependent on x_1,..x_{i-1})
        # logdet(T-inverse)= log(abs((partial T_1/partial x_1)
        #           *(partial T_2/partial x_2),..(partial T_d/partial x_d)))
        # (partial T_i/partial x_i)= sum of squares (where u = x_i)
        logdet = torch.log(torch.abs(self._transform(zz, C))).sum(dim=1)

        # logdet: log(jacobian of T-inverse)
        # logdet: -log(jacobian of T )
        return ss, logdet

    def inverse(self, z, context=None):
        xx, logdet = self.forward(z, context=context)
        return xx, logdet
