"""
Implements various flows.
Each flow is invertible so it can be forward()ed and inverse()ed.
Each flow also outputs its log det J "regularization"

Reference:
Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF
 is that it can both generate data and estimate densities with one forward pass only,
  whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)
"""

import logging

import torch
from torch import nn

from nflib.nn.networks import ARMLP, MLP

logger = logging.getLogger('main.nflib.flows.AffineFlows')


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.s = nn.Parameter(torch.zeros((1, dim)), requires_grad=False)
        self.t = nn.Parameter(torch.zeros((1, dim)), requires_grad=False)
        if self.scale:
            self.s = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        if self.shift:
            self.t = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        self.register_buffer('placeholder', torch.randn(1))

    def forward(self, x, context=None):
        z = x * torch.exp(self.s) + self.t
        log_det = torch.sum(self.s, dim=1)
        return z, log_det

    def inverse(self, z, context=None):
        x = (z - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s, dim=1)
        return x, log_det


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, hidden_dim=24, scale=True, shift=True, base_network=MLP, **base_network_kwargs):
        super().__init__()
        self.dim = dim
        self.s_cond = lambda x, context: x.new_zeros(x.size(0), self.dim // 2, device=x.device)
        self.t_cond = lambda x, context: x.new_zeros(x.size(0), self.dim // 2, device=x.device)
        if scale:
            self.s_cond = base_network(self.dim // 2,
                                       self.dim // 2,
                                       hidden_dim,
                                       **base_network_kwargs)
        if shift:
            self.t_cond = base_network(self.dim - (self.dim // 2),
                                       self.dim - (self.dim // 2),
                                       hidden_dim,
                                       **base_network_kwargs)

    def forward(self, x, context=None):
        x0, x1 = x[:, ::2], x[:, 1::2]
        s = self.s_cond(x0, context=context)
        t = self.t_cond(x0, context=context)
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        z = torch.cat([x0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def inverse(self, z, context=None):
        z0, z1 = z[:, ::2], z[:, 1::2]
        s = self.s_cond(z0, context)
        t = self.t_cond(z0, context)
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        x = torch.cat([z0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """

    def __init__(self, dim, base_network=ARMLP, hidden_dim=24, **base_network_kwargs):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.dim = dim
        self.net = base_network(dim,
                                dim * 2,
                                hidden_dim,
                                **base_network_kwargs)

    def forward(self, x, context=None):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x, context=context)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def inverse(self, z, context=None):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0], device=self.placeholder.device)
        for i in range(self.dim):
            st = self.net(x.clone(), context=context)  # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det


class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.inverse = self.inverse, self.forward
