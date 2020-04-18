"""
Predefined Networks
"""

import torch
from torch import nn

from nflib.nn.networks_utils import MADE, ResidualBlock


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh, context=False):
        """ context  - int, False or zero if None""" 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin + int(context), nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x, context=None):
        if context is not None:
            return self.net(torch.cat([context, x], dim=1))
        return self.net(x)


class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh, **base_network_kwargs):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, **base_network_kwargs)

    def forward(self, x, context=None):
        return self.net(x, context=context)


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self, nin, nout, nh, context=False, num_blocks=2):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(nin + int(context), nh),
            nn.BatchNorm1d(nh, eps=1e-3),
            nn.LeakyReLU(0.2)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(
                features=nh,
                context=context,
            ) for _ in range(num_blocks)
        ])
        self.final_layer = nn.Linear(nh, nout)

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=1)
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x, context=context)
        outputs = self.final_layer(x)
        return outputs
