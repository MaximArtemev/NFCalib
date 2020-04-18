import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger('main.nflib.flows.SequenceFlows')


class InvertiblePermutation(nn.Module):
    # @robdrynkin
    def __init__(self, dim):
        super().__init__()
        self.perm = np.random.permutation(dim)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, x, context=None):
        return x[:, self.perm], 0

    def inverse(self, z, context=None):
        return z[:, self.inv_perm], 0


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.flows = nn.ModuleList(flows)

    def forward(self, x, context=None):
        m, _ = x.shape
        log_det = torch.zeros(m, device=self.placeholder.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, context=context)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def inverse(self, z, context=None):
        m, _ = z.shape
        log_det = torch.zeros(m, device=self.placeholder.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, context=context)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x, context=None):
        zs, log_det = self.flow.forward(x, context=context)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.shape[0], -1).sum(1).to(self.placeholder.device)
        return zs, prior_logprob.view(-1, 1), log_det.view(-1, 1)

    def inverse(self, z, context=None):
        xs, log_det = self.flow.inverse(z, context=context)
        return xs, log_det.view(-1)

    def sample(self, num_samples, context=None):
        z = self.prior.sample((num_samples,)).to(self.placeholder.device)
        xs, _ = self.inverse(z, context=context)
        return xs[-1]
