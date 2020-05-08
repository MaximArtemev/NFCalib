from typing import Union

import torch
import torch.nn as nn

from .distribution import BaseUnconditionalDistribution, BaseConditionalDistribution


def neg_log_likelihood(model: BaseUnconditionalDistribution, x: torch.Tensor) -> float:
    if isinstance(model, nn.Module):
        model.eval()
    with torch.no_grad():
        log_prob = getattr(model, "log_prob", None)
        if callable(log_prob):
            logp_x = model.log_prob(x)
        else:
            _, logp_x = model.inverse(x)
    return -float(logp_x.mean())


def cond_neg_log_likelihood(model: BaseConditionalDistribution, x: torch.Tensor, *condition: torch.Tensor) -> float:
    if isinstance(model, nn.Module):
        model.eval()
    with torch.no_grad():
        log_prob = getattr(model, "log_prob", None)
        if callable(log_prob):
            logp_x = model.log_prob(x, *condition)
        else:
            _, logp_x = model.inverse(x, *condition)
    return -float(logp_x.mean())
