import torch
import torch.nn as nn
from torch.distributions import Distribution


def neg_log_likelihood(model: Distribution, x: torch.Tensor) -> float:
    if isinstance(model, nn.Module):
        model.eval()
    with torch.no_grad():
        log_prob = getattr(model, "log_prob", None)
        if callable(log_prob):
            logp_x = model.log_prob(x)
        else:
            _, logp_x = model.inverse(x)
    return -float(logp_x.mean())
