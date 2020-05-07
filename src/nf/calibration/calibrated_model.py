import torch
import torch.nn as nn
from torch.distributions import Distribution
import numpy as np
from catboost import CatBoostClassifier

from src.nf.utils import to_numpy, to_torch, Vector


class CalibratedModel(Distribution):
    def __init__(self, clf, model: Distribution, logit=False, n_samples_c=100):
        super().__init__()
        self.clf = self._clf_wrapper(clf)
        self.model = model
        self.logit = logit
        self.c = self.deduce_rejection_constant(n_samples_c)

    def log_prob(self, x: Vector) -> np.ndarray:
        discr = self.clf(to_numpy(x))

        if isinstance(self.model, nn.Module):
            with torch.no_grad():
                self.model.eval()
                x = to_torch(x).to(next(self.model.parameters()).device)
                log_probs = self.model.log_prob(x).detach().cpu().numpy()
        else:
            log_probs = self.model.log_prob(x)

        if self.logit:
            return log_probs + discr
        else:
            return log_probs + np.log(discr) - np.log1p(-discr)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> np.ndarray:
        n_samples = sample_shape.numel()
        return rejection_sampling(self, self.model, self.c, n_samples)

    def deduce_rejection_constant(self, n_samples: int, eps=0.1):
        x = to_numpy(self.model.sample_n(n_samples))
        logits = self.clf(x)
        return np.max(np.exp(logits)) + eps

    @staticmethod
    def _clf_wrapper(clf):
        if isinstance(clf, CatBoostClassifier):
            return lambda x: clf.predict(x, prediction_type='RawFormulaVal')
        else:
            return clf

    def __repr__(self):
        return f'CalibratedModel(clf={self.clf}, model={self.model}, c={self.c}, logit={self.logit})'


def rejection_sampling(density: Distribution, major_dist: Distribution, c: float, n_samples: int) -> np.ndarray:
    n_rej_samples = int(n_samples * c + 1)
    samples = major_dist.sample_n(n_rej_samples)
    major_log_probs = to_numpy(major_dist.log_prob(samples))
    samples = to_numpy(samples)

    log_probs = density.log_prob(samples)

    accept_log_prob = log_probs - np.log(c) - major_log_probs

    is_accept = accept_log_prob > np.log(np.random.uniform(0, 1, n_rej_samples))

    samples = samples[is_accept]
    if samples.shape[0] < n_samples:
        new_c = n_rej_samples / samples.shape[0] if samples.shape[0] != 0 else 2 * c
        new_samples = rejection_sampling(density, major_dist, new_c, n_samples - samples.shape[0])
        samples = np.vstack([samples, new_samples])
    return samples[:n_samples]
