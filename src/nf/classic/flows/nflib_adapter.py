from typing import Tuple

import torch
import torch.nn as nn

from src.nf.classic.utils import FCNN
from src.nf.classic.base import BaseUnconditionalFlow, BaseConditionalFlow


class NFLibFlowAdapter(BaseConditionalFlow):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.forward(x, condition)

    def backward(self, z: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.inverse(z, condition)
