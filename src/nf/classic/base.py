from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseUnconditionalFlow(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def backward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class BaseConditionalFlow(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def backward(self, z: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
