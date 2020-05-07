from typing import Union

import numpy as np
import torch

Vector = Union[torch.Tensor, np.ndarray]


def to_numpy(x: Vector) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError


def to_torch(x: Vector) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise TypeError


def one_hot_encoding(arr: np.ndarray, s: int, return_tensor: bool = False) -> Vector:
    res = np.zeros((arr.size, s), dtype=np.float32)
    res[np.arange(arr.size), arr] = 1
    if not return_tensor:
        return res
    else:
        return torch.from_numpy(res.astype(np.float32))
