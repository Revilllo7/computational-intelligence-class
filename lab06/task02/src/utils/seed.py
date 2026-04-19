from __future__ import annotations

import random

import numpy as np

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()

import torch  # noqa: E402


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
