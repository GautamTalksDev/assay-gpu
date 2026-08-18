"""IEEE-754 constants from the running NumPy, not guessed cutoffs."""

from __future__ import annotations

import numpy as np

FP32_EPS = float(np.finfo(np.float32).eps)
FP64_EPS = float(np.finfo(np.float64).eps)
