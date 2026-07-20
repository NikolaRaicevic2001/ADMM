"""Horizon dual shift test (standalone)."""

from __future__ import annotations

import numpy as np

from utils.math_utils import shift_horizon_zero_tail


def test_shift_last_is_zero() -> None:
    seq = np.ones((5, 3))
    out = shift_horizon_zero_tail(seq)
    assert np.allclose(out[:-1], 1.0)
    assert np.allclose(out[-1], 0.0)
