"""Timing models for unified prefill+decode forward passes.

Two implementations:
*   ``LinearTimingModel`` — simple linear formula (default).
*   ``ProfileTimingModel`` — interpolates from a hardware profile table.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Protocol

from vllm_sim.engine.config import EngineSimConfig


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TimingModel(Protocol):
    """Callable that returns the wall-clock duration of one forward pass.

    *p_hit* / *p_miss*: prefill tokens (hits discounted).
    *d_reqs*: number of requests that decoded this step.
    *d_mult*: max tokens any single decode request generated (≥1,
              the effective number of forward passes being simulated).
    """

    def step_us(self, p_hit: int, p_miss: int, d_reqs: int, d_mult: int) -> float: ...


# ---------------------------------------------------------------------------
# Linear (default)
# ---------------------------------------------------------------------------


class LinearTimingModel:
    """Linear model, prefix-cache aware.

    Hit tokens cost a fraction (``prefix_hit_cost_ratio``) of a full
    prefill token because their KV is already in memory.
    """

    def __init__(self, config: EngineSimConfig) -> None:
        self._cfg = config

    def step_us(self, p_hit: int, p_miss: int, d_reqs: int, d_mult: int) -> float:
        if p_hit <= 0 and p_miss <= 0 and d_reqs <= 0:
            return 0.0
        effective_p = p_miss + self._cfg.prefix_hit_cost_ratio * p_hit
        d_cost = 0.0
        if d_reqs > 0:
            d_cost = max(d_mult, 1) * (
                self._cfg.decode_base_us + self._cfg.decode_us_per_token * d_reqs
            )
        return self._cfg.prefill_base_us + self._cfg.prefill_us_per_token * effective_p + d_cost


# ---------------------------------------------------------------------------
# Profile-driven (hardware benchmark)
# ---------------------------------------------------------------------------


class ProfileTimingModel:
    r"""Interpolate from a 2-D hardware profile table.

    The profile is a JSON file::

        {
          "prefill": [[128, 120], [1024, 380], [4096, 980], [8192, 1700]],
          "decode":  [[1, 5200], [8, 5500], [32, 6800], [128, 11000]],
          "base_us": 120
        }

    For a mixed batch the model computes the *bottleneck*::

        step_us = base_us + max(prefill_us(P), decode_us(D))

    where *P* is the effective prefill token count (hits discounted).
    """

    def __init__(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        self._base_us: float = float(raw.get("base_us", 0.0))
        self._prefill_x: list[int] = []
        self._prefill_y: list[float] = []
        for x, y in raw["prefill"]:
            self._prefill_x.append(int(x))
            self._prefill_y.append(float(y))

        self._decode_x: list[int] = []
        self._decode_y: list[float] = []
        for x, y in raw["decode"]:
            self._decode_x.append(int(x))
            self._decode_y.append(float(y))

    _OVERLAP_FRACTION = 0.3

    def step_us(self, p_hit: int, p_miss: int, d_reqs: int, d_mult: int) -> float:
        effective_p = p_miss + int(0.1 * p_hit)
        if effective_p <= 0 and d_reqs <= 0:
            return 0.0
        p_us = self._lookup(self._prefill_x, self._prefill_y, effective_p)
        if d_reqs <= 0:
            return self._base_us + p_us
        d_us = self._lookup(self._decode_x, self._decode_y, d_reqs)
        single = self._base_us + max(p_us, d_us) + self._OVERLAP_FRACTION * min(p_us, d_us)
        return single * max(d_mult, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _lookup(xs: list[int], ys: list[float], n: int) -> float:
        if n <= 0:
            return 0.0
        if n <= xs[0]:
            return ys[0] * n / xs[0]
        if n >= xs[-1]:
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return ys[-1] + slope * (n - xs[-1])
        i = bisect_left(xs, n)
        if xs[i] == n:
            return ys[i]
        frac = (n - xs[i - 1]) / (xs[i] - xs[i - 1])
        return ys[i - 1] + frac * (ys[i] - ys[i - 1])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_timing(config: EngineSimConfig) -> TimingModel:
    """Create a timing model from config."""
    if config.timing_profile:
        return ProfileTimingModel(config.timing_profile)
    return LinearTimingModel(config)
