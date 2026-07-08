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

    *P* is the number of prefill tokens, *D* the number of decode
    requests in this step's batch.
    """

    def step_us(self, prefill_tokens: int, decode_tokens: int) -> float: ...


# ---------------------------------------------------------------------------
# Linear (default)
# ---------------------------------------------------------------------------


class LinearTimingModel:
    """Linear model: ``base + cost_per_token × P + cost_per_decode × D``."""

    def __init__(self, config: EngineSimConfig) -> None:
        self._cfg = config

    def step_us(self, prefill_tokens: int, decode_tokens: int) -> float:
        if prefill_tokens <= 0 and decode_tokens <= 0:
            return 0.0
        return (
            self._cfg.prefill_base_us
            + self._cfg.prefill_us_per_token * prefill_tokens
            + self._cfg.decode_base_us
            + self._cfg.decode_us_per_token * decode_tokens
        )


# ---------------------------------------------------------------------------
# Profile-driven (hardware benchmark)
# ---------------------------------------------------------------------------


class ProfileTimingModel:
    r"""Interpolate from a 2-D profile table.

    The profile is a JSON file::

        {
          "prefill": [[128, 120], [1024, 380], [4096, 980], [8192, 1700]],
          "decode":  [[1, 5200], [8, 5500], [32, 6800], [128, 11000]],
          "base_us": 120
        }

    ``prefill`` maps batch token counts → latency (us).
    ``decode`` maps concurrent request counts → per-step latency (us).
    ``base_us`` is a fixed per-step overhead (kernel launch, etc.).

    For a mixed batch the model computes the *bottleneck*::

        step_us = base_us + max(prefill_us(P), decode_us(D))

    This reflects that prefill and decode compete for the same memory
    bandwidth in one forward pass — the slower dominates.
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

    def step_us(self, prefill_tokens: int, decode_tokens: int) -> float:
        if prefill_tokens <= 0 and decode_tokens <= 0:
            return 0.0
        p_us = self._lookup(self._prefill_x, self._prefill_y, prefill_tokens)
        d_us = self._lookup(self._decode_x, self._decode_y, decode_tokens)
        return self._base_us + max(p_us, d_us)

    # ------------------------------------------------------------------
    @staticmethod
    def _lookup(xs: list[int], ys: list[float], n: int) -> float:
        """Linear interpolation.  Clamps to endpoints."""
        if n <= 0:
            return 0.0
        if n <= xs[0]:
            return ys[0] * n / xs[0]
        if n >= xs[-1]:
            # Extrapolate with the slope of the last segment.
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return ys[-1] + slope * (n - xs[-1])

        i = bisect_left(xs, n)
        if xs[i] == n:
            return ys[i]
        # Linear between (xs[i-1], ys[i-1]) and (xs[i], ys[i]).
        frac = (n - xs[i - 1]) / (xs[i] - xs[i - 1])
        return ys[i - 1] + frac * (ys[i] - ys[i - 1])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_timing(config: EngineSimConfig) -> TimingModel:
    """Create a timing model from config.

    If ``config.timing_profile`` is set, loads a profile file;
    otherwise returns the linear default.
    """
    if config.timing_profile:
        return ProfileTimingModel(config.timing_profile)
    return LinearTimingModel(config)
