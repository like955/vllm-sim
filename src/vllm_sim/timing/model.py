"""Timing models for unified prefill+decode forward passes.

Three implementations:
*   ``LinearTimingModel`` — per-token linear (default).
*   ``ProfileTimingModel`` — interpolates from a hardware profile table.
*   ``AnalyticalTimingModel`` — physics-based: GEMM + Attention + penalty.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Protocol

from vllm_sim.engine.config import EngineSimConfig


class TimingModel(Protocol):
    """One forward pass duration.

    *p_hit* / *p_miss*: prefill tokens (hits discounted).
    *d_reqs* / *d_mult*: decode concurrency and jump-depth.
    *seq_len_sum_sqrt*: Σ sqrt(seq_len_i) for attention IO term.
    *is_mixed*: whether this batch mixes prefill and decode.
    """

    def step_us(
        self, p_hit: int, p_miss: int, d_reqs: int, d_mult: int,
        seq_len_sum_sqrt: float = 0.0, is_mixed: bool = False,
    ) -> float: ...


# ---------------------------------------------------------------------------
# Linear (default)
# ---------------------------------------------------------------------------


class LinearTimingModel:
    """Per-token linear model, prefix-cache aware."""

    def __init__(self, config: EngineSimConfig) -> None:
        self._cfg = config

    def step_us(
        self, p_hit: int, p_miss: int, d_reqs: int, d_mult: int,
        seq_len_sum_sqrt: float = 0.0, is_mixed: bool = False,
    ) -> float:
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
# Profile-driven
# ---------------------------------------------------------------------------


class ProfileTimingModel:
    """2-D hardware profile lookup table."""

    _OVERLAP_FRACTION = 0.3

    def __init__(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self._base_us = float(raw.get("base_us", 0.0))
        self._prefill_x, self._prefill_y = self._parse(raw["prefill"])
        self._decode_x, self._decode_y = self._parse(raw["decode"])

    @staticmethod
    def _parse(pairs: list) -> tuple[list[int], list[float]]:
        xs, ys = [], []
        for x, y in pairs:
            xs.append(int(x))
            ys.append(float(y))
        return xs, ys

    def step_us(
        self, p_hit: int, p_miss: int, d_reqs: int, d_mult: int,
        seq_len_sum_sqrt: float = 0.0, is_mixed: bool = False,
    ) -> float:
        effective_p = p_miss + int(0.1 * p_hit)
        if effective_p <= 0 and d_reqs <= 0:
            return 0.0
        p_us = self._lookup(self._prefill_x, self._prefill_y, effective_p)
        if d_reqs <= 0:
            return self._base_us + p_us
        d_us = self._lookup(self._decode_x, self._decode_y, d_reqs)
        single = self._base_us + max(p_us, d_us) + self._OVERLAP_FRACTION * min(p_us, d_us)
        return single * max(d_mult, 1)

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
# Analytical (physics-based)
# ---------------------------------------------------------------------------


class AnalyticalTimingModel:
    r"""Physics-based model grounded in GPU kernel behaviour.

    One forward pass costs::

        step_us = [ α × N   (GEMM — total tokens)
                  + β × Σ sqrt(L_i)   (Attention IO — KV cache scan)
                  + γ × 1_{mixed} ]   (non-uniform batch penalty)
                × d_mult              (jump-decoding multiplier)

    *N*: effective tokens = p_miss + hit_discount × p_hit + d_reqs.
    *L_i*: sequence length of each request in the batch.
    *d_mult*: decode jump-depth (passes being simulated).
    """

    def __init__(self, config: EngineSimConfig) -> None:
        self._alpha = config.analytical_alpha_us
        self._beta = config.analytical_beta_us
        self._gamma = config.analytical_gamma_us
        self._hit_ratio = config.prefix_hit_cost_ratio

    def step_us(
        self, p_hit: int, p_miss: int, d_reqs: int, d_mult: int,
        seq_len_sum_sqrt: float = 0.0, is_mixed: bool = False,
    ) -> float:
        N = p_miss + self._hit_ratio * p_hit + d_reqs
        if N <= 0:
            return 0.0
        single = self._alpha * N + self._beta * seq_len_sum_sqrt
        if is_mixed:
            single += self._gamma
        return single * max(d_mult, 1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_timing(config: EngineSimConfig) -> TimingModel:
    if config.timing_profile:
        return ProfileTimingModel(config.timing_profile)
    if config.timing_model == "analytical":
        return AnalyticalTimingModel(config)
    return LinearTimingModel(config)
