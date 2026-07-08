"""Physics-based analytical timing model for unified forward passes.

Models one GPU forward pass as three components:

    step_us = [α × N  +  β × Σ sqrt(L_i)  +  γ × 1_mixed] × d_mult

*N*: effective tokens (hits discounted to 10%).
*L_i*: per-request sequence lengths (prompt + generated).
*d_mult*: decode jump-depth (passes simulated in one step).
"""

import math

from vllm_sim.engine.config import EngineSimConfig


class AnalyticalTimingModel:
    r"""Physics-based forward-pass timing.

    =====  =====  ===========================================
    Term   Param  Physics
    =====  =====  ===========================================
    α×N    alpha  GEMM — per-token compute (QKV, FFN)
    β×Σ√L  beta   Attention IO — scanning KV cache per request
    γ      gamma  Mixed-batch penalty (non-uniform kernel)
    =====  =====  ===========================================
    """

    def __init__(self, config: EngineSimConfig) -> None:
        self._alpha = config.alpha_us
        self._beta = config.beta_us
        self._gamma = config.gamma_us
        self._hit_ratio = config.prefix_hit_cost_ratio
        self._d_mult = 1.0  # overridden per call

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


def make_timing(config: EngineSimConfig) -> AnalyticalTimingModel:
    return AnalyticalTimingModel(config)
