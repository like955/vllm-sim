"""Pluggable timing models for prefill and decode latency."""

from vllm_sim.engine.config import EngineSimConfig


class TimingModel:
    """Computes wall-clock duration of prefill and decode steps.

    The default model is linear in the number of tokens processed.
    Override ``prefill_us`` and ``decode_us`` to implement custom
    models (e.g. non-linear, memory-bandwidth-aware).
    """

    def __init__(self, config: EngineSimConfig) -> None:
        self._cfg = config

    def prefill_us(self, num_tokens: int) -> float:
        """Latency for prefilling *num_tokens* in a single step."""
        if num_tokens <= 0:
            return 0.0
        return self._cfg.prefill_base_us + self._cfg.prefill_us_per_token * num_tokens

    def decode_us(self, num_requests: int) -> float:
        """Latency for one decode token across *num_requests*."""
        if num_requests <= 0:
            return 0.0
        return self._cfg.decode_base_us + self._cfg.decode_us_per_token * num_requests
