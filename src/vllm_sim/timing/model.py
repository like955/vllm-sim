"""Pluggable timing models for unified prefill+decode steps."""

from vllm_sim.engine.config import EngineSimConfig


class TimingModel:
    """Computes wall-clock duration of one unified forward pass.

    There is no separate "prefill step" or "decode step" — one forward
    pass may mix *P* prefill tokens and *D* decode requests in the same
    batch, just like vLLM's scheduler.
    """

    def __init__(self, config: EngineSimConfig) -> None:
        self._cfg = config

    def step_us(self, prefill_tokens: int, decode_tokens: int) -> float:
        """Latency for one batch with *P* prefill tokens and *D* decodes.

        The total is linear in both dimensions: each prefill token and
        each decode request consume memory bandwidth in one forward pass.
        """
        if prefill_tokens <= 0 and decode_tokens <= 0:
            return 0.0
        return (
            self._cfg.prefill_base_us
            + self._cfg.prefill_us_per_token * prefill_tokens
            + self._cfg.decode_base_us
            + self._cfg.decode_us_per_token * decode_tokens
        )
