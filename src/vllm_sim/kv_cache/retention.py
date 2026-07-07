"""Staged-free KV cache retention: step-TTL policy (Continuum-style)."""

from dataclasses import dataclass


@dataclass
class RetentionState:
    """Prefix blocks retained after a request finishes.

    The list always holds exactly the blocks still referenced (a
    contiguous prefix ``B₀ … B_{k-1}``).
    """

    session_id: str
    block_ids: list[int]
    finish_time_us: float


class StepTTLPolicy:
    """Hold all blocks for *ttl_us*, then free everything."""

    def __init__(self, ttl_us: float) -> None:
        self.ttl_us = ttl_us

    def evaluate(self, state: RetentionState, now_us: float) -> int:
        return len(state.block_ids) if (now_us - state.finish_time_us) < self.ttl_us else 0
