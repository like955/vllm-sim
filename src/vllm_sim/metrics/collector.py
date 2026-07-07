"""Metrics collector for the simulator."""

from pydantic import BaseModel, Field


class TurnMetrics(BaseModel):
    """Metrics for one turn of a session."""

    turn: int
    request_id: str
    prefill_tokens: int = 0
    prefill_blocks: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    queue_time_us: float = 0.0
    prefill_time_us: float = 0.0
    decode_time_us: float = 0.0
    total_time_us: float = 0.0


class SessionMetrics(BaseModel):
    """Per-session aggregated metrics."""

    session_id: str
    num_turns: int = 0
    turns: list[TurnMetrics] = Field(default_factory=list)

    @property
    def e2e_latency_us(self) -> float:
        """End-to-end latency: arrival of first turn -> finish of last turn."""
        if not self.turns:
            return 0.0
        return self.turns[-1].total_time_us

    @property
    def total_queue_us(self) -> float:
        return sum(t.queue_time_us for t in self.turns)

    @property
    def total_prefill_us(self) -> float:
        return sum(t.prefill_time_us for t in self.turns)

    @property
    def total_decode_us(self) -> float:
        return sum(t.decode_time_us for t in self.turns)

    @property
    def prefix_hit_rate(self) -> float:
        total = sum(t.prefix_hits + t.prefix_misses for t in self.turns)
        if total == 0:
            return 0.0
        return sum(t.prefix_hits for t in self.turns) / total


class SystemMetrics(BaseModel):
    """System-level aggregated metrics."""

    total_sessions: int = 0
    total_turns: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    total_prefix_hits: int = 0
    total_prefix_misses: int = 0
    total_sim_time_us: float = 0.0

    kv_usage_samples: list[tuple[float, float]] = Field(default_factory=list)
    """List of ``(time_us, block_usage_fraction)`` snapshots."""

    session_metrics: dict[str, SessionMetrics] = Field(default_factory=dict)

    @property
    def prefix_hit_rate(self) -> float:
        total = self.total_prefix_hits + self.total_prefix_misses
        return (self.total_prefix_hits / total) if total > 0 else 0.0

    @property
    def throughput_tokens_per_sec(self) -> float:
        total_tokens = self.total_prefill_tokens + self.total_decode_tokens
        if self.total_sim_time_us <= 0:
            return 0.0
        return total_tokens / (self.total_sim_time_us / 1_000_000)


class MetricsCollector:
    """Collects metrics during a simulation run."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionMetrics] = {}
        self.system = SystemMetrics()
        self._current_turn: dict[str, TurnMetrics] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_turn_start(
        self, session_id: str, turn: int, request_id: str,
        prefill_tokens: int, clock_us: float,
    ) -> None:
        tm = TurnMetrics(
            turn=turn,
            request_id=request_id,
            prefill_tokens=prefill_tokens,
            total_time_us=clock_us,  # temporary: store arrival time here
        )
        self._current_turn[session_id] = tm

    def record_turn_admitted(
        self, session_id: str, num_blocks: int, hits: int, misses: int,
        clock_us: float,
    ) -> None:
        tm = self._current_turn.get(session_id)
        if tm is None:
            return
        tm.queue_time_us = clock_us - tm.total_time_us  # total_time_us held arrival
        tm.prefill_blocks = num_blocks
        tm.prefix_hits = hits
        tm.prefix_misses = misses

    def record_turn_finish(self, session_id: str, clock_us: float) -> TurnMetrics | None:
        tm = self._current_turn.pop(session_id, None)
        if tm is None:
            return None
        tm.total_time_us = clock_us - tm.total_time_us  # now e2e

        sm = self.sessions.get(session_id)
        if sm is None:
            sm = SessionMetrics(session_id=session_id)
            self.sessions[session_id] = sm
        sm.turns.append(tm)
        sm.num_turns = len(sm.turns)

        self.system.total_turns += 1
        self.system.total_prefill_tokens += tm.prefill_tokens
        self.system.total_prefix_hits += tm.prefix_hits
        self.system.total_prefix_misses += tm.prefix_misses

        return tm

    def record_kv_snapshot(self, time_us: float, usage: float) -> None:
        self.system.kv_usage_samples.append((time_us, usage))

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------

    def summarize(self, sim_time_us: float) -> SystemMetrics:
        self.system.total_sim_time_us = sim_time_us
        self.system.total_sessions = len(self.sessions)
        self.system.session_metrics = dict(self.sessions)
        return self.system
