"""TraceDriver: translates a ``WorkloadTrace`` into EngineSim events and
reacts to completed turns by scheduling the next turn for each session.
"""

import json
import sys
import time as _time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vllm_sim.concur.controller import ConcurConfig, ConcurEngine
from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.engine.engine import EngineSim
from vllm_sim.engine.request import Request
from vllm_sim.metrics.collector import MetricsCollector, SystemMetrics

from .schema import SessionTrace, WorkloadTrace
from .session import SessionManager

# Signature: (completed_sessions, total_sessions, completed_turns, sim_time_s)
ProgressCallback = Callable[[int, int, int, float], None]


def _default_progress(
    completed_sessions: int, total_sessions: int,
    completed_turns: int, sim_time_s: float,
) -> None:
    """Print a single-line progress update to stderr every call."""
    pct = completed_sessions * 100 / max(total_sessions, 1)
    print(
        f"\r  sessions {completed_sessions}/{total_sessions} ({pct:.0f}%)  "
        f"turns {completed_turns}  sim {sim_time_s:.1f}s",
        end="", file=sys.stderr, flush=True,
    )


class TraceDriver:
    """Drives an ``EngineSim`` from a workload trace.

    1. Reads (or receives) a ``WorkloadTrace``.
    2. Injects the first turn of every session at the appropriate arrival time.
    3. When a turn completes, schedules the next turn after the tool-call delay.
    4. Collects metrics throughout.

    Set ``progress_callback`` to receive periodic updates during the run.

    Pass ``concur_config`` to wrap the engine with CONCUR agent-level
    admission control — the driver itself needs **no other changes**.
    """

    def __init__(
        self,
        trace: WorkloadTrace,
        config: EngineSimConfig | None = None,
        seed: int = 42,
        progress_callback: ProgressCallback | None = None,
        concur_config: ConcurConfig | None = None,
    ) -> None:
        self._trace = trace
        self._config = config or EngineSimConfig()
        self._rng = np.random.default_rng(seed)
        self._session_mgr = SessionManager(trace.sessions)

        # Create the raw engine, then optionally wrap with CONCUR.
        _raw = EngineSim(self._config)
        if concur_config is not None and concur_config.enabled:
            if concur_config.max_window is None:
                concur_config.max_window = self._config.max_num_seqs
            self._engine = ConcurEngine(_raw, concur_config)
        else:
            self._engine = _raw

        self._metrics = MetricsCollector()
        self._progress = progress_callback
        self._total_sessions = len(trace.sessions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> tuple[EngineSim, SystemMetrics]:
        """Run the full simulation and return ``(engine, metrics)``."""
        t0 = _time.perf_counter()

        # --- Phase 1: inject initial requests ---
        arrivals = self._session_mgr.build_initial_arrivals(
            pattern=self._trace.arrival_pattern,
            rate_per_sec=self._trace.arrival_rate_per_sec,
            rng=self._rng,
        )
        for session_id, arr_s in arrivals.items():
            turn = 0
            req = self._make_request(session_id, turn)
            at_us = arr_s * 1_000_000
            self._engine.add_request_at(at_us, req)
            self._metrics.record_turn_start(
                session_id, turn, req.request_id,
                req.total_prompt_tokens, at_us,
            )

        # --- Phase 2: event loop ---
        snapshot_interval_us = 1_000_000  # Record KV snapshot every 1ms sim time.
        next_snapshot_us = snapshot_interval_us
        completed_sessions = 0
        completed_turns = 0
        last_progress_wall_s = _time.perf_counter()

        while self._engine.has_pending_work():
            completed = self._engine.run_until_next_output()

            for req in completed:
                sid = req.session_id
                turn = req.turn

                # Record admission stats.
                self._metrics.record_turn_admitted(
                    sid,
                    num_blocks=len(req.block_ids),
                    hits=req.prefix_hits,
                    misses=req.prefix_misses,
                    clock_us=self._engine.current_time_us,
                )
                # Record finish.
                self._metrics.record_turn_finish(
                    sid, self._engine.current_time_us,
                )

                # Advance session state.
                self._session_mgr.advance_turn(sid)
                completed_turns += 1

                if self._session_mgr.is_complete(sid):
                    completed_sessions += 1

                # Progress callback (throttled to ~4 Hz).
                now = _time.perf_counter()
                if self._progress and now - last_progress_wall_s > 0.25:
                    self._progress(
                        completed_sessions, self._total_sessions,
                        completed_turns,
                        self._engine.current_time_us / 1_000_000,
                    )
                    last_progress_wall_s = now

                # Schedule next turn (if any).
                if not self._session_mgr.is_complete(sid):
                    next_turn = self._session_mgr.current_turn(sid)
                    tool_wait_s = self._session_mgr.tool_wait_s(sid, turn)
                    next_arrival_us = (
                        self._engine.current_time_us + tool_wait_s * 1_000_000
                    )
                    next_req = self._make_request(sid, next_turn)
                    self._engine.add_request_at(next_arrival_us, next_req)
                    self._metrics.record_turn_start(
                        sid, next_turn, next_req.request_id,
                        next_req.total_prompt_tokens, next_arrival_us,
                    )

            # Periodic KV-usage snapshot.
            if self._engine.current_time_us >= next_snapshot_us:
                snap = self._engine.snapshot()
                self._metrics.record_kv_snapshot(
                    snap["time_us"], snap["block_usage"],
                )
                next_snapshot_us = self._engine.current_time_us + snapshot_interval_us

        # --- Finalize ---
        sys_metrics = self._metrics.summarize(self._engine.current_time_us)
        sys_metrics.total_decode_tokens = sum(
            s.final_assistant_response_length + sum(s.assistant_response_length)
            for s in self._trace.sessions
        )
        elapsed = _time.perf_counter() - t0
        print(
            f"Simulation complete: {sys_metrics.total_sessions} sessions, "
            f"{sys_metrics.total_turns} turns, "
            f"{self._engine.current_time_us / 1_000_000:.3f}s sim time, "
            f"{elapsed:.3f}s wall time"
        )

        return self._engine, sys_metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_request(self, session_id: str, turn: int) -> Request:
        segments, max_tokens = self._session_mgr.make_request_params(session_id, turn)
        request_id = self._session_mgr.make_turn_request_id(session_id, turn)
        return Request(
            request_id=request_id,
            session_id=session_id,
            turn=turn,
            segments=segments,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Factory: load from JSON
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        config: EngineSimConfig | None = None,
        seed: int = 42,
        concur_config: ConcurConfig | None = None,
    ) -> "TraceDriver":
        """Load a trace from a JSON file and return a ready-to-run driver.

        The JSON file should contain an object with a ``"sessions"`` key
        (list of ``SessionTrace``-compatible dicts).
        """
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        sessions = [
            SessionTrace(
                session_id=s.get("session_id", f"session-{i:04d}"),
                input_prompt_length=s["input_prompt_length"],
                assistant_response_length=s.get("assistant_response_length", []),
                tool_call_latency=s.get("tool_call_latency", []),
                tool_call_output_length=s.get("tool_call_output_length", []),
                final_assistant_response_length=s.get("final_assistant_response_length", 0),
                system_prompt_hash=s.get("system_prompt_hash"),
                system_prompt_length=s.get("system_prompt_length", 0),
                arrival_time_s=s.get("arrival_time_s", 0.0),
            )
            for i, s in enumerate(raw["sessions"])
        ]

        trace = WorkloadTrace(
            sessions=sessions,
            arrival_pattern=raw.get("arrival_pattern", "poisson"),
            arrival_rate_per_sec=raw.get("arrival_rate_per_sec", 0.0),
            description=raw.get("description", ""),
        )
        return cls(trace, config, seed, concur_config=concur_config)
