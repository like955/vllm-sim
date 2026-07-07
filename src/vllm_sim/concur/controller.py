"""CONCUR agent-level admission controller.

CONCUR wraps an ``EngineSim`` as a transparent proxy, intercepting
``add_request_at`` and ``run_until_next_output`` to apply agent-level
congestion control.  This means the ``TraceDriver`` needs **zero** changes
to use CONCUR — it just receives a ``ConcurEngine`` instead of a raw
``EngineSim``.

Control law (Equation 1 from the paper)::

    W_{t+1} = W_t + α          if U_t < U_low
              W_t × β          if U_t > U_high ∧ H_t < H_thresh
              W_t              otherwise

where ``W`` is the *congestion window* (max concurrently active agents).
"""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from vllm_sim.engine.engine import EngineSim
    from vllm_sim.engine.request import Request


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConcurConfig(BaseModel):
    """Configuration for the CONCUR admission controller.

    All parameters follow the paper's defaults and are held constant
    across models, workloads, and serving engines.
    """

    # --- AIMD parameters ---
    alpha: int = Field(default=2, ge=1)
    """Additive increase step (number of agents)."""

    beta: float = Field(default=0.5, gt=0, lt=1)
    """Multiplicative decrease factor."""

    # --- Thresholds ---
    U_low: float = Field(default=0.2, ge=0, le=1)
    """KV-cache usage below which the controller probes upward."""

    U_high: float = Field(default=0.5, ge=0, le=1)
    """KV-cache usage above which thrashing is suspected."""

    H_thresh: float = Field(default=0.2, ge=0, le=1)
    """Cache hit rate below which the system is considered thrashing."""

    # --- Window bounds ---
    initial_window: int = Field(default=2, ge=1)
    """Starting congestion window (number of agents)."""

    min_window: int = Field(default=1, ge=1)
    """Floor for the congestion window."""

    max_window: int | None = Field(default=None)
    """Ceiling for the congestion window.  ``None`` means no explicit cap
    (the engine's ``max_num_seqs`` is the effective limit)."""

    # --- Update cadence ---
    window_update_interval_us: float = Field(default=100_000, ge=1)
    """Minimum sim-time between window-size re-evaluations (µs)."""

    enabled: bool = True
    """When ``False`` all requests are admitted immediately (bypass)."""

    verbose: bool = False
    """When ``True``, log every window-size change to an internal buffer."""


# ---------------------------------------------------------------------------
# Pending entry
# ---------------------------------------------------------------------------


@dataclass
class _PendingEntry:
    """A request held in the CONCUR pending queue."""

    session_id: str
    request: "Request"
    ready_time_us: float  # When the agent was ready (tool wait ended).


# ---------------------------------------------------------------------------
# ConcurEngine — transparent proxy around EngineSim
# ---------------------------------------------------------------------------


class ConcurEngine:
    """Wraps an ``EngineSim`` with agent-level admission control.

    This is a **drop-in replacement** for ``EngineSim``.  It exposes
    the same interface (``add_request_at``, ``run_until_next_output``,
    ``has_pending_work``, ``snapshot``, etc.) so that the ``TraceDriver``
    needs no changes at all.

    Usage::

        raw = EngineSim(config)
        engine = ConcurEngine(raw, ConcurConfig(initial_window=4))
        # ... use engine exactly like EngineSim ...
    """

    def __init__(
        self,
        engine: "EngineSim",
        config: ConcurConfig | None = None,
    ) -> None:
        self._engine = engine
        self._cfg = config or ConcurConfig()

        # If no explicit max_window, respect the engine's max_seqs.
        if self._cfg.max_window is None:
            self._cfg.max_window = engine.config.max_num_seqs

        # --- Congestion window ---
        self.W: int = self._cfg.initial_window
        self._max_window: int = self._cfg.max_window or 10_000

        # --- Agent tracking ---
        self._active_agents: set[str] = set()
        # Two-tier priority: continuing agents (turn > 0) before new arrivals.
        self._pending_cont: deque[_PendingEntry] = deque()
        self._pending_new: deque[_PendingEntry] = deque()

        # --- Hit-rate window ---
        self._recent_hits: int = 0
        self._recent_misses: int = 0

        # --- Window update state ---
        self._last_W_update_us: float = 0.0

        # --- History ---
        self._window_history: list[tuple[float, int, float, float]] = []
        self._change_log: list[str] = []

    # ------------------------------------------------------------------
    # EngineSim-compatible public API
    # ------------------------------------------------------------------

    def add_request_at(self, at_time_us: float, request: "Request") -> None:
        """Schedule *request* for admission (may be paused by CONCUR)."""
        session_id = request.session_id

        if not self._cfg.enabled:
            self._engine.add_request_at(at_time_us, request)
            return

        # An agent can only have one in-flight request at a time.
        if session_id in self._active_agents:
            self._enqueue_pending(session_id, request, at_time_us)
            return

        # Check congestion window.
        if len(self._active_agents) < self.W:
            self._active_agents.add(session_id)
            self._engine.add_request_at(at_time_us, request)
            return

        # Window is full — pause the agent.
        self._enqueue_pending(session_id, request, at_time_us)

    def has_pending_work(self) -> bool:
        """True when the engine or CONCUR has unfinished work."""
        return self._engine.has_pending_work() or self._has_pending()

    def run_until_next_output(self) -> list["Request"]:
        """Advance simulation until at least one request finishes.

        Pending agents are resumed at the **start** of this call (deferred
        from the previous completion) so that the driver has a chance to
        schedule a completing agent's next turn first — preserving
        execution continuity and KV-cache locality.
        """
        # --- Phase 1: resume pending agents (deferred from last completion) ---
        self._try_resume_pending()

        # --- Phase 2: feed the engine if it's still idle ---
        while not self._engine.has_pending_work() and self._has_pending():
            self._try_resume_pending()

        if not self._engine.has_pending_work():
            return []

        # --- Phase 3: run the engine ---
        completed = self._engine.run_until_next_output()

        # --- Phase 4: cleanup completed agents & update window ---
        # IMPORTANT: do NOT resume pending here.  The driver will
        # schedule the completing agent's next turn between this
        # return and the next call, preserving cache continuity.
        if completed:
            self._last_step_usage = getattr(
                self._engine, "_last_step_usage", 0.0,
            )
            self._cleanup_completed(completed)

        return completed

    # --- Direct delegation (read-only properties used by TraceDriver) ---

    @property
    def current_time_us(self) -> float:
        return self._engine.current_time_us

    @property
    def config(self):
        return self._engine.config

    @property
    def num_free_blocks(self) -> int:
        return self._engine.num_free_blocks

    @property
    def block_pool(self):
        return self._engine.block_pool

    def snapshot(self) -> dict:
        """Return engine snapshot with CONCUR state appended."""
        snap = self._engine.snapshot()
        snap["concur"] = {
            "W": self.W,
            "active_agents": len(self._active_agents),
            "pending": len(self._pending_cont) + len(self._pending_new),
            "window_history": list(self._window_history[-20:]),
            "change_log": list(self._change_log),
        }
        return snap

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _has_pending(self) -> bool:
        return bool(self._pending_cont or self._pending_new)

    def _enqueue_pending(
        self, session_id: str, request: "Request", at_time_us: float,
    ) -> None:
        """Route a paused request to the appropriate priority queue."""
        entry = _PendingEntry(session_id, request, at_time_us)
        if request.turn > 0:
            self._pending_cont.append(entry)
        else:
            self._pending_new.append(entry)

    def _try_resume_pending(self) -> None:
        """Admit paused agents into the engine (continuing agents first)."""
        now = self._engine.current_time_us

        # Tier 1: continuing agents (turn > 0) — preserve execution continuity.
        while self._pending_cont and len(self._active_agents) < self.W:
            entry = self._pending_cont.popleft()
            if entry.session_id in self._active_agents:
                continue
            self._active_agents.add(entry.session_id)
            self._engine.add_request_at(now, entry.request)

        # Tier 2: new agents.
        while self._pending_new and len(self._active_agents) < self.W:
            entry = self._pending_new.popleft()
            if entry.session_id in self._active_agents:
                continue
            self._active_agents.add(entry.session_id)
            self._engine.add_request_at(now, entry.request)

    def _cleanup_completed(self, completed: list["Request"]) -> None:
        """Remove completed agents and update window (does NOT resume pending).

        Pending resumption is deferred to the *next* ``run_until_next_output``
        call so that the driver can schedule the completing agent's next turn
        before other paused agents steal the slot — preserving execution
        continuity and KV-cache locality.
        """
        for req in completed:
            self._active_agents.discard(req.session_id)
            self._recent_hits += req.prefix_hits
            self._recent_misses += req.prefix_misses

        now = self._engine.current_time_us
        if now - self._last_W_update_us >= self._cfg.window_update_interval_us:
            self._update_window()
            self._last_W_update_us = now
            self._recent_hits = 0
            self._recent_misses = 0

    def _update_window(self) -> None:
        """Evaluate the AIMD control law and adjust ``W``."""
        U = getattr(self, "_last_step_usage", 0.0)
        total = self._recent_hits + self._recent_misses
        H = self._recent_hits / total if total > 0 else 1.0
        old_W = self.W

        if U < self._cfg.U_low:
            self.W = min(self.W + self._cfg.alpha, self._max_window)
        elif U > self._cfg.U_high and H < self._cfg.H_thresh:
            self.W = max(int(self.W * self._cfg.beta), self._cfg.min_window)

        self._window_history.append((self._engine.current_time_us, self.W, U, H))

        if self._cfg.verbose and old_W != self.W:
            direction = "↑" if self.W > old_W else "↓"
            t = self._engine.current_time_us / 1_000_000
            self._change_log.append(
                f"[CONCUR] t={t:.3f}s W {old_W}→{self.W} {direction}  "
                f"U={U:.2f} H={H:.2f}"
            )
