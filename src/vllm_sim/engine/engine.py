"""The main ``EngineSim`` — discrete-event inference engine simulator.

Usage sketch::

    config = EngineSimConfig(total_kv_memory_gb=16.0, kv_mib_per_token=0.5)
    engine = EngineSim(config)

    # Inject work from a TraceDriver or programmatically.
    engine.add_request_at(time_us=0, request=req_a)
    engine.add_request_at(time_us=500_000, request=req_b)

    # Run, yielding completed requests as they finish.
    while engine.has_pending_work():
        for finished in engine.run_until_next_output():
            print(f"{finished.request_id} done @ {engine.current_time_us} us")
"""

import heapq
from collections.abc import Iterator

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.kv_cache.block_pool import BlockPool
from vllm_sim.kv_cache.manager import KVCacheManager

from .request import Request
from .scheduler import Scheduler


# Internal event type markers stored in the priority queue.
_EVT_ARRIVAL = 0  # (time, _EVT_ARRIVAL, request)

# Sentinel for an empty event queue.
_NOW = 0  # index into heap tuple


class EngineSim:
    """Discrete-event simulator of a vLLM inference engine.

    The engine owns the simulation clock and drives the scheduler /
    KV cache step-by-step.  External code (e.g. ``TraceDriver``)
    injects requests via ``add_request_at`` and drains completed
    requests via ``run_until_next_output``.
    """

    def __init__(self, config: EngineSimConfig | None = None) -> None:
        self.config = config or EngineSimConfig()

        # Clock (microseconds).
        self.current_time_us: float = 0.0

        # KV cache.
        self._block_pool = BlockPool(
            num_blocks=self.config.num_gpu_blocks,
            block_size=self.config.block_size,
            enable_prefix_cache=self.config.enable_prefix_cache,
        )
        self._kv_cache = KVCacheManager(self._block_pool)

        # Scheduler.
        self._scheduler = Scheduler(self.config, self._kv_cache)

        # Future-event priority queue: list of (time_us, counter, evt_type, payload).
        self._events: list[tuple[float, int, int, Request]] = []
        self._event_counter: int = 0

        # Track peak KV-cache usage observed across all steps.
        self._peak_usage: float = 0.0
        # Track the most recent pre-step usage (before any freeing).
        # Used by CONCUR to read accurate KV-cache utilization.
        self._last_step_usage: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request_at(self, at_time_us: float, request: Request) -> None:
        """Schedule *request* to arrive at *at_time_us* (simulation time).

        The request will be enqueued into the scheduler when the
        simulation clock reaches (or passes) *at_time_us*.
        """
        self._event_counter += 1
        heapq.heappush(
            self._events,
            (at_time_us, self._event_counter, _EVT_ARRIVAL, request),
        )

    def has_pending_work(self) -> bool:
        """True when there are future events or in-flight requests."""
        return bool(self._events) or self._scheduler.has_work()

    def run_until_next_output(self) -> list[Request]:
        """Advance simulation until at least one request finishes.

        Returns the list of requests that completed during this call.
        An empty list is returned only when there is truly no more
        work to do (``has_pending_work() == False``).
        """
        while True:
            # 1) Process all arrivals at-or-before current time.
            self._drain_arrivals()

            # 2) Try to admit waiting requests.
            self._scheduler.try_admit(self.current_time_us)

            # 3) If nothing is running, jump to the next arrival.
            if self._scheduler.get_running_count() == 0:
                if not self._events:
                    return []  # No more work at all.
                self._jump_to_next_event()
                continue

            # 4) Capture usage *before* the step (post-admit, pre-free).
            usage = self._block_pool.get_usage()
            self._last_step_usage = usage
            if usage > self._peak_usage:
                self._peak_usage = usage

            # 5) Execute one step.
            result = self._scheduler.step(self.current_time_us)
            self.current_time_us += result.step_time_us

            # 6) If any request finished, return them.
            if result.completed:
                return result.completed

    @property
    def num_free_blocks(self) -> int:
        return self._kv_cache.get_num_free_blocks()

    @property
    def block_pool(self) -> BlockPool:
        return self._block_pool

    def snapshot(self) -> dict:
        """Return a point-in-time snapshot of engine state (for metrics)."""
        return {
            "time_us": self.current_time_us,
            "free_blocks": self._kv_cache.get_num_free_blocks(),
            "total_blocks": self.config.num_gpu_blocks,
            "block_usage": self._block_pool.get_usage(),
            "prefix_cache_entries": self._block_pool.prefix_cache_size,
            "peak_usage": self._peak_usage,
            "running": self._scheduler.get_running_count(),
            "prefilling": self._scheduler.get_prefilling_count(),
            "waiting": self._scheduler.get_waiting_count(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain_arrivals(self) -> None:
        """Enqueue every request whose arrival time has passed."""
        while self._events and self._events[0][_NOW] <= self.current_time_us:
            _, _, _, request = heapq.heappop(self._events)
            request.arrival_time = self.current_time_us
            self._scheduler.enqueue(request, self.current_time_us)

    def _jump_to_next_event(self) -> None:
        """Advance the clock to the next future event's time."""
        if self._events:
            self.current_time_us = max(self.current_time_us, self._events[0][_NOW])
