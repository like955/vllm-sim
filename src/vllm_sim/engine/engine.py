"""Discrete-event simulation engine."""

import heapq

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.kv_cache.block_pool import BlockPool
from vllm_sim.kv_cache.manager import KVCacheManager
from vllm_sim.kv_cache.retention import StepTTLPolicy

from .request import Request
from .scheduler import Scheduler

_EVT_ARRIVAL = 0
_EVT_RETENTION_DECAY = 1
_NOW = 0


class EngineSim:
    """Discrete-event simulator of a vLLM inference engine."""

    def __init__(self, config: EngineSimConfig | None = None) -> None:
        self.config = config or EngineSimConfig()
        self.current_time_us: float = 0.0

        self._block_pool = BlockPool(
            num_blocks=self.config.num_gpu_blocks,
            block_size=self.config.block_size,
            enable_prefix_cache=self.config.enable_prefix_cache,
        )

        policy = None
        if self.config.retention_enabled:
            policy = StepTTLPolicy(self.config.retention_ttl_us)
        self._kv_cache = KVCacheManager(self._block_pool, policy)

        self._scheduler = Scheduler(self.config, self._kv_cache)
        self._events: list[tuple[float, int, int, Request | None]] = []
        self._event_counter = 0
        self._peak_usage = 0.0
        self._last_step_usage = 0.0

    # -- Public API --------------------------------------------------------

    def add_request_at(self, at_time_us: float, request: Request) -> None:
        self._event_counter += 1
        heapq.heappush(
            self._events,
            (at_time_us, self._event_counter, _EVT_ARRIVAL, request),
        )

    def has_pending_work(self) -> bool:
        return any(evt == _EVT_ARRIVAL for _, _, evt, _ in self._events) \
            or self._scheduler.has_work()

    def run_until_next_output(self) -> list[Request]:
        while True:
            self._drain_events()
            self._scheduler.try_admit(self.current_time_us)

            if self._scheduler.get_running_count() == 0:
                if not self._events:
                    return []
                self._jump_to_next_event()
                continue

            usage = self._block_pool.get_usage()
            self._last_step_usage = usage
            if usage > self._peak_usage:
                self._peak_usage = usage

            result = self._scheduler.step(self.current_time_us)
            self.current_time_us += result.step_time_us

            if result.completed:
                self._trigger_decay()
                return result.completed

    @property
    def num_free_blocks(self) -> int:
        return self._kv_cache.get_num_free_blocks()

    @property
    def block_pool(self) -> BlockPool:
        return self._block_pool

    @property
    def kv_cache(self) -> KVCacheManager:
        return self._kv_cache

    def snapshot(self) -> dict:
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
            "retained_requests": self._kv_cache.retained_count,
            "retained_blocks": self._kv_cache.retained_block_count,
        }

    # -- Internal ----------------------------------------------------------

    def _drain_events(self) -> None:
        while self._events and self._events[0][_NOW] <= self.current_time_us:
            _, _, evt_type, payload = heapq.heappop(self._events)
            if evt_type == _EVT_ARRIVAL:
                payload.arrival_time = self.current_time_us  # type: ignore[union-attr]
                self._scheduler.enqueue(payload, self.current_time_us)  # type: ignore[arg-type]
            elif evt_type == _EVT_RETENTION_DECAY:
                self._kv_cache.decay_retentions(self.current_time_us)
                if self._kv_cache.retained_count > 0:
                    self._schedule_decay()

    def _jump_to_next_event(self) -> None:
        if self._events:
            self.current_time_us = max(self.current_time_us, self._events[0][_NOW])

    def _schedule_decay(self) -> None:
        self._event_counter += 1
        heapq.heappush(
            self._events,
            (self.current_time_us + self.config.retention_ttl_us,
             self._event_counter, _EVT_RETENTION_DECAY, None),
        )

    def _trigger_decay(self) -> None:
        self._kv_cache.decay_retentions(self.current_time_us)
        if self._kv_cache.retained_count > 0:
            self._schedule_decay()
