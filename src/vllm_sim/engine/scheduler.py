"""FCFS scheduler with chunked prefill and KV-cache admission control."""

from collections import deque
from dataclasses import dataclass, field

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.kv_cache.manager import KVCacheManager
from vllm_sim.timing.model import TimingModel

from .request import Request, RequestStatus


@dataclass
class StepResult:
    """Result of one scheduler step."""

    step_time_us: float
    completed: list[Request] = field(default_factory=list)
    prefill_tokens_processed: int = 0
    decode_tokens_generated: int = 0


class Scheduler:
    """FCFS scheduler with chunked-prefill and admission control."""

    def __init__(self, config: EngineSimConfig, kv_cache: KVCacheManager) -> None:
        self._cfg = config
        self._kv_cache = kv_cache
        self._timing = TimingModel(config)

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._prefilling: list[Request] = []
        self._total_blocks = config.num_gpu_blocks
        self._block_size = config.block_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, request: Request, clock_us: float) -> None:
        """Add a request to the waiting queue."""
        request.status = RequestStatus.WAITING
        request.enqueue_time = clock_us
        self.waiting.append(request)

    def try_admit(self, clock_us: float) -> list[Request]:
        """Admit waiting requests until KV cache or max_seqs is exhausted.

        Two OOM checks (both raise ``MemoryError``):

        1. Prompt alone exceeds pool → can never prefill.
        2. Prompt + max_tokens exceeds pool → would OOM during decode.
        """
        admitted: list[Request] = []

        while self.waiting and len(self.running) < self._cfg.max_num_seqs:
            req = self.waiting[0]

            # --- OOM check: actual blocks from per-segment rounding ---
            # Each ContentSegment's tokens are independently rounded up
            # to block_size, so the true block count is higher than
            # ceil(total_tokens/block_size).
            prompt_blocks = sum(
                self._kv_cache.blocks_needed(s.num_tokens)
                for s in req.segments
            )
            decode_blocks = self._kv_cache.blocks_needed(req.max_tokens)
            if prompt_blocks + decode_blocks > self._total_blocks:
                raise MemoryError(
                    f"OOM: request {req.request_id} "
                    f"({req.total_prompt_tokens} prompt + {req.max_tokens} decode "
                    f"= {prompt_blocks} + {decode_blocks} = "
                    f"{prompt_blocks + decode_blocks} blocks "
                    f"> {self._total_blocks} pool capacity)."
                )

            # --- Try allocation (prefix-cache hits may reduce demand) ---
            segments = [(s.content_hash, s.num_tokens) for s in req.segments]
            block_ids, hits, misses = self._kv_cache.allocate_request(
                req.request_id, segments
            )
            if block_ids is None:
                break  # Temporary shortage — retry later.

            self.waiting.popleft()
            req.block_ids = block_ids
            req.prefix_hits = hits
            req.prefix_misses = misses
            req.status = RequestStatus.PREFILLING
            req.prefill_start_us = clock_us
            self.running.append(req)
            self._prefilling.append(req)
            admitted.append(req)

        return admitted

    def step(self, clock_us: float) -> StepResult:
        if self._prefilling:
            return self._step_prefill(clock_us)
        if self.running:
            return self._step_decode(clock_us)
        return StepResult(step_time_us=0.0)

    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    def get_running_count(self) -> int:
        return len(self.running)

    def get_waiting_count(self) -> int:
        return len(self.waiting)

    def get_prefilling_count(self) -> int:
        return len(self._prefilling)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step_prefill(self, clock_us: float) -> StepResult:
        budget = self._cfg.max_num_batched_tokens
        processed = 0
        completed: list[Request] = []

        still_prefilling: list[Request] = []
        for req in self._prefilling:
            if budget <= 0:
                still_prefilling.append(req)
                continue
            take = min(req.pending_prefill_tokens, budget)
            req.num_computed_tokens += take
            budget -= take
            processed += take
            if req.is_prefill_complete:
                req.status = RequestStatus.DECODING
                req.decode_start_us = clock_us
            else:
                still_prefilling.append(req)

        self._prefilling = still_prefilling
        step_us = self._timing.prefill_us(processed)
        return StepResult(
            step_time_us=step_us,
            completed=completed,
            prefill_tokens_processed=processed,
        )

    def _step_decode(self, clock_us: float) -> StepResult:
        # Find the smallest remaining-token gap among decoding requests.
        min_remaining: int | None = None
        for req in self.running:
            if req.status != RequestStatus.DECODING:
                continue
            rem = req.max_tokens - req.num_generated_tokens
            if rem <= 0:
                continue
            if min_remaining is None or rem < min_remaining:
                min_remaining = rem

        if min_remaining is None:
            return StepResult(step_time_us=0.0)

        completed: list[Request] = []
        total_tokens = 0
        for req in self.running:
            if req.status != RequestStatus.DECODING:
                continue
            rem = req.max_tokens - req.num_generated_tokens
            if rem <= 0:
                continue
            take = min(rem, min_remaining)
            req.num_generated_tokens += take
            total_tokens += take
            if req.num_generated_tokens >= req.max_tokens:
                req.status = RequestStatus.FINISHED
                req.finish_time_us = clock_us
                completed.append(req)

        finished_ids = {r.request_id for r in completed}
        self.running = [r for r in self.running if r.request_id not in finished_ids]

        for req in completed:
            self._kv_cache.free_request(req.request_id)

        step_us = self._timing.decode_us(total_tokens)
        return StepResult(
            step_time_us=step_us,
            completed=completed,
            decode_tokens_generated=total_tokens,
        )
