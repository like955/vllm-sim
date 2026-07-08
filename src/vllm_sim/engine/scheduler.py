"""FCFS scheduler with unified prefill+decode steps (vLLM-style)."""

import math
from collections import deque
from dataclasses import dataclass, field

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.kv_cache.manager import KVCacheManager
from vllm_sim.timing.model import make_timing

from .request import Request, RequestStatus


@dataclass
class StepResult:
    """Result of one scheduler step."""

    step_time_us: float
    completed: list[Request] = field(default_factory=list)
    prefill_tokens_processed: int = 0
    decode_tokens_generated: int = 0


class Scheduler:
    """FCFS scheduler with unified prefill + decode steps.

    There is no separate prefill or decode phase.  Every step takes a
    token budget (``max_num_batched_tokens``) and distributes it across
    *all* running requests — both those still prefilling and those
    generating output.  This mirrors vLLM's scheduler design.
    """

    def __init__(self, config: EngineSimConfig, kv_cache: KVCacheManager) -> None:
        self._cfg = config
        self._kv_cache = kv_cache
        self._timing = make_timing(config)

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._total_blocks = config.num_gpu_blocks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, request: Request, clock_us: float) -> None:
        request.status = RequestStatus.WAITING
        request.enqueue_time = clock_us
        self.waiting.append(request)

    def try_admit(self, clock_us: float) -> list[Request]:
        """Admit waiting requests until KV cache or max_seqs is exhausted."""
        admitted: list[Request] = []

        while self.waiting and len(self.running) < self._cfg.max_num_seqs:
            req = self.waiting[0]

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

            segments = [(s.content_hash, s.num_tokens) for s in req.segments]
            block_ids, hits, misses = self._kv_cache.allocate_request(
                req.request_id, segments
            )
            if block_ids is None:
                break

            self.waiting.popleft()
            req.block_ids = block_ids
            req.prefix_hits = hits
            req.prefix_misses = misses
            req.status = RequestStatus.PREFILLING
            req.prefill_start_us = clock_us
            self.running.append(req)
            admitted.append(req)

        return admitted

    def step(self, clock_us: float) -> StepResult:
        """One unified forward pass — prefill and decode compete for budget."""
        budget = self._cfg.max_num_batched_tokens
        p_hit = 0
        p_miss = 0
        d_reqs = 0
        d_mult = 0
        d_tokens = 0
        completed: list[Request] = []

        for req in self.running:
            if budget <= 0:
                break

            if not req.is_prefill_complete:
                take = min(req.pending_prefill_tokens, budget)
                # Split prefill tokens by hit/miss ratio from allocation.
                total_blk = req.prefix_hits + req.prefix_misses
                hit_frac = req.prefix_hits / total_blk if total_blk > 0 else 0.0
                p_hit += int(take * hit_frac)
                p_miss += take - int(take * hit_frac)
                req.num_computed_tokens += take
                budget -= take
                if req.is_prefill_complete:
                    req.status = RequestStatus.DECODING
                    req.decode_start_us = clock_us
            else:
                take = min(req.max_tokens - req.num_generated_tokens, budget)
                req.num_generated_tokens += take
                budget -= take
                d_tokens += take
                d_reqs += 1
                if take > d_mult:
                    d_mult = take
                if req.num_generated_tokens >= req.max_tokens:
                    req.status = RequestStatus.FINISHED
                    req.finish_time_us = clock_us
                    completed.append(req)

        finished_ids = {r.request_id for r in completed}
        self.running = [r for r in self.running if r.request_id not in finished_ids]

        for req in completed:
            self._kv_cache.free_request(req.request_id)

        p_total = p_hit + p_miss
        # Attention IO factor: Σ sqrt(seq_len) for requests in this batch.
        seq_len_sum_sqrt = sum(
            math.sqrt(r.total_prompt_tokens + r.num_generated_tokens)
            for r in self.running
        )
        is_mixed = p_total > 0 and d_reqs > 0
        step_us = self._timing.step_us(p_hit, p_miss, d_reqs, d_mult,
                                       seq_len_sum_sqrt, is_mixed)
        return StepResult(
            step_time_us=step_us,
            completed=completed,
            prefill_tokens_processed=p_total,
            decode_tokens_generated=d_tokens,
        )

    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    def get_running_count(self) -> int:
        return len(self.running)

    def get_waiting_count(self) -> int:
        return len(self.waiting)

    def get_prefilling_count(self) -> int:
        return sum(1 for r in self.running if not r.is_prefill_complete)
