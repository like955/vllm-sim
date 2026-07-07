"""Tests for the scheduler."""

import pytest

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.engine.request import ContentSegment, Request
from vllm_sim.engine.scheduler import Scheduler
from vllm_sim.kv_cache.block_pool import BlockPool
from vllm_sim.kv_cache.manager import KVCacheManager


def make_request(
    req_id: str,
    session_id: str = "s1",
    turn: int = 0,
    prompt_len: int = 100,
    max_tokens: int = 20,
) -> Request:
    return Request(
        request_id=req_id,
        session_id=session_id,
        turn=turn,
        segments=[ContentSegment(content_hash=None, num_tokens=prompt_len)],
        max_tokens=max_tokens,
    )


class TestScheduler:
    def test_enqueue_and_admit(self) -> None:
        config = EngineSimConfig(num_gpu_blocks_override=64, block_size=16)
        pool = BlockPool(config.num_gpu_blocks, config.block_size)
        kv = KVCacheManager(pool)
        sched = Scheduler(config, kv)

        req = make_request("r1", prompt_len=100)  # 7 blocks
        sched.enqueue(req, clock_us=0)
        assert sched.get_waiting_count() == 1

        admitted = sched.try_admit(clock_us=0)
        assert len(admitted) == 1
        assert sched.get_waiting_count() == 0
        assert sched.get_running_count() == 1
        assert sched.get_prefilling_count() == 1
        assert req.status.name == "PREFILLING"

    def test_admission_blocked_by_kv(self) -> None:
        # 10 blocks total; pre-allocate 3 → 7 free.  Request needs 7 blocks
        # which fits in total capacity (no OOM) but exceeds free count.
        config = EngineSimConfig(num_gpu_blocks_override=10, block_size=16)
        pool = BlockPool(config.num_gpu_blocks, config.block_size)
        # Occupy 3 blocks with a dummy request.
        kv = KVCacheManager(pool)
        kv.allocate_request("dummy", [(None, 64)])  # 4 blocks → 6 free
        sched = Scheduler(config, kv)

        req = make_request("r1", prompt_len=112)  # 7 blocks, 6 free → temp shortage
        sched.enqueue(req, clock_us=0)

        admitted = sched.try_admit(clock_us=0)
        assert len(admitted) == 0  # Cannot admit — temp shortage.
        assert sched.get_waiting_count() == 1

    def test_oom_prompt_alone(self) -> None:
        # prompt 176 tok = 11 blocks > 6 pool → OOM check 1.
        config = EngineSimConfig(num_gpu_blocks_override=6, block_size=16)
        pool = BlockPool(config.num_gpu_blocks, config.block_size)
        kv = KVCacheManager(pool)
        sched = Scheduler(config, kv)

        req = make_request("r1", prompt_len=176)
        sched.enqueue(req, clock_us=0)

        with pytest.raises(MemoryError, match="OOM"):
            sched.try_admit(clock_us=0)

    def test_oom_prompt_plus_decode(self) -> None:
        # prompt 80 tok = 5 blocks, decode 32 tok = 2 blocks
        # 5+2=7 > 6 pool → OOM.
        config = EngineSimConfig(num_gpu_blocks_override=6, block_size=16)
        pool = BlockPool(config.num_gpu_blocks, config.block_size)
        kv = KVCacheManager(pool)
        sched = Scheduler(config, kv)

        req = make_request("r1", prompt_len=80, max_tokens=32)
        sched.enqueue(req, clock_us=0)

        with pytest.raises(MemoryError, match="OOM"):
            sched.try_admit(clock_us=0)

    def test_chunked_prefill(self) -> None:
        config = EngineSimConfig(
            num_gpu_blocks_override=64, block_size=16, max_num_batched_tokens=32
        )
        pool = BlockPool(config.num_gpu_blocks, config.block_size)
        kv = KVCacheManager(pool)
        sched = Scheduler(config, kv)

        req = make_request("r1", prompt_len=100)  # 100 tokens > 32, needs chunking
        sched.enqueue(req, clock_us=0)
        sched.try_admit(clock_us=0)

        # Step 1: prefill 32 tokens.
        r1 = sched.step(clock_us=0)
        assert r1.prefill_tokens_processed == 32
        assert req.num_computed_tokens == 32
        assert not req.is_prefill_complete
        assert sched.get_prefilling_count() == 1

        # Step 2: prefill 32 more.
        sched.step(clock_us=0)
        assert req.num_computed_tokens == 64

        # Step 3: prefill 32 more (96 total, 4 remaining).
        sched.step(clock_us=0)
        assert req.num_computed_tokens == 96
        assert not req.is_prefill_complete

        # Step 4: prefill the last 4 tokens.
        sched.step(clock_us=0)
        assert req.num_computed_tokens == 100
        assert req.is_prefill_complete
        assert sched.get_prefilling_count() == 0

    def test_decode_and_finish(self) -> None:
        config = EngineSimConfig(num_gpu_blocks_override=64, block_size=16)
        pool = BlockPool(config.num_gpu_blocks, config.block_size)
        kv = KVCacheManager(pool)
        sched = Scheduler(config, kv)

        req = make_request("r1", prompt_len=32, max_tokens=3)
        sched.enqueue(req, clock_us=0)
        sched.try_admit(clock_us=0)

        # Prefill (all 32 tokens in one shot).
        sched.step(clock_us=0)
        assert req.is_prefill_complete
        assert req.status.name == "DECODING"

        # Decode token-by-token (chunk_size=16 by default, but max_tokens=3).
        # Step 1: generate up to 16 tokens, finishes after 3.
        r3 = sched.step(clock_us=0)
        assert req.num_generated_tokens == 3
        assert req.is_finished
        assert req in r3.completed

    def test_chunked_prefill_with_multiple_seq(self) -> None:
        config = EngineSimConfig(
            num_gpu_blocks_override=64, block_size=16, max_num_batched_tokens=50
        )
        pool = BlockPool(config.num_gpu_blocks, config.block_size)
        kv = KVCacheManager(pool)
        sched = Scheduler(config, kv)

        # These two requests each need 4 blocks (48 tokens / 16 + 1 = 3 blocks
        # actually: ceil(48/16) = 3 blocks each, both fit).
        # But max_num_batched_tokens=50, so first step processes 50 tokens
        # (finishes req1's prefill + 2 tokens of req2), second step does rest.
        r1 = make_request("r1", prompt_len=48, max_tokens=5)
        r2 = make_request("r2", prompt_len=48, max_tokens=5)

        sched.enqueue(r1, clock_us=0)
        sched.enqueue(r2, clock_us=0)
        sched.try_admit(clock_us=0)

        assert sched.get_running_count() == 2
        assert sched.get_prefilling_count() == 2

        result = sched.step(clock_us=0)
        assert result.prefill_tokens_processed == 50  # 48 for r1 + 2 for r2
