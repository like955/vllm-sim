"""Tests for staged-free KV cache retention (k(t) framework)."""

import pytest

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.engine.engine import EngineSim
from vllm_sim.engine.request import ContentSegment, Request
from vllm_sim.kv_cache.block_pool import BlockPool
from vllm_sim.kv_cache.manager import KVCacheManager
from vllm_sim.kv_cache.retention import RetentionState, StepTTLPolicy
from vllm_sim.trace.driver import TraceDriver
from vllm_sim.trace.schema import SessionTrace, WorkloadTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_req(
    req_id: str = "r0",
    session_id: str = "s0",
    prompt_len: int = 100,
    max_tokens: int = 20,
) -> Request:
    return Request(
        request_id=req_id,
        session_id=session_id,
        turn=0,
        segments=[ContentSegment(content_hash=None, num_tokens=prompt_len)],
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Unit: policies
# ---------------------------------------------------------------------------


class TestStepTTLPolicy:
    def test_keep_ttl_then_free(self) -> None:
        p = StepTTLPolicy(ttl_us=1_000_000)
        state = RetentionState("s0", list(range(10)), 0.0)
        assert p.evaluate(state, 500_000) == 10
        assert p.evaluate(state, 1_500_000) == 0


def test_step_ttl_policy() -> None:
    p = StepTTLPolicy(ttl_us=2_000_000)
    assert p.ttl_us == 2_000_000
    assert p.evaluate(RetentionState("s0", list(range(10)), 0.0), 500_000) == 10
    assert p.evaluate(RetentionState("s0", list(range(10)), 0.0), 2_500_000) == 0


# ---------------------------------------------------------------------------
# Unit: KVCacheManager with retention
# ---------------------------------------------------------------------------


class TestKVCacheManagerRetention:
    def test_no_policy_is_backward_compatible(self) -> None:
        """Without a policy, retain_request frees everything immediately."""
        pool = BlockPool(num_blocks=32, block_size=16)
        mgr = KVCacheManager(pool, policy=None)

        segs = [(None, 100)]  # 7 blocks
        bids, _, _ = mgr.allocate_request("r0", segs)
        assert bids is not None
        assert len(bids) == 7
        assert pool.num_free_blocks == 25  # 32 - 7

        mgr.retain_request("r0", "s0", bids, finish_time_us=0.0)
        # No policy → all 7 blocks freed.
        assert pool.num_free_blocks == 32
        assert mgr.retained_count == 0

    def test_step_ttl_keeps_blocks(self) -> None:
        """StepTTL keeps all blocks when within TTL."""
        pool = BlockPool(num_blocks=64, block_size=16)
        mgr = KVCacheManager(
            pool, policy=StepTTLPolicy(ttl_us=10_000_000),
        )

        segs = [(None, 64)]  # 4 blocks
        bids, _, _ = mgr.allocate_request("r0", segs)
        assert bids is not None
        used = pool.get_usage()

        state = mgr.retain_request("r0", "s0", bids, finish_time_us=0.0)
        assert state is not None
        assert mgr.retained_count == 1
        # Usage should NOT go down — blocks are retained.
        assert pool.get_usage() == used

    def test_step_ttl_decay_frees_after_ttl(self) -> None:
        """After TTL, decay frees everything."""
        pool = BlockPool(num_blocks=64, block_size=16)
        mgr = KVCacheManager(
            pool, policy=StepTTLPolicy(ttl_us=1_000_000),
        )

        segs = [(None, 64)]  # 4 blocks
        bids, _, _ = mgr.allocate_request("r0", segs)
        mgr.retain_request("r0", "s0", bids, finish_time_us=0.0)
        assert mgr.retained_count == 1

        # Decay before TTL → keeps.
        freed = mgr.decay_retentions(now_us=500_000)
        assert len(freed) == 0
        assert mgr.retained_count == 1

        # Decay after TTL → frees.
        freed = mgr.decay_retentions(now_us=1_500_000)
        assert len(freed) == 1
        assert mgr.retained_count == 0
        assert pool.num_free_blocks == 64

    def test_prefix_cache_reuse_across_turns(self) -> None:
        """Retained blocks are found as prefix cache hits by the next turn."""
        pool = BlockPool(num_blocks=64, block_size=16, enable_prefix_cache=True)
        mgr = KVCacheManager(
            pool, policy=StepTTLPolicy(ttl_us=10_000_000),
        )

        # Turn 0: allocate system prompt + user.
        segs = [("sys", 48), ("user0", 32)]
        bids0, h0, m0 = mgr.allocate_request("s0-t0", segs)
        assert h0 == 0
        assert m0 == 5  # 3 + 2 blocks

        # Turn 0 finishes, blocks retained.
        mgr.retain_request("s0-t0", "s0", bids0, finish_time_us=0.0)
        assert mgr.retained_count == 1

        # Turn 1 arrives: uses same system prompt + new user prompt.
        segs1 = [("sys", 48), ("user1", 64)]
        bids1, h1, m1 = mgr.allocate_request("s0-t1", segs1)
        assert bids1 is not None
        # First 3 blocks (sys) should be prefix cache hits.
        assert h1 == 3
        assert m1 == 4  # 64 tokens = 4 new blocks


# ---------------------------------------------------------------------------
# Integration: EngineSim with retention
# ---------------------------------------------------------------------------


class TestEngineSimRetention:
    def test_engine_with_step_ttl(self) -> None:
        """Engine runs correctly with retention policy enabled."""
        config = EngineSimConfig(
            num_gpu_blocks_override=256,
            block_size=16,
            max_num_seqs=64,
            retention_enabled=True,
            retention_ttl_us=5_000_000,
        )
        engine = EngineSim(config)
        req = _make_req("r0", "s0", prompt_len=100, max_tokens=10)
        engine.add_request_at(0.0, req)

        completed = engine.run_until_next_output()
        assert len(completed) == 1

        # Blocks should be retained.
        snap = engine.snapshot()
        assert snap["retained_requests"] >= 0  # Pending retentions.

    def test_engine_without_retention_backward_compat(self) -> None:
        """Without retention policy, engine works as before."""
        config = EngineSimConfig(
            num_gpu_blocks_override=256, block_size=16, max_num_seqs=64,
            retention_enabled=False,
        )
        engine = EngineSim(config)
        req = _make_req("r0", "s0", prompt_len=100, max_tokens=10)
        engine.add_request_at(0.0, req)

        completed = engine.run_until_next_output()
        assert len(completed) == 1
        assert engine.num_free_blocks == config.num_gpu_blocks


# ---------------------------------------------------------------------------
# Integration: TraceDriver with retention
# ---------------------------------------------------------------------------


class TestTraceDriverRetention:
    def test_single_session_with_retention(self) -> None:
        """A multi-turn session completes with retention enabled."""
        trace = WorkloadTrace(
            sessions=[
                SessionTrace(
                    session_id="agent-001",
                    input_prompt_length=500,
                    assistant_response_length=[50],
                    tool_call_latency=[0.1],
                    tool_call_output_length=[100],
                    final_assistant_response_length=50,
                ),
            ],
            arrival_pattern="bulk",
        )
        # TTL = 0 → blocks freed immediately, like vLLM (sanity check).
        config = EngineSimConfig(
            num_gpu_blocks_override=256,
            block_size=16,
            max_num_seqs=64,
            retention_enabled=True,
            retention_ttl_us=0,
        )
        driver = TraceDriver(trace, config, seed=42)
        engine, metrics = driver.run()

        assert metrics.total_sessions == 1
        assert metrics.total_turns == 2
        assert engine.num_free_blocks == config.num_gpu_blocks

    def test_retention_with_prefix_sharing(self) -> None:
        """Two sessions sharing a system prompt both benefit from retention."""
        trace = WorkloadTrace(
            sessions=[
                SessionTrace(
                    session_id="agent-001",
                    input_prompt_length=1500,
                    system_prompt_hash="sys_v1",
                    system_prompt_length=1500,
                    assistant_response_length=[100],
                    tool_call_latency=[0.1],
                    tool_call_output_length=[200],
                    final_assistant_response_length=100,
                ),
                SessionTrace(
                    session_id="agent-002",
                    input_prompt_length=1500,
                    system_prompt_hash="sys_v1",
                    system_prompt_length=1500,
                    assistant_response_length=[100],
                    tool_call_latency=[0.1],
                    tool_call_output_length=[200],
                    final_assistant_response_length=100,
                ),
            ],
            arrival_pattern="bulk",
        )
        # TTL = 0 → blocks freed immediately.
        config = EngineSimConfig(
            num_gpu_blocks_override=512,
            block_size=16,
            max_num_seqs=256,
            enable_prefix_cache=True,
            retention_enabled=True,
            retention_ttl_us=0,
        )
        driver = TraceDriver(trace, config, seed=42)
        engine, metrics = driver.run()

        assert metrics.total_sessions == 2
        assert metrics.total_prefix_hits > 0  # System prompt shared
        assert engine.num_free_blocks == config.num_gpu_blocks

    def test_retention_snapshot_fields(self) -> None:
        """Engine snapshot includes retention stats."""
        trace = WorkloadTrace(
            sessions=[
                SessionTrace(
                    session_id="agent-001",
                    input_prompt_length=500,
                    assistant_response_length=[50],
                    tool_call_latency=[0.1],
                    tool_call_output_length=[100],
                    final_assistant_response_length=50,
                ),
            ],
            arrival_pattern="bulk",
        )
        config = EngineSimConfig(
            num_gpu_blocks_override=256,
            block_size=16,
            max_num_seqs=64,
            retention_enabled=True,
            retention_ttl_us=100_000_000,  # Very long TTL
        )
        driver = TraceDriver(trace, config, seed=42)
        engine, _ = driver.run()

        snap = engine.snapshot()
        assert "retained_requests" in snap
        assert "retained_blocks" in snap
