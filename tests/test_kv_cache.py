"""Tests for the KVCacheManager."""

from vllm_sim.kv_cache.block_pool import BlockPool
from vllm_sim.kv_cache.manager import KVCacheManager


class TestKVCacheManager:
    def test_allocate_simple(self) -> None:
        pool = BlockPool(num_blocks=32, block_size=16)
        mgr = KVCacheManager(pool)

        # 100 tokens → 7 blocks (ceil(100/16) = 7).
        segs = [(None, 100)]
        bids, hits, misses = mgr.allocate_request("r1", segs)
        assert bids is not None
        assert len(bids) == 7
        assert hits == 0
        assert misses == 7

    def test_allocate_with_prefix(self) -> None:
        pool = BlockPool(num_blocks=64, block_size=16)
        mgr = KVCacheManager(pool)

        # First request: system prompt (48 tokens → 3 blocks) + user (32 → 2).
        segs_a = [("sys", 48), (None, 32)]
        bids_a, h_a, m_a = mgr.allocate_request("a", segs_a)
        assert bids_a is not None
        assert h_a == 0
        assert m_a == 5  # 3 + 2

        # Second request: same system prompt (3 blocks hit).
        segs_b = [("sys", 48), (None, 64)]
        bids_b, h_b, m_b = mgr.allocate_request("b", segs_b)
        assert bids_b is not None
        assert h_b == 3  # System prompt hits.
        assert m_b == 4  # 64 tokens → 4 blocks.

    def test_oom_rollback(self) -> None:
        pool = BlockPool(num_blocks=3, block_size=16)
        mgr = KVCacheManager(pool)

        # 64 tokens → 4 blocks, but only 3 available.
        segs = [(None, 64)]
        bids, _, _ = mgr.allocate_request("r1", segs)
        assert bids is None  # Allocation failed, rolled back.

        # Pool should be unaffected.
        assert pool.num_free_blocks == 3

    def test_free_request(self) -> None:
        pool = BlockPool(num_blocks=16, block_size=16)
        mgr = KVCacheManager(pool)

        segs = [(None, 32)]
        mgr.allocate_request("r1", segs)
        assert pool.num_free_blocks == 14  # 16 - 2

        mgr.free_request("r1")
        assert pool.num_free_blocks == 16

    def test_blocks_needed(self) -> None:
        pool = BlockPool(num_blocks=16, block_size=16)
        mgr = KVCacheManager(pool)
        assert mgr.blocks_needed(0) == 0
        assert mgr.blocks_needed(1) == 1
        assert mgr.blocks_needed(16) == 1
        assert mgr.blocks_needed(17) == 2
