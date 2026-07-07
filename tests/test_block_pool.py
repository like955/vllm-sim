"""Tests for the KV cache BlockPool."""

import pytest

from vllm_sim.kv_cache.block_pool import BlockPool


class TestBlockPool:
    def test_allocate_and_free(self) -> None:
        pool = BlockPool(num_blocks=4, block_size=16)
        assert pool.num_free_blocks == 4

        b0 = pool.allocate_block()
        assert b0 is not None
        assert pool.num_free_blocks == 3
        assert not pool.blocks[b0].is_free

        pool.free_block(b0)
        assert pool.num_free_blocks == 4
        assert pool.blocks[b0].is_free

    def test_oom(self) -> None:
        pool = BlockPool(num_blocks=2, block_size=16)
        pool.allocate_block()
        pool.allocate_block()
        assert pool.allocate_block() is None  # OOM

    def test_prefix_cache_hit_same_content(self) -> None:
        pool = BlockPool(num_blocks=8, block_size=16)

        b0, hit0 = pool.allocate_with_prefix_lookup(("sys", 0))
        assert b0 is not None
        assert not hit0  # first time -> miss

        # Same content key -> hit (block ref_count > 0).
        b1, hit1 = pool.allocate_with_prefix_lookup(("sys", 0))
        assert b1 == b0
        assert hit1
        assert pool.blocks[b0].ref_count == 2

    def test_prefix_cache_reuse_after_free(self) -> None:
        pool = BlockPool(num_blocks=8, block_size=16)

        b0, _ = pool.allocate_with_prefix_lookup(("sys", 0))
        pool.free_block(b0)

        # Same content -> should reuse the freed block.
        b1, hit = pool.allocate_with_prefix_lookup(("sys", 0))
        assert b1 == b0
        assert hit

    def test_prefix_cache_invalidation(self) -> None:
        """A prefix-cache entry is cleared when its block gets re-used
        via allocate_block() for different content."""
        pool = BlockPool(num_blocks=3, block_size=16)

        # Populate cache: ("sys", 0) on block-0.
        b0, _ = pool.allocate_with_prefix_lookup(("sys", 0))
        assert b0 == 0
        pool.free_block(b0)
        # LIFO: free_blocks = [0, 1, 2]  (block 0 prepended).

        # allocate_block() pops left -> block 0 again.
        # _invalidate_prefix_entries(0) removes ("sys", 0).
        bx = pool.allocate_block()
        assert bx == 0

        # ("sys", 0) now misses because its entry was invalidated.
        # allocate_block pops left -> block 1.
        bid, hit = pool.allocate_with_prefix_lookup(("sys", 0))
        assert not hit
        assert bid == 1  # fresh block from free pool
