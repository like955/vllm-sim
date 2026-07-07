"""KV cache block pool with prefix cache support.

Models vLLM's ``BlockPool``: a fixed-size array of ``KVCacheBlock``
instances, a free-block queue in eviction order, and an optional
prefix-cache dictionary keyed by ``(content_hash, block_index)``.
"""

from collections import deque
from dataclasses import dataclass, field

from .block import KVCacheBlock


@dataclass
class PrefixCacheEntry:
    """An entry in the prefix cache mapping a content key to a block."""

    block_id: int
    ref_count: int = 0


class BlockPool:
    """Fixed-size pool of KV cache blocks with optional prefix caching.

    When ``enable_prefix_cache`` is True the pool maintains a dictionary
    that maps ``(content_hash, block_idx)`` → block_id so that requests
    with identical prompt prefixes can share the same physical blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_cache: bool = True,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_prefix_cache = enable_prefix_cache

        # All blocks are pre-allocated.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_blocks)
        ]

        # Free blocks in eviction order (FIFO – pop left, append right).
        self.free_blocks: deque[int] = deque(range(num_blocks))

        # Prefix cache: (content_hash, block_idx) → PrefixCacheEntry
        self._prefix_cache: dict[tuple[str, int], PrefixCacheEntry] = {}

        # Reverse map for invalidation when a block is re-used.
        self._block_to_keys: dict[int, set[tuple[str, int]]] = {
            i: set() for i in range(num_blocks)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_num_free_blocks(self) -> int:
        return self.num_free_blocks

    def allocate_block(self) -> int | None:
        """Allocate a free block.  Returns ``block_id`` or None if OOM."""
        if not self.free_blocks:
            return None
        block_id = self.free_blocks.popleft()
        self._invalidate_prefix_entries(block_id)
        self.blocks[block_id].acquire()
        return block_id

    def allocate_with_prefix_lookup(
        self, content_key: tuple[str, int] | None
    ) -> tuple[int | None, bool]:
        """Allocate a block, checking the prefix cache first.

        Returns ``(block_id, is_hit)``.  ``block_id`` is None on OOM.

        A *hit* means the content was already in the cache AND the
        target block is either (a) still allocated (shared prefix) or
        (b) free but not yet re-purposed (cache-resident reuse).
        """
        if content_key is not None and self.enable_prefix_cache:
            entry = self._prefix_cache.get(content_key)
            if entry is not None:
                block = self.blocks[entry.block_id]
                if block.ref_count > 0:
                    # Shared prefix – block is still in use.
                    block.acquire()
                    return entry.block_id, True
                if block.is_free:
                    # Cache-resident reuse – block was freed but not
                    # yet re-allocated for different content.
                    self.free_blocks.remove(entry.block_id)
                    block.acquire()
                    return entry.block_id, True
                # Stale entry – block was re-allocated for different
                # content.  Fall through to miss.
                del self._prefix_cache[content_key]

        # Miss: allocate a fresh block.
        block_id = self.allocate_block()
        if block_id is None:
            return None, False

        if content_key is not None and self.enable_prefix_cache:
            self._prefix_cache[content_key] = PrefixCacheEntry(block_id=block_id)
            self._block_to_keys[block_id].add(content_key)

        return block_id, False

    def free_block(self, block_id: int) -> None:
        """Release one reference on *block_id*.

        When the reference count reaches 0 the block is returned to
        the free pool.  Prefix-cache entries are intentionally kept so
        that future requests can still reuse the block.
        """
        block = self.blocks[block_id]
        ref = block.release()
        if ref == 0:
            # LIFO: prepend so that recently-freed (tail) blocks are evicted
            # first.  This matches vLLM's behaviour, protecting prefix-cache
            # blocks at the head of the sequence from premature eviction.
            self.free_blocks.appendleft(block_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _invalidate_prefix_entries(self, block_id: int) -> None:
        """Remove all prefix-cache entries that point to *block_id*."""
        for key in self._block_to_keys.get(block_id, set()):
            self._prefix_cache.pop(key, None)
        self._block_to_keys[block_id].clear()

    # ------------------------------------------------------------------
    # Introspection (for metrics / tests)
    # ------------------------------------------------------------------

    def get_usage(self) -> float:
        """Return fraction of blocks currently in use (0.0 – 1.0)."""
        used = self.num_blocks - self.num_free_blocks
        return used / max(self.num_blocks, 1)

    @property
    def prefix_cache_size(self) -> int:
        return len(self._prefix_cache)
