"""KV cache manager: the allocation / freeing layer above BlockPool."""

from .block_pool import BlockPool


class KVCacheManager:
    """Orchestrates block allocation and freeing for individual requests.

    Maintains a per-request ledger (``request_id`` → list of block_ids)
    so that all blocks belonging to a finished request can be released
    atomically.
    """

    def __init__(self, block_pool: BlockPool) -> None:
        self.block_pool = block_pool
        self._request_blocks: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()

    def blocks_needed(self, num_tokens: int) -> int:
        """Return how many blocks are needed to hold *num_tokens*."""
        bs = self.block_pool.block_size
        return (num_tokens + bs - 1) // bs

    def allocate_request(
        self,
        request_id: str,
        segments: list[tuple[str | None, int]],
    ) -> tuple[list[int] | None, int, int]:
        """Allocate blocks for every segment of a request's prompt.

        Args:
            request_id: Unique request identifier.
            segments: List of ``(content_hash, num_tokens)``.  A
                ``content_hash`` of ``None`` means the segment is
                unique and cannot benefit from prefix caching.

        Returns:
            ``(block_ids, num_hits, num_misses)``.  ``block_ids`` is
            ``None`` when the allocation fails (not enough free blocks).
        """
        block_ids: list[int] = []
        # Track keys we added to the prefix cache during THIS allocation
        # so that rollback only removes our own entries, leaving shared
        # (ref_count > 0) entries intact.
        added_keys: list[tuple[str, int]] = []
        num_hits = 0
        num_misses = 0
        bs = self.block_pool.block_size

        for content_hash, num_tokens in segments:
            num_seg_blocks = (num_tokens + bs - 1) // bs
            for blk_idx in range(num_seg_blocks):
                cache_key = (
                    (content_hash, blk_idx) if content_hash is not None else None
                )
                bid, is_hit = self.block_pool.allocate_with_prefix_lookup(cache_key)
                if bid is None:
                    # Roll-back: only invalidate prefix-cache entries
                    # that this partial allocation created (misses).
                    # Shared blocks (hits) remain cached for other users.
                    self._rollback_entries(added_keys)
                    self._free_blocks(block_ids)
                    return None, num_hits, num_misses
                block_ids.append(bid)
                if is_hit:
                    num_hits += 1
                else:
                    num_misses += 1
                    if cache_key is not None:
                        added_keys.append(cache_key)

        self._request_blocks[request_id] = block_ids
        return block_ids, num_hits, num_misses

    def free_request(self, request_id: str) -> None:
        """Release every block owned by *request_id*."""
        block_ids = self._request_blocks.pop(request_id, [])
        self._free_blocks(block_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _free_blocks(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self.block_pool.free_block(bid)

    def _rollback_entries(self, keys: list[tuple[str, int]]) -> None:
        """Remove prefix-cache entries created during a failed allocation."""
        for key in keys:
            self.block_pool._prefix_cache.pop(key, None)
