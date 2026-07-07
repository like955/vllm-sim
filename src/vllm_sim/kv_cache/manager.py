"""KV cache manager: the allocation / freeing layer above BlockPool."""

from .block_pool import BlockPool
from .retention import RetentionState, StepTTLPolicy


class KVCacheManager:
    """Orchestrates block allocation and freeing for individual requests."""

    def __init__(
        self,
        block_pool: BlockPool,
        policy: StepTTLPolicy | None = None,
    ) -> None:
        self.block_pool = block_pool
        self._request_blocks: dict[str, list[int]] = {}
        self._policy = policy
        self._retained: dict[str, RetentionState] = {}

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
    # Retention
    # ------------------------------------------------------------------

    def retain_request(
        self,
        request_id: str,
        session_id: str,
        block_ids: list[int],
        finish_time_us: float,
    ) -> RetentionState | None:
        """Finish *request_id* — keep blocks pinned if TTL > 0."""
        self._request_blocks.pop(request_id, None)

        # Free previous retention for this session (blocks were already
        # reused by the new request via prefix-cache hits).
        old = self._retained.pop(session_id, None)
        if old is not None:
            self._free_blocks(old.block_ids)

        if self._policy is None:
            self._free_blocks(block_ids)
            return None

        k = self._policy.evaluate(
            RetentionState(session_id, list(block_ids), finish_time_us),
            finish_time_us,
        )
        if k == 0:
            self._free_blocks(block_ids)
            return None

        # k == len(block_ids): keep everything.
        state = RetentionState(session_id, block_ids, finish_time_us)
        self._retained[session_id] = state
        return state

    def decay_retentions(self, now_us: float) -> list[str]:
        """Re-evaluate TTL for every retained session.

        Returns *session_ids* that were fully freed.
        """
        if self._policy is None:
            return []

        freed: list[str] = []
        for sid, state in list(self._retained.items()):
            k = self._policy.evaluate(state, now_us)
            if k == 0:
                self._free_blocks(state.block_ids)
                freed.append(sid)

        for sid in freed:
            del self._retained[sid]
        return freed

    def has_retained(self, session_id: str) -> bool:
        return session_id in self._retained

    @property
    def retained_count(self) -> int:
        """Number of requests currently in retention."""
        return len(self._retained)

    @property
    def retained_block_count(self) -> int:
        """Total blocks held across all retained requests."""
        return sum(len(s.block_ids) for s in self._retained.values())

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
