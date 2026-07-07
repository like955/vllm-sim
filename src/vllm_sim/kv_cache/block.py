"""A single KV cache block."""

from dataclasses import dataclass


@dataclass
class KVCacheBlock:
    """One block in the KV cache block pool.

    Each block holds ``block_size`` tokens worth of KV cache data.
    Reference counting enables prefix cache sharing: multiple requests
    can reference the same block when their prompts share a prefix.
    """

    block_id: int
    ref_count: int = 0
    _is_free: bool = True

    @property
    def is_free(self) -> bool:
        return self._is_free

    def acquire(self) -> None:
        """Increment reference count.  Marks the block as in-use."""
        self.ref_count += 1
        self._is_free = False

    def release(self) -> int:
        """Decrement reference count.  Returns the new ref_count."""
        self.ref_count = max(0, self.ref_count - 1)
        if self.ref_count == 0:
            self._is_free = True
        return self.ref_count
