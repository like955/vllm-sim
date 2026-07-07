"""Simulator configuration."""

from pydantic import BaseModel, Field, computed_field


class EngineSimConfig(BaseModel):
    """Configuration for the vLLM simulator engine.

    KV cache capacity can be specified in two ways:

    *   **Memory-based** (production): set ``total_kv_memory_gb`` and
        ``kv_mib_per_token``.  The block count is derived automatically.
    *   **Block-based** (tests): set ``num_gpu_blocks_override`` for
        exact control over the block-pool size.

    All timing values are in microseconds (us) internally.  The trace
    driver converts seconds to microseconds when injecting events.
    """

    # --- KV Cache ---
    total_kv_memory_gb: float = Field(default=16.0, ge=0)
    """Total GPU memory budget for KV cache (GiB)."""

    kv_mib_per_token: float = Field(default=0.5, ge=0)
    """MiB of KV cache consumed per token.  Default (0.5 MiB) is typical
    for a 7-8B model with FP16 KV cache."""

    block_size: int = Field(default=16, ge=1)
    """Number of tokens per KV cache block (vLLM default: 16)."""

    enable_prefix_cache: bool = True
    """Whether to enable prefix cache matching via content hash."""

    # Private override for tests.
    num_gpu_blocks_override: int | None = Field(default=None, exclude=True)

    # --- Scheduler ---
    max_num_batched_tokens: int = Field(default=8192, ge=1)
    """Maximum tokens to process in a single prefill step (chunked prefill)."""

    max_num_seqs: int = Field(default=256, ge=1)
    """Maximum number of concurrent sequences (running requests)."""

    # --- Timing model (analytical, physics-based) ---
    prefix_hit_cost_ratio: float = Field(default=0.1, ge=0, le=1)
    """Cost ratio of a prefix-cache-hit prefill token vs a miss token."""

    alpha_us: float = Field(default=8.5, ge=0)
    """GEMM cost per token (µs).  ~8.5 µs/token for Llama-8B on H100."""

    beta_us: float = Field(default=0.4, ge=0)
    """Attention IO cost per sqrt(seq_len) (µs).  ~0.4 µs for H100."""

    gamma_us: float = Field(default=500.0, ge=0)
    """Mixed-batch penalty (µs).  ~500 µs when prefill+decode coexist."""

    # --- Retention (staged-free k(t)) ---
    retention_enabled: bool = Field(default=False)
    """Enable Continuum-style TTL retention after request completion."""

    retention_ttl_us: float = Field(default=1_000_000, ge=0)
    """KV-cache time-to-live after request completion (µs)."""

    retention_priority: bool = Field(default=False)
    """Admit sessions with retained KV blocks ahead of new arrivals."""

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @computed_field
    @property
    def num_gpu_blocks(self) -> int:
        """Number of KV cache blocks (derived or overridden)."""
        if self.num_gpu_blocks_override is not None:
            return self.num_gpu_blocks_override
        bytes_per_block = self.block_size * self._bytes_per_token
        total_bytes = int(self.total_kv_memory_gb * 1024**3)
        return max(1, total_bytes // int(bytes_per_block))

    @computed_field
    @property
    def max_model_len(self) -> int:
        """Maximum sequence length = total KV capacity in tokens."""
        return self.num_gpu_blocks * self.block_size

    @property
    def _bytes_per_token(self) -> float:
        return self.kv_mib_per_token * 1024 * 1024  # noqa: ASYNC102

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_blocks(cls, num_blocks: int, **kwargs) -> "EngineSimConfig":
        """Create a config with an exact block-pool size (for tests)."""
        return cls(num_gpu_blocks_override=num_blocks, **kwargs)
