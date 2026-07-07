"""Request and content-segment data structures."""

from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    """Lifecycle states of a simulated request."""

    WAITING = auto()    # Queued, waiting for KV-cache admission.
    PREFILLING = auto() # Actively being prefilled (possibly chunked).
    DECODING = auto()   # Generating output tokens one-by-one.
    FINISHED = auto()   # Completed successfully.
    ABORTED = auto()    # Aborted before completion.


@dataclass
class ContentSegment:
    """One contiguous region of a prompt with an optional content hash.

    When ``content_hash`` is not ``None`` the segment can benefit from
    prefix caching: blocks holding the same hash are shared across
    requests.
    """

    content_hash: str | None
    num_tokens: int


@dataclass
class Request:
    """A single inference request tracked by the simulator.

    Each *turn* of a multi-turn agent session is modelled as an
    independent ``Request`` (no resumable / KV-pinning between turns).
    """

    request_id: str
    session_id: str
    turn: int

    # --- Prompt description ---
    segments: list[ContentSegment] = field(default_factory=list)
    max_tokens: int = 256

    # --- Dynamic state ---
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0   # Tokens already prefilled.
    num_generated_tokens: int = 0  # Tokens generated during decode.

    # --- KV cache ---
    block_ids: list[int] = field(default_factory=list)

    # --- Timing bookkeeping (all in us) ---
    arrival_time: float = 0.0
    enqueue_time: float = 0.0
    prefill_start_us: float | None = None
    decode_start_us: float | None = None
    finish_time_us: float | None = None

    # --- Prefix cache stats for this request ---
    prefix_hits: int = 0
    prefix_misses: int = 0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_prompt_tokens(self) -> int:
        return sum(seg.num_tokens for seg in self.segments)

    @property
    def pending_prefill_tokens(self) -> int:
        return self.total_prompt_tokens - self.num_computed_tokens

    @property
    def is_prefill_complete(self) -> bool:
        return self.num_computed_tokens >= self.total_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    @property
    def is_running(self) -> bool:
        return self.status in (RequestStatus.PREFILLING, RequestStatus.DECODING)
