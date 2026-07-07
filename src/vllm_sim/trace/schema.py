"""Trace data structures matching the real multi-turn agent workload format."""

from pydantic import BaseModel, Field


class SessionTrace(BaseModel):
    """One multi-turn agent session trace.

    The arrays ``assistant_response_length``, ``tool_call_latency``,
    and ``tool_call_output_length`` all have the same length, equal to
    the number of "tool-calling" turns.  After the last tool turn
    there is one final turn whose output length is given by
    ``final_assistant_response_length``.
    """

    session_id: str
    """Unique identifier for this session."""

    # --- Required fields (matching the real trace format) ---
    input_prompt_length: int = Field(ge=1)
    """Length of the first prompt (tokens).  Includes system prompt if any."""

    assistant_response_length: list[int] = Field(default_factory=list)
    """Output token count for each tool-calling turn."""

    tool_call_latency: list[float] = Field(default_factory=list)
    """Wall-clock latency of each tool call (seconds)."""

    tool_call_output_length: list[int] = Field(default_factory=list)
    """Tokens returned by each tool call (prefilled in the *next* turn)."""

    final_assistant_response_length: int = Field(default=0, ge=0)
    """Output token count for the final turn (no tool call afterwards)."""

    # --- Optional: prefix-cache modelling ---
    system_prompt_hash: str | None = None
    """Content hash for the system-prompt portion."""

    system_prompt_length: int = Field(default=0, ge=0)
    """Tokens belonging to the shared system prompt."""

    arrival_time_s: float = Field(default=0.0, ge=0)
    """Absolute arrival time of the first request (seconds)."""

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def num_tool_turns(self) -> int:
        """Number of turns that include a tool call."""
        return len(self.assistant_response_length)

    @property
    def num_total_turns(self) -> int:
        """Total number of turns (tool turns + 1 final turn)."""
        return self.num_tool_turns + 1


class WorkloadTrace(BaseModel):
    """A collection of session traces with an arrival model."""

    sessions: list[SessionTrace] = Field(default_factory=list)
    arrival_pattern: str = Field(default="poisson")
    """One of ``"explicit"``, ``"poisson"``, or ``"bulk"``."""

    arrival_rate_per_sec: float = Field(default=0.0, ge=0)
    """Mean arrival rate for the Poisson process (sessions / second)."""

    description: str = ""
