"""Session manager: tracks every active multi-turn session.

For each session we remember:
* The current turn index.
* Cumulative prompt length (grows with every turn).
* Which ``ContentSegment`` list to pass to the engine for the next turn.
"""

import numpy as np

from vllm_sim.engine.request import ContentSegment

from .schema import SessionTrace


class SessionManager:
    """Tracks state for all concurrent multi-turn sessions."""

    def __init__(self, sessions: list[SessionTrace]) -> None:
        self._sessions: dict[str, SessionTrace] = {s.session_id: s for s in sessions}
        # session_id → current turn index (0 = first turn).
        self._turn: dict[str, int] = {s.session_id: 0 for s in sessions}
        # session_id → cumulative full-prompt length (tokens).
        self._cum_len: dict[str, int] = {
            s.session_id: s.input_prompt_length for s in sessions
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def get(self, session_id: str) -> SessionTrace | None:
        return self._sessions.get(session_id)

    def current_turn(self, session_id: str) -> int:
        return self._turn.get(session_id, -1)

    def is_complete(self, session_id: str) -> bool:
        s = self._sessions.get(session_id)
        if s is None:
            return True
        return self._turn[session_id] >= s.num_total_turns

    def active_count(self) -> int:
        return sum(1 for sid in self._sessions if not self.is_complete(sid))

    def tool_wait_s(self, session_id: str, turn: int) -> float:
        """Tool-call latency for *turn* (seconds)."""
        s = self._sessions[session_id]
        if turn < s.num_tool_turns:
            return s.tool_call_latency[turn]
        return 0.0

    # ------------------------------------------------------------------
    # Turn construction
    # ------------------------------------------------------------------

    def make_first_turn_request_id(self, session_id: str) -> str:
        return f"{session_id}-t0"

    def make_turn_request_id(self, session_id: str, turn: int) -> str:
        return f"{session_id}-t{turn}"

    def build_segments(self, session_id: str, turn: int) -> list[ContentSegment]:
        """Build the ``ContentSegment`` list for session *session_id*, turn *turn*.

        The segments model how the KV cache sees the prompt:

        1. **System prompt** (turn 0 only, optional).  If the session
           has ``system_prompt_hash`` and ``system_prompt_length``, this
           segment can be shared across sessions via prefix cache.

        2. **History** — all tokens from previous turns.  Each turn's
           assistant output and tool result receive their own content
           hash so that within-session prefix caching works (turn N
           reuses turn N-1ʼs blocks).

        3. **Tool result** (turn > 0) — the tool output from the
           *previous* turn, prefilled at the start of this turn.
        """
        s = self._sessions[session_id]
        segments: list[ContentSegment] = []

        # --- First prompt (always included, even for turn > 0) ---
        # The first prompt is the same content every turn within a
        # session, so it gets a session-scoped hash for prefix caching.
        first_hash = f"{session_id}-first-prompt"
        if s.system_prompt_hash and s.system_prompt_length > 0:
            segments.append(
                ContentSegment(
                    content_hash=s.system_prompt_hash,
                    num_tokens=s.system_prompt_length,
                )
            )
            remainder = s.input_prompt_length - s.system_prompt_length
            if remainder > 0:
                segments.append(
                    ContentSegment(content_hash=first_hash, num_tokens=remainder)
                )
        else:
            segments.append(
                ContentSegment(content_hash=first_hash, num_tokens=s.input_prompt_length)
            )

        # --- History from completed turns ---
        for t in range(turn):
            # Assistant response of turn t.
            asst_hash = f"{session_id}-t{t}-asst"
            segments.append(
                ContentSegment(
                    content_hash=asst_hash,
                    num_tokens=s.assistant_response_length[t],
                )
            )
            # Tool result after turn t (prefilled in turn t+1).
            tool_hash = f"{session_id}-t{t}-tool"
            segments.append(
                ContentSegment(
                    content_hash=tool_hash,
                    num_tokens=s.tool_call_output_length[t],
                )
            )

        return segments

    def make_request_params(
        self, session_id: str, turn: int
    ) -> tuple[list[ContentSegment], int]:
        """Return ``(segments, max_tokens)`` for the given turn."""
        s = self._sessions[session_id]
        segments = self.build_segments(session_id, turn)

        if turn < s.num_tool_turns:
            max_tokens = s.assistant_response_length[turn]
        else:
            max_tokens = s.final_assistant_response_length

        return segments, max_tokens

    def advance_turn(self, session_id: str) -> None:
        """Mark session as having completed its current turn."""
        s = self._sessions[session_id]
        t = self._turn[session_id]

        # Update cumulative prompt length.
        if t < s.num_tool_turns:
            self._cum_len[session_id] += (
                s.assistant_response_length[t] + s.tool_call_output_length[t]
            )
        # Final turn adds only the final response length (no tool call).
        # Not strictly needed for simulation since the session ends.

        self._turn[session_id] = t + 1

    def build_initial_arrivals(
        self, pattern: str = "poisson", rate_per_sec: float = 4.0, rng: np.random.Generator | None = None
    ) -> dict[str, float]:
        """Compute the arrival time (seconds) for every session's first turn.

        Returns ``{session_id: arrival_time_s}``.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        arrivals: dict[str, float] = {}
        ids = sorted(self._sessions.keys())
        n = len(ids)

        if pattern == "bulk":
            for sid in ids:
                arrivals[sid] = 0.0

        elif pattern == "poisson":
            # Generate inter-arrival gaps from exponential distribution.
            gaps = rng.exponential(1.0 / rate_per_sec, size=n) if rate_per_sec > 0 else [0.0] * n
            t = 0.0
            for sid, gap in zip(ids, gaps):
                arrivals[sid] = t
                t += gap

        else:  # "explicit"
            for sid in ids:
                arrivals[sid] = self._sessions[sid].arrival_time_s

        return arrivals
