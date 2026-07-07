"""End-to-end tests: EngineSim + TraceDriver with real data format."""

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.trace.driver import TraceDriver
from vllm_sim.trace.schema import SessionTrace, WorkloadTrace


def _make_single_session_trace() -> WorkloadTrace:
    """Create a trace matching the user's real data format."""
    return WorkloadTrace(
        sessions=[
            SessionTrace(
                session_id="agent-001",
                input_prompt_length=6342,
                assistant_response_length=[85, 46, 59, 78, 187, 518, 79, 84, 382, 428, 252],
                tool_call_latency=[37.197, 1.56, 0.899, 0.189, 0.226, 2.205, 1.259, 0.126, 0.605, 13.228, 0.814],
                tool_call_output_length=[664, 664, 35, 407, 594, 20, 98, 188, 245, 102, 32],
                final_assistant_response_length=288,
            ),
        ],
        arrival_pattern="bulk",
    )


def _make_config() -> EngineSimConfig:
    """Config with enough KV cache for the trace."""
    return EngineSimConfig(
        num_gpu_blocks_override=2048,      # 2048×16=32768 tokens, enough for 11-turn prompt.
        block_size=16,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        enable_prefix_cache=True,
    )


class TestE2ESingleSession:
    def test_single_session_completes(self) -> None:
        trace = _make_single_session_trace()
        config = _make_config()
        driver = TraceDriver(trace, config, seed=42)

        engine, metrics = driver.run()

        # One session, 12 turns (11 tool turns + 1 final).
        assert metrics.total_sessions == 1
        assert metrics.total_turns == 12

        # Verify no block leak.
        assert engine.num_free_blocks == config.num_gpu_blocks

    def test_single_session_accumulates_prompt(self) -> None:
        """Verify that later turns have larger prompts (no resumable)."""
        trace = _make_single_session_trace()
        config = _make_config()
        driver = TraceDriver(trace, config, seed=42)

        engine, metrics = driver.run()

        sm = metrics.session_metrics.get("agent-001")
        assert sm is not None
        assert sm.num_turns == 12

        # First turn prefill = 6342 tokens.
        assert sm.turns[0].prefill_tokens == 6342

        # Later turns should have larger prefills (history accumulated).
        assert sm.turns[5].prefill_tokens > sm.turns[0].prefill_tokens
        assert sm.turns[-1].prefill_tokens > sm.turns[5].prefill_tokens


class TestE2EMultiSession:
    def test_two_concurrent_sessions(self) -> None:
        """Two identical sessions arriving at the same time."""
        trace = WorkloadTrace(
            sessions=[
                SessionTrace(
                    session_id="agent-001",
                    input_prompt_length=1000,
                    assistant_response_length=[100, 150],
                    tool_call_latency=[0.5, 0.3],
                    tool_call_output_length=[200, 300],
                    final_assistant_response_length=200,
                ),
                SessionTrace(
                    session_id="agent-002",
                    input_prompt_length=1000,
                    assistant_response_length=[100, 150],
                    tool_call_latency=[0.5, 0.3],
                    tool_call_output_length=[200, 300],
                    final_assistant_response_length=200,
                ),
            ],
            arrival_pattern="bulk",
        )
        config = EngineSimConfig(
            num_gpu_blocks_override=512, block_size=16,
            max_num_batched_tokens=8192, max_num_seqs=256,
        )
        driver = TraceDriver(trace, config, seed=42)

        engine, metrics = driver.run()

        assert metrics.total_sessions == 2
        assert metrics.total_turns == 6  # 2 sessions × 3 turns each.
        assert engine.num_free_blocks == config.num_gpu_blocks

    def test_prefix_cache_sharing(self) -> None:
        """Two sessions with the same system prompt hash share blocks."""
        trace = WorkloadTrace(
            sessions=[
                SessionTrace(
                    session_id="agent-001",
                    input_prompt_length=1500,
                    system_prompt_hash="sys_v1",
                    system_prompt_length=1500,
                    assistant_response_length=[100],
                    tool_call_latency=[0.1],
                    tool_call_output_length=[200],
                    final_assistant_response_length=100,
                ),
                SessionTrace(
                    session_id="agent-002",
                    input_prompt_length=1500,
                    system_prompt_hash="sys_v1",
                    system_prompt_length=1500,
                    assistant_response_length=[100],
                    tool_call_latency=[0.1],
                    tool_call_output_length=[200],
                    final_assistant_response_length=100,
                ),
            ],
            arrival_pattern="bulk",
        )
        config = EngineSimConfig(
            num_gpu_blocks_override=512, block_size=16,
            max_num_batched_tokens=8192, max_num_seqs=256,
            enable_prefix_cache=True,
        )
        driver = TraceDriver(trace, config, seed=42)

        engine, metrics = driver.run()

        # Second session should have prefix hits for its system prompt.
        assert metrics.total_prefix_hits > 0

    def test_poisson_arrivals(self) -> None:
        """Sessions arriving via a Poisson process."""
        sessions = []
        for i in range(5):
            sessions.append(
                SessionTrace(
                    session_id=f"agent-{i:03d}",
                    input_prompt_length=500,
                    assistant_response_length=[50],
                    tool_call_latency=[0.1],
                    tool_call_output_length=[100],
                    final_assistant_response_length=50,
                )
            )

        trace = WorkloadTrace(
            sessions=sessions,
            arrival_pattern="poisson",
            arrival_rate_per_sec=2.0,
        )
        config = EngineSimConfig(
            num_gpu_blocks_override=256, block_size=16,
            max_num_batched_tokens=4096, max_num_seqs=64,
        )
        driver = TraceDriver(trace, config, seed=42)

        engine, metrics = driver.run()

        assert metrics.total_sessions == 5
        assert metrics.total_turns == 10  # 2 turns each.
        # With Poisson arrivals, KV usage should vary over time.
        assert len(metrics.kv_usage_samples) > 0
