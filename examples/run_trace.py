#!/usr/bin/env python3
"""Example: run the simulator with the real multi-turn agent trace format.

Usage:
    cd vllm-sim
    uv run python examples/run_trace.py
"""

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.trace.driver import TraceDriver
from vllm_sim.trace.schema import SessionTrace, WorkloadTrace


def main() -> None:
    # --- 1) Build trace from the real data ---
    trace = WorkloadTrace(
        sessions=[
            SessionTrace(
                session_id="agent-001",
                input_prompt_length=6342,
                assistant_response_length=[
                    85, 46, 59, 78, 187, 518, 79, 84, 382, 428, 252,
                ],
                tool_call_latency=[
                    37.197, 1.56, 0.899, 0.189, 0.226, 2.205,
                    1.259, 0.126, 0.605, 13.228, 0.814,
                ],
                tool_call_output_length=[
                    664, 664, 35, 407, 594, 20, 98, 188, 245, 102, 32,
                ],
                final_assistant_response_length=288,
            ),
        ],
        arrival_pattern="bulk",
        description="Single multi-turn agent session (11 tool turns + final)",
    )

    # --- 2) Configure the engine ---
    config = EngineSimConfig(
        total_kv_memory_gb=16.0,        # 16 GiB KV cache (≈2048 blocks @ 0.5 MiB/tok).
        block_size=16,
        max_num_batched_tokens=8192,    # Chunked prefill budget.
        max_num_seqs=256,
        prefill_us_per_token=50.0,      # Timing model params.
        decode_us_per_token=200.0,
        enable_prefix_cache=True,
    )

    # --- 3) Run simulation ---
    driver = TraceDriver(trace, config, seed=42)
    engine, metrics = driver.run()

    # --- 4) Print results ---
    s = metrics.session_metrics.get("agent-001")
    if s is None:
        print("No results.")
        return

    print(f"\n{'='*60}")
    print(f"Session: {s.session_id}")
    print(f"Turns:  {s.num_turns}")
    print(f"E2E latency: {s.e2e_latency_us / 1_000_000:.3f}s")
    print(f"Queue time:   {s.total_queue_us / 1_000_000:.3f}s")
    print(f"Prefix hit rate: {s.prefix_hit_rate:.1%}")
    print(f"\n{'Turn':<6} {'Prompt tok':>10} {'Queue ms':>10} {'Prefix hit':>10}")
    print("-" * 42)
    for tm in s.turns:
        print(
            f"{tm.turn:<6} {tm.prefill_tokens:>10} "
            f"{tm.queue_time_us/1000:>10.1f} "
            f"{tm.prefix_hits/tm.prefill_blocks if tm.prefill_blocks else 0:>9.1%}"
        )

    print(f"\n{'='*60}")
    print(f"System throughput: {metrics.throughput_tokens_per_sec:.0f} tok/s")
    print(f"Sim time:          {metrics.total_sim_time_us / 1_000_000:.3f}s")
    print(f"KV samples:        {len(metrics.kv_usage_samples)}")
    if metrics.kv_usage_samples:
        avg_kv = sum(u for _, u in metrics.kv_usage_samples) / len(
            metrics.kv_usage_samples
        )
        peak_kv = max(u for _, u in metrics.kv_usage_samples)
        print(f"Avg KV usage:      {avg_kv:.1%}")
        print(f"Peak KV usage:     {peak_kv:.1%}")


if __name__ == "__main__":
    main()
