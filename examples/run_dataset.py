#!/usr/bin/env python3
"""Run the simulator on the full agentic-coding-8k dataset.

All sessions share a 4096-token system prompt (256 blocks @ block_size=16).
"""
import json
import sys
from pathlib import Path

from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.trace.driver import TraceDriver
from vllm_sim.trace.schema import SessionTrace, WorkloadTrace


def load_sessions(path: Path, system_prompt_length: int = 4096) -> list[SessionTrace]:
    """Load all sessions from a JSONL file.

    Every session gets ``system_prompt_hash="shared_sys"`` and
    ``system_prompt_length=4096`` so that the prefix cache can share
    the 256 system-prompt blocks across all sessions.
    """
    sessions: list[SessionTrace] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sessions.append(
                SessionTrace(
                    session_id=f"agent-{i:04d}",
                    input_prompt_length=obj["input_prompt_length"],
                    assistant_response_length=obj["assistant_response_length"],
                    tool_call_latency=obj["tool_call_latency"],
                    tool_call_output_length=obj["tool_call_output_length"],
                    final_assistant_response_length=obj["final_assistant_response_length"],
                    system_prompt_hash="shared_sys",
                    system_prompt_length=system_prompt_length,
                )
            )
    return sessions


def main() -> None:
    dataset = Path(__file__).resolve().parents[1] / "agentic_coding_8k.jsonl"
    if not dataset.exists():
        print(f"Dataset not found: {dataset}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    SAMPLE = 100  # Set to e.g. 10 for quick test, 100 for medium, None for all 8192.
    ARRIVAL_PATTERN = "poisson"
    ARRIVAL_RATE = 4.0  # sessions / second
    SYSTEM_PROMPT_LEN = 4096  # tokens shared across all sessions.
    KV_MEMORY_GB = 625.0  # 80k blocks @ 0.5 MiB/token → ~625 GiB.
    KV_MIB_PER_TOKEN = 0.5  # MiB/token (FP16, ~7-8B model).

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    all_sessions = load_sessions(dataset, system_prompt_length=SYSTEM_PROMPT_LEN)
    total_sessions = len(all_sessions)

    if SAMPLE is not None:
        all_sessions = all_sessions[:SAMPLE]

    print(f"Loaded {len(all_sessions)} / {total_sessions} sessions")
    print(f"  System prompt: {SYSTEM_PROMPT_LEN} tokens ({SYSTEM_PROMPT_LEN // 16} blocks) shared")
    print(f"  Arrival: {ARRIVAL_PATTERN}, rate={ARRIVAL_RATE}/s")
    print(f"  KV memory: {KV_MEMORY_GB} GiB, {KV_MIB_PER_TOKEN} MiB/token")

    # Stats about the dataset.
    turns_per_session = [
        1 + len(s.assistant_response_length) for s in all_sessions
    ]
    prompt_lengths = [s.input_prompt_length for s in all_sessions]
    print(f"  Turns/session: min={min(turns_per_session)}, max={max(turns_per_session)}, "
          f"avg={sum(turns_per_session) / len(turns_per_session):.1f}")
    print(f"  Input prompt:  min={min(prompt_lengths)}, max={max(prompt_lengths)}, "
          f"avg={sum(prompt_lengths) / len(prompt_lengths):.0f}")
    print(f"  Total tokens (decode): "
          f"{sum(sum(s.assistant_response_length) + s.final_assistant_response_length for s in all_sessions):,}")

    # ------------------------------------------------------------------
    # Configure
    # ------------------------------------------------------------------
    config = EngineSimConfig(
        total_kv_memory_gb=KV_MEMORY_GB,
        kv_mib_per_token=KV_MIB_PER_TOKEN,
        block_size=16,
        max_num_batched_tokens=8192,
        max_num_seqs=512,
        enable_prefix_cache=True,
        prefill_us_per_token=50.0,
        decode_us_per_token=200.0,
    )

    trace = WorkloadTrace(
        sessions=all_sessions,
        arrival_pattern=ARRIVAL_PATTERN,
        arrival_rate_per_sec=ARRIVAL_RATE,
        description=f"agentic-coding-8k ({len(all_sessions)} sessions, "
                    f"{SYSTEM_PROMPT_LEN}-tok shared system prompt)",
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    driver = TraceDriver(trace, config, seed=42)
    engine, metrics = driver.run()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"Simulation Results")
    print(f"{'='*65}")
    print(f"  Sessions:          {metrics.total_sessions}")
    print(f"  Total turns:       {metrics.total_turns}")
    print(f"  Sim time:          {metrics.total_sim_time_us / 1_000_000:.1f}s")
    print(f"  Throughput:        {metrics.throughput_tokens_per_sec:.0f} tok/s")
    print(f"  Prefix hit rate:   {metrics.prefix_hit_rate:.1%}")
    print(f"  Total hits:        {metrics.total_prefix_hits:,}")
    print(f"  Total misses:      {metrics.total_prefix_misses:,}")
    print(f"  KV samples:        {len(metrics.kv_usage_samples)}")
    if metrics.kv_usage_samples:
        usages = [u for _, u in metrics.kv_usage_samples]
        print(f"  KV usage (avg):    {sum(usages) / len(usages):.1%}")
        print(f"  KV usage (peak):   {max(usages):.1%}")
        print(f"  KV usage (p99):    {sorted(usages)[int(len(usages) * 0.99)]:.1%}")

    # Per-session stats.
    e2e_latencies = [
        sm.e2e_latency_us / 1_000_000
        for sm in metrics.session_metrics.values()
    ]
    queue_times = [
        sm.total_queue_us / 1_000_000
        for sm in metrics.session_metrics.values()
    ]
    hit_rates = [sm.prefix_hit_rate for sm in metrics.session_metrics.values()]

    print(f"\n  Per-session stats (n={len(e2e_latencies)}):")
    print(f"    E2E latency:  min={min(e2e_latencies):.1f}s, "
          f"med={sorted(e2e_latencies)[len(e2e_latencies) // 2]:.1f}s, "
          f"max={max(e2e_latencies):.1f}s")
    print(f"    Queue time:   min={min(queue_times):.1f}s, "
          f"med={sorted(queue_times)[len(queue_times) // 2]:.1f}s, "
          f"max={max(queue_times):.1f}s")
    print(f"    Hit rate:     min={min(hit_rates):.1%}, "
          f"med={sorted(hit_rates)[len(hit_rates) // 2]:.1%}, "
          f"max={max(hit_rates):.1%}")

    # Final KV state.
    snap = engine.snapshot()
    print(f"\n  Final engine state:")
    print(f"    Free blocks:  {snap['free_blocks']} / {snap['total_blocks']}")
    print(f"    Running:      {snap['running']}")
    print(f"    Waiting:      {snap['waiting']}")
    print(f"    Prefilling:   {snap['prefilling']}")


if __name__ == "__main__":
    main()
