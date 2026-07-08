"""Command-line interface for vllm-sim.

Usage examples::

    uv run vllm-sim agentic_coding_8k.jsonl -n 10 -p
    uv run vllm-sim agentic_coding_8k.jsonl -n 100 --rate 8 --kv-gb 64 -p
    uv run vllm-sim agentic_coding_8k.jsonl --arrival bulk -p -o metrics.csv
"""

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from vllm_sim.concur.controller import ConcurConfig
from vllm_sim.engine.config import EngineSimConfig
from vllm_sim.trace.driver import TraceDriver
from vllm_sim.trace.schema import SessionTrace, WorkloadTrace

app = typer.Typer()
console = Console(force_terminal=True, legacy_windows=False)


def _load_sessions(
    path: Path, system_prompt_length: int
) -> list[SessionTrace]:
    import json

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


@app.command()
def run(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to JSONL dataset file.",
        ),
    ],
    # --- Dataset ---
    sample: Annotated[
        int | None,
        typer.Option("--sample", "-n", help="Number of sessions to load (default: all)."),
    ] = None,
    sys_prompt_len: Annotated[
        int,
        typer.Option("--sys-prompt-len", help="Shared system-prompt length (tokens)."),
    ] = 4096,
    # --- Arrival ---
    arrival: Annotated[
        Literal["explicit", "poisson", "bulk"],
        typer.Option("--arrival", help="Session arrival pattern."),
    ] = "poisson",
    progress: Annotated[
        bool,
        typer.Option("--progress", "-p", help="Show rich progress bar."),
    ] = False,
    rate: Annotated[
        float,
        typer.Option("--rate", "-r", help="Arrival rate for Poisson (sessions/sec)."),
    ] = 4.0,
    # --- KV Cache ---
    kv_gb: Annotated[
        float,
        typer.Option("--kv-gb", help="Total GPU memory for KV cache (GiB)."),
    ] = 32.0,
    kv_mib_per_tok: Annotated[
        float,
        typer.Option("--kv-mib-per-tok", help="KV cache memory per token (MiB)."),
    ] = 0.5,
    block_size: Annotated[
        int,
        typer.Option("--block-size", help="Tokens per KV cache block."),
    ] = 16,
    no_prefix_cache: Annotated[
        bool,
        typer.Option("--no-prefix-cache", help="Disable prefix caching."),
    ] = False,
    # --- Scheduler ---
    max_batched_tokens: Annotated[
        int,
        typer.Option("--max-batched-tokens", help="Max tokens per prefill step."),
    ] = 8192,
    max_seqs: Annotated[
        int,
        typer.Option("--max-seqs", help="Max concurrent sequences."),
    ] = 512,
    # --- CONCUR (agent-level admission control) ---
    concur: Annotated[
        bool,
        typer.Option("--concur/--no-concur", help="Enable CONCUR agent-level admission control."),
    ] = False,
    concur_alpha: Annotated[
        int,
        typer.Option("--concur-alpha", help="AIMD additive increase step (agents)."),
    ] = 2,
    concur_beta: Annotated[
        float,
        typer.Option("--concur-beta", help="AIMD multiplicative decrease factor."),
    ] = 0.5,
    concur_u_low: Annotated[
        float,
        typer.Option("--concur-u-low", help="KV usage below which CONCUR probes upward."),
    ] = 0.2,
    concur_u_high: Annotated[
        float,
        typer.Option("--concur-u-high", help="KV usage above which thrashing is suspected."),
    ] = 0.5,
    concur_h_thresh: Annotated[
        float,
        typer.Option("--concur-h-thresh", help="Hit rate below which system is thrashing."),
    ] = 0.2,
    concur_init_window: Annotated[
        int,
        typer.Option("--concur-init-window", help="Initial congestion window (agents)."),
    ] = 2,
    concur_verbose: Annotated[
        bool,
        typer.Option("--concur-verbose", help="Log every CONCUR window-size change."),
    ] = False,
    # --- Timing ---
    prefill_us: Annotated[
        float,
        typer.Option("--prefill-us", help="Prefill latency per token (us)."),
    ] = 50.0,
    decode_us: Annotated[
        float,
        typer.Option("--decode-us", help="Decode latency per token (us)."),
    ] = 200.0,
    timing_profile: Annotated[
        Path | None,
        typer.Option("--timing-profile", help="JSON profile for hardware-aware timing model."),
    ] = None,
    # --- Output ---
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for arrival generation."),
    ] = 42,
    trace_exec: Annotated[
        bool,
        typer.Option("--trace-exec", help="Print per-request admission/finish log."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write per-session metrics to CSV."),
    ] = None,
) -> None:
    """Run the simulator on a multi-turn agent trace dataset."""

    # --- Load ---
    all_sessions = _load_sessions(dataset, sys_prompt_len)
    total = len(all_sessions)
    if sample is not None:
        all_sessions = all_sessions[:sample]

    config = EngineSimConfig(
        total_kv_memory_gb=kv_gb,
        kv_mib_per_token=kv_mib_per_tok,
        block_size=block_size,
        enable_prefix_cache=not no_prefix_cache,
        max_num_batched_tokens=max_batched_tokens,
        max_num_seqs=max_seqs,
        prefill_us_per_token=prefill_us,
        decode_us_per_token=decode_us,
        timing_profile=str(timing_profile) if timing_profile else None,
    )

    # --- Config summary ---
    cfg_lines = [
        f"[bold]Sessions:[/] {len(all_sessions)} / {total}",
        f"[bold]Arrival:[/] {arrival}" + (f" @ {rate}/s" if arrival == "poisson" else ""),
        f"[bold]KV cache:[/] {kv_gb} GiB / {kv_mib_per_tok} MiB/tok → {config.num_gpu_blocks} blocks",
        f"[bold]System prompt:[/] {sys_prompt_len} tokens ({sys_prompt_len // block_size} blocks) shared",
        f"[bold]Prefix cache:[/] {'on' if config.enable_prefix_cache else 'off'}",
    ]
    if concur:
        cfg_lines.append(
            f"[bold]CONCUR:[/] on  alpha={concur_alpha} beta={concur_beta} "
            f"U_low={concur_u_low} U_high={concur_u_high} H_thresh={concur_h_thresh} "
            f"W0={concur_init_window}"
        )
    console.print(Panel("\n".join(cfg_lines), title="Configuration", border_style="blue"))

    trace = WorkloadTrace(
        sessions=all_sessions,
        arrival_pattern=arrival,
        arrival_rate_per_sec=rate,
        description=f"{dataset.name} ({len(all_sessions)} sessions)",
    )

    # --- CONCUR config ---
    _concur_cfg: ConcurConfig | None = None
    if concur:
        _concur_cfg = ConcurConfig(
            enabled=True,
            alpha=concur_alpha,
            beta=concur_beta,
            U_low=concur_u_low,
            U_high=concur_u_high,
            H_thresh=concur_h_thresh,
            initial_window=concur_init_window,
            verbose=concur_verbose,
        )

    # --- Run ---
    cb = _make_rich_progress(len(all_sessions)) if progress else None

    if trace_exec:
        # Instrument the scheduler to log per-request admission.
        _patch_trace_exec(all_sessions)

    driver = TraceDriver(
        trace, config, seed=seed,
        progress_callback=cb, concur_config=_concur_cfg,
    )
    try:
        engine, metrics = driver.run()
    except MemoryError as exc:
        _stop_progress(cb)
        console.print(f"\n[bold red]✗ OOM:[/] {exc}")
        raise typer.Exit(code=1)
    finally:
        _stop_progress(cb)

    # --- Results ---
    _print_results(engine, metrics)

    # --- Trace ---
    if trace_exec:
        _print_trace(all_sessions)
        _print_concur_log(engine)

    # --- CSV ---
    if output is not None:
        _write_csv(output, metrics)
        console.print(f"\n[green]✓[/] Metrics written to [bold]{output}[/]")


# ---------------------------------------------------------------------------
# Rich progress bar
# ---------------------------------------------------------------------------

def _make_rich_progress(total_sessions: int):
    """Return a callback that drives a rich ``Progress`` bar."""

    prg = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )
    prg.start()
    task = prg.add_task(
        "[cyan]Simulating", total=total_sessions, completed=0,
    )

    def callback(completed_sessions: int, _total: int, completed_turns: int, sim_time_s: float) -> None:
        prg.update(task, completed=completed_sessions,
                    description=f"[cyan]Simulating[/] ({completed_turns} turns, sim {sim_time_s:.0f}s)")

    # Wrap to stop the progress bar when done.
    original_callback = callback

    def cleanup_callback(*args, **kwargs):
        original_callback(*args, **kwargs)

    cleanup_callback._progress = prg  # type: ignore[attr-defined]
    return cleanup_callback


def _stop_progress(callback) -> None:
    if callback is None:
        return
    prg = getattr(callback, "_progress", None)
    if prg is not None:
        prg.stop()


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_results(engine, metrics) -> None:
    console.print()

    # --- System summary ---
    table = Table(box=box.ROUNDED, title="Simulation Results", border_style="green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Sessions", str(metrics.total_sessions))
    table.add_row("Turns", str(metrics.total_turns))
    table.add_row("Sim time", f"{metrics.total_sim_time_us / 1_000_000:.1f}s")
    table.add_row("Throughput", f"{metrics.throughput_tokens_per_sec:,.0f} tok/s")
    table.add_row("Prefix hit rate", f"{metrics.prefix_hit_rate:.1%}")
    table.add_row("Hits / Misses", f"{metrics.total_prefix_hits:,} / {metrics.total_prefix_misses:,}")

    if metrics.kv_usage_samples:
        usages = [u for _, u in metrics.kv_usage_samples]
        table.add_row("KV usage (avg)", f"{sum(usages) / len(usages):.1%}")
    snap = engine.snapshot()
    table.add_row("KV usage (peak)", f"{snap['peak_usage']:.1%}")
    table.add_row("Prefix cache entries", f"{snap['prefix_cache_entries']:,}")

    # --- CONCUR stats (if enabled) ---
    concur_info = snap.get("concur")
    if concur_info is not None:
        table.add_row("CONCUR final window W", str(concur_info["W"]))
        wh = concur_info.get("window_history", [])
        if wh:
            final_u = wh[-1][2]
            final_h = wh[-1][3]
            table.add_row("CONCUR final U / H", f"{final_u:.2f} / {final_h:.2f}")

    console.print(table)

    # --- Per-session latency ---
    latencies = [
        sm.e2e_latency_us / 1_000_000
        for sm in metrics.session_metrics.values()
    ]
    if latencies:
        srt = sorted(latencies)
        lt = Table(box=box.SIMPLE, title="Per-Session E2E Latency (s)")
        lt.add_column("min", style="dim")
        lt.add_column("p50", style="yellow")
        lt.add_column("p99", style="yellow")
        lt.add_column("max", style="red")
        lt.add_row(
            f"{min(srt):.1f}",
            f"{srt[len(srt)//2]:.1f}",
            f"{srt[int(len(srt)*0.99)]:.1f}",
            f"{max(srt):.1f}",
        )
        console.print(lt)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------

_trace_rows: list[dict] = []


def _patch_trace_exec(sessions: list[SessionTrace]) -> None:
    """Monkey-patch the scheduler to log every request admission."""
    from vllm_sim.engine import scheduler as smod
    from vllm_sim.engine.request import RequestStatus

    _orig_try_admit = smod.Scheduler.try_admit

    def _traced_try_admit(self, clock_us):
        result = _orig_try_admit(self, clock_us)
        for req in result:
            if req.status == RequestStatus.PREFILLING:
                _trace_rows.append({
                    "t_us": clock_us,
                    "req": req.request_id,
                    "prompt": req.total_prompt_tokens,
                    "max_tok": req.max_tokens,
                    "blocks": len(req.block_ids),
                    "hits": req.prefix_hits,
                    "misses": req.prefix_misses,
                })
        return result

    smod.Scheduler.try_admit = _traced_try_admit


def _print_trace(sessions: list[SessionTrace]) -> None:
    """Print the execution trace as a table."""
    from rich.table import Table
    from rich import box

    table = Table(box=box.SIMPLE, title="Execution Trace")
    table.add_column("time", style="dim")
    table.add_column("request", style="cyan")
    table.add_column("prompt tok", justify="right")
    table.add_column("decode tok", justify="right")
    table.add_column("blocks", justify="right")
    table.add_column("hits", justify="right", style="green")
    table.add_column("misses", justify="right", style="red")
    table.add_column("hit %", justify="right")

    for row in _trace_rows:
        total = row["hits"] + row["misses"]
        pct = f"{row['hits'] / total:.0%}" if total else "-"
        table.add_row(
            f"{row['t_us'] / 1_000_000:.3f}s",
            row["req"],
            str(row["prompt"]),
            str(row["max_tok"]),
            str(row["blocks"]),
            str(row["hits"]),
            str(row["misses"]),
            pct,
        )
    console.print(table)


def _print_concur_log(engine) -> None:
    """Print CONCUR window-change log if available."""
    snap = engine.snapshot()
    concur_info = snap.get("concur")
    if concur_info is None:
        return
    change_log = concur_info.get("change_log", [])
    if not change_log:
        return

    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE, title="CONCUR Window Changes")
    table.add_column("Event", style="cyan")
    for entry in change_log:
        table.add_row(entry)
    console.print(table)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(path: Path, metrics) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "session_id", "turns", "e2e_latency_s", "total_queue_s",
            "prefix_hit_rate", "tokens_prefill_total",
        ])
        for sm in metrics.session_metrics.values():
            writer.writerow([
                sm.session_id,
                sm.num_turns,
                f"{sm.e2e_latency_us / 1_000_000:.3f}",
                f"{sm.total_queue_us / 1_000_000:.3f}",
                f"{sm.prefix_hit_rate:.4f}",
                sum(t.prefill_tokens for t in sm.turns),
            ])


if __name__ == "__main__":
    app()
