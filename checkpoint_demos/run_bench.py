"""Sweep driver for checkpoint/restore benchmarks.

Runs three sweeps:
  A (size):     num_tensors=1; size_mb varies.
  B (count):    size_mb=1024; num_tensors varies (both roles equal).
  C (symmetry): ping fixed; pong varies.

For each (sweep, config) it spawns a ping/pong pair, waits, and aggregates
results into a single CSV. Use ``--full`` for the large configs; the default
uses a small smoke-test set that finishes in well under a minute.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time


HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_baton.py")


# ---- Sweep definitions ----


def sweep_A_size(full: bool):
    """Total size sweep (both roles symmetric)."""
    sizes = [64, 256, 1024, 4096, 16384, 40960] if full else [64, 256]
    for size_mb in sizes:
        config_id = f"size_mb={size_mb}"
        yield (
            "size",
            config_id,
            {"size_mb": size_mb, "num_tensors": 1},
            {"size_mb": size_mb, "num_tensors": 1},
        )


def sweep_B_count(full: bool):
    """Tensor count sweep (both roles symmetric, total size fixed at 1 GiB)."""
    counts = [1, 10, 100, 1_000, 10_000, 100_000] if full else [1, 10]
    for n in counts:
        config_id = f"num_tensors={n}"
        yield (
            "count",
            config_id,
            {"size_mb": 1024, "num_tensors": n},
            {"size_mb": 1024, "num_tensors": n},
        )


def sweep_C_symmetry(full: bool):
    """Symmetry sweep. Ping fixed at (1024 MB, 100 tensors). Pong varies."""
    pings = {"size_mb": 1024, "num_tensors": 100}
    pongs = (
        [
            {"size_mb": 64, "num_tensors": 100},
            {"size_mb": 1024, "num_tensors": 100},
            {"size_mb": 16384, "num_tensors": 100},
            {"size_mb": 1024, "num_tensors": 10_000},
        ]
        if full
        else [
            {"size_mb": 64, "num_tensors": 100},
            {"size_mb": 1024, "num_tensors": 100},
        ]
    )
    for pong in pongs:
        config_id = f"pong_size_mb={pong['size_mb']},pong_n={pong['num_tensors']}"
        yield ("symmetry", config_id, pings, pong)


SWEEPS = [sweep_A_size, sweep_B_count, sweep_C_symmetry]


# ---- Runner ----


def _run_pair(sweep_name, config_id, ping_cfg, pong_cfg, iters, out_csv, gpu):
    """Launch a ping/pong pair for one config. Raises on non-zero exit."""
    baton_dir = tempfile.mkdtemp(prefix="gpu_bench_")
    env = os.environ.copy()
    # Pin both processes to the same single GPU. Without this they could pick
    # different devices on multi-GPU hosts and checkpoint/restore would not
    # actually contend for memory on the same device.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        # Seed token so ping starts as holder.
        tmp = os.path.join(baton_dir, "token.tmp")
        with open(tmp, "w") as f:
            f.write("ping")
        os.replace(tmp, os.path.join(baton_dir, "token"))

        def _cmd(role, cfg):
            return [
                sys.executable,
                "-u",
                BENCH,
                "--role",
                role,
                "--baton-dir",
                baton_dir,
                "--size-mb",
                str(cfg["size_mb"]),
                "--num-tensors",
                str(cfg["num_tensors"]),
                "--iters",
                str(iters),
                "--out-csv",
                out_csv,
                "--sweep-name",
                sweep_name,
                "--config-id",
                config_id,
            ]

        ping_proc = subprocess.Popen(_cmd("ping", ping_cfg), env=env)
        pong_proc = subprocess.Popen(_cmd("pong", pong_cfg), env=env)
        rc_ping = ping_proc.wait()
        rc_pong = pong_proc.wait()
        if rc_ping != 0 or rc_pong != 0:
            raise RuntimeError(
                f"sweep={sweep_name} config={config_id} failed "
                f"(ping rc={rc_ping}, pong rc={rc_pong})"
            )
    finally:
        shutil.rmtree(baton_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run the large sweep configs (tens of minutes).",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Hop iterations per config (iter 0 = warmup).",
    )
    ap.add_argument("--out-csv", default="bench_results.csv")
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Pin both ping and pong to this GPU via CUDA_VISIBLE_DEVICES (default 0).",
    )
    args = ap.parse_args()

    # Fresh CSV each run.
    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    t_start = time.monotonic()
    run_count = 0
    for sweep in SWEEPS:
        for sweep_name, config_id, ping_cfg, pong_cfg in sweep(args.full):
            print(
                f"[run_bench] {sweep_name}: {config_id} "
                f"ping={ping_cfg} pong={pong_cfg}",
                flush=True,
            )
            t0 = time.monotonic()
            _run_pair(
                sweep_name,
                config_id,
                ping_cfg,
                pong_cfg,
                args.iters,
                args.out_csv,
                args.gpu,
            )
            print(f"[run_bench]   done in {time.monotonic() - t0:.1f}s", flush=True)
            run_count += 1

    print(
        f"[run_bench] finished {run_count} configs in "
        f"{time.monotonic() - t_start:.1f}s -> {args.out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
