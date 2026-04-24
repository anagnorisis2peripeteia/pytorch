"""Benchmark worker for checkpoint/restore perf.

One copy runs as ``--role ping`` (initial token holder), one as ``--role pong``
(waits for initial handoff). Each role allocates ``--num-tensors`` distinct
float32 tensors totalling roughly ``--size-mb`` MiB, then runs ``--iters`` hop
iterations. Each iteration times one ``checkpoint_self`` and one
``restore_self`` call (wall time, perf_counter_ns). Rows are appended to
``--out-csv``.

CSV columns:
    sweep,config_id,role,iter,phase,elapsed_ms,size_mb,num_tensors

Launched by ``run_bench.py`` — don't invoke directly unless you're debugging.
"""

import argparse
import csv
import os
import sys
import time

import torch


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoint")
)
from baton import Baton
from cuda_checkpoint import checkpoint_self, restore_self


def log(role, *args):
    print(f"[bench:{role}]", *args, flush=True)


def _allocate(size_mb: int, num_tensors: int):
    total_bytes = size_mb * 1024 * 1024
    elems_per = max(1, total_bytes // (4 * num_tensors))  # float32 = 4 bytes
    return [
        torch.empty(elems_per, dtype=torch.float32, device="cuda")
        for _ in range(num_tensors)
    ]


def _append_row(out_csv, row):
    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(
                [
                    "sweep",
                    "config_id",
                    "role",
                    "iter",
                    "phase",
                    "elapsed_ms",
                    "size_mb",
                    "num_tensors",
                ]
            )
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("ping", "pong"), required=True)
    ap.add_argument("--baton-dir", required=True)
    ap.add_argument("--size-mb", type=int, required=True)
    ap.add_argument("--num-tensors", type=int, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--sweep-name", required=True)
    ap.add_argument("--config-id", required=True)
    args = ap.parse_args()

    role = args.role
    other = "pong" if role == "ping" else "ping"
    baton = Baton(args.baton_dir, my_name=role, peer_name=other)

    # Pong cannot touch CUDA until it owns the token. Ping holds it by default.
    if role == "pong":
        log(role, "waiting for initial token handoff...")
        if not baton.acquire():
            log(role, "peer signaled done before we started; exiting")
            return

    torch.cuda.init()
    torch.cuda.synchronize()

    # Print the physical GPU identity so an orchestrator can confirm both
    # ping and pong are actually sharing one device.
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", "<unknown>")
    log(
        role,
        f"device: name={props.name!r} uuid={uuid} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
    )

    log(
        role,
        f"allocating {args.num_tensors} tensors totalling ~{args.size_mb} MiB",
    )
    tensors = _allocate(args.size_mb, args.num_tensors)  # noqa: F841 (kept alive)
    torch.cuda.synchronize()

    for i in range(args.iters):
        # At this point we are entitled to run (we hold the token):
        #   - ping iter 0: holder from start
        #   - ping iter >0: just restored via baton.acquire() below
        #   - pong iter 0: acquired before allocate
        #   - pong iter >0: just restored via baton.acquire() below

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        t0 = time.perf_counter_ns()
        checkpoint_self()
        t1 = time.perf_counter_ns()
        _append_row(
            args.out_csv,
            [
                args.sweep_name,
                args.config_id,
                role,
                i,
                "checkpoint",
                (t1 - t0) / 1e6,
                args.size_mb,
                args.num_tensors,
            ],
        )

        # Hand the token to peer. No CUDA ops allowed here.
        baton._write_token(other)

        # Wait for peer to yield back. File-only poll.
        while True:
            cur = baton._read_token()
            if cur == role:
                break
            if cur == "__DONE__":
                log(role, "peer signaled done while waiting; exiting")
                return
            time.sleep(0.05)

        t2 = time.perf_counter_ns()
        restore_self()
        t3 = time.perf_counter_ns()
        _append_row(
            args.out_csv,
            [
                args.sweep_name,
                args.config_id,
                role,
                i,
                "restore",
                (t3 - t2) / 1e6,
                args.size_mb,
                args.num_tensors,
            ],
        )

        log(
            role,
            f"iter {i}: ckpt={(t1 - t0) / 1e6:.1f}ms restore={(t3 - t2) / 1e6:.1f}ms",
        )

    log(role, "all iterations done; signaling done")
    baton.done()


if __name__ == "__main__":
    main()
