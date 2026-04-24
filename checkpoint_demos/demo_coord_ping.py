"""Coordinator-backed ping (rank 0): does a trivial 2-way all-reduce per
iteration via ``prepare()``. Demonstrates the fast-path and slow-path
branches of the API.

Run via ``run_coord_demo.py``.
"""

import argparse
import os
import sys
import time

import torch


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoint")
)
from coord_client import CoordClient, NoPeers, PeerGone


RANK = 0
PEER = 1


def log(*args):
    print("[ping]", *args, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coord-addr", default=os.environ.get("COORD_ADDR"))
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    assert args.coord_addr, "pass --coord-addr or set COORD_ADDR"

    client = CoordClient(addr=args.coord_addr)
    client.register(RANK)
    # Rank 0 is the initial holder (see --initial-rank 0 in the launcher),
    # so wait_for_turn is a no-op here — we already hold the baton.
    client.wait_for_turn()

    torch.cuda.init()
    torch.cuda.synchronize()
    log(f"start: free VRAM = {torch.cuda.mem_get_info()[0]} B")

    for i in range(args.iters):
        # Local contribution: my rank + iteration number.
        my = torch.full((4,), float(RANK + i * 10), device="cuda")
        log(f"iter {i}: my contribution sum={my.sum().item():.1f}")

        # Do the all-reduce via prepare().
        t0 = time.monotonic()
        result = client.prepare(send={(PEER,): my}, recv=(PEER,))
        if result is None:
            # Slow path: peer hasn't deposited. Release the GPU and wait.
            log(f"iter {i}: prepare returned block; releasing")
            try:
                result = client.release_gpu()
            except (PeerGone, NoPeers) as e:
                log(f"iter {i}: peer gone ({e!r}); exiting")
                break
            log(f"iter {i}: resumed after {time.monotonic() - t0:.2f}s")
        else:
            log(f"iter {i}: fast path in {(time.monotonic() - t0) * 1000:.0f}ms")

        peer_t = result[PEER]
        reduced = my + peer_t
        log(f"iter {i}: reduced sum={reduced.sum().item():.1f}")

        # Expected: 2 * (RANK + i*10) + (PEER + i*10) - RANK = RANK + PEER + 2*i*10
        expected = float(RANK + PEER + 2 * i * 10) * 4
        assert abs(reduced.sum().item() - expected) < 1e-3, (
            f"iter {i}: expected sum {expected}, got {reduced.sum().item()}"
        )
        log(f"iter {i}: all-reduce correct")

    log("done; signaling coordinator")
    client.done()


if __name__ == "__main__":
    main()
