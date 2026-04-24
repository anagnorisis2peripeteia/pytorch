"""Ping process: initial GPU holder. Allocates a persistent tensor, iterates
(touch, release-GPU, reacquire), and verifies pointer + value stability across
each checkpoint/restore cycle.

Run via ``run_ping_pong.py`` — don't launch directly.
"""

import argparse
import time

from baton import Baton

import torch


def log(*args):
    print("[ping]", *args, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baton-dir", required=True)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    # Force CUDA init before anything else.
    torch.cuda.init()
    torch.cuda.synchronize()

    log(f"start: free VRAM = {torch.cuda.mem_get_info()[0]} B")

    baton = Baton(args.baton_dir, my_name="ping", peer_name="pong")

    # Allocate a persistent tensor we'll check across checkpoints.
    x = torch.arange(10.0, device="cuda")
    baseline = x.clone().cpu()
    ptr = x.data_ptr()
    log(f"point 0: x.sum={x.sum().item():.4f}, ptr={hex(ptr)}")
    log(f"point 0: free VRAM = {torch.cuda.mem_get_info()[0]} B")

    for i in range(args.iters):
        # "Training": mutate x deterministically, then revert so we can
        # compare against baseline across iterations.
        x.add_(1.0)
        x.sub_(1.0)
        torch.cuda.synchronize()

        log(
            f"iter {i}: pre-release x.sum={x.sum().item():.4f}, ptr={hex(x.data_ptr())}"
        )
        log(f"iter {i}: pre-release free VRAM = {torch.cuda.mem_get_info()[0]} B")

        # Yield the GPU.
        t0 = time.monotonic()
        baton.release()
        log(
            f"iter {i}: released (checkpoint took {time.monotonic() - t0:.2f}s); waiting for pong..."
        )

        # Now we own no GPU. Wait for pong to finish.
        took = baton.acquire()
        if not took:
            log("peer is done; exiting early")
            break
        log(f"iter {i}: reacquired after {time.monotonic() - t0:.2f}s total")

        # Verify pointer stability and value preservation.
        assert x.data_ptr() == ptr, (
            f"pointer changed! {hex(ptr)} -> {hex(x.data_ptr())}"
        )
        torch.testing.assert_close(x.cpu(), baseline)
        log(
            f"iter {i}: post-restore x.sum={x.sum().item():.4f}, ptr={hex(x.data_ptr())} (OK)"
        )

    log("all iterations done; signaling done")
    baton.done()


if __name__ == "__main__":
    main()
