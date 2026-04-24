"""Pong process: non-initial GPU holder. Waits for ping to release the GPU,
verifies that VRAM is (nearly) fully available, allocates a workload that only
fits when ping is checkpointed, releases back, and loops.

Run via ``run_ping_pong.py`` — don't launch directly.
"""

import argparse
import time

from baton import Baton

import torch


def log(*args):
    print("[pong]", *args, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baton-dir", required=True)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    # Block until ping hands us the GPU. We have NO CUDA state yet, so first
    # acquire() is just a token wait (no restore needed).
    baton = Baton(args.baton_dir, my_name="pong", peer_name="ping")
    log("waiting for initial turn...")
    if not baton.acquire():
        log("ping signaled done before pong started; nothing to do")
        return

    torch.cuda.init()
    torch.cuda.synchronize()

    log(f"start: free VRAM = {torch.cuda.mem_get_info()[0]} B")

    # Persistent tensor on this side too, to exercise our own pointer stability.
    y = torch.full((10,), 42.0, device="cuda")
    baseline = y.clone().cpu()
    ptr = y.data_ptr()
    log(f"point 0: y.sum={y.sum().item():.4f}, ptr={hex(ptr)}")

    for i in range(1, args.iters + 1):
        log(f"iter {i}: free VRAM = {torch.cuda.mem_get_info()[0]} GB (ping is paused)")

        # Do some work that would be contentious with ping's tensors if ping
        # weren't checkpointed. We won't literally grab all VRAM (that could
        # hit driver reserves), but we'll do a sizeable chunk.
        big = torch.randn(1024 * i, 1024 * i, 256, device="cuda")  # 1 GiB fp32
        _ = (big * 2).sum()
        del big
        torch.cuda.synchronize()

        log(
            f"iter {i}: pre-release y.sum={y.sum().item():.4f}, ptr={hex(y.data_ptr())}"
        )

        t0 = time.monotonic()
        baton.release()
        log(
            f"iter {i}: released (checkpoint took {time.monotonic() - t0:.2f}s); waiting for ping..."
        )

        took = baton.acquire()
        if not took:
            log("ping is done; exiting")
            break
        log(f"iter {i}: reacquired after {time.monotonic() - t0:.2f}s total")

        assert y.data_ptr() == ptr, (
            f"pointer changed! {hex(ptr)} -> {hex(y.data_ptr())}"
        )
        torch.testing.assert_close(y.cpu(), baseline)
        log(f"iter {i}: post-restore y.sum={y.sum().item():.4f} (OK)")

    log("all iterations done; signaling done")
    baton.done()


if __name__ == "__main__":
    main()
