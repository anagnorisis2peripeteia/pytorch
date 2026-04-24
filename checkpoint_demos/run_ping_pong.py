"""Orchestrator for the ping-pong GPU baton demo.

Creates a baton dir, seeds the token to "ping" (so ping starts holding the
GPU), launches both demo processes, streams their stdout interleaved, and
cleans up on exit.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument(
        "--keep-dir",
        action="store_true",
        help="Don't delete baton dir on exit (useful for debugging)",
    )
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Pin both processes to this GPU via CUDA_VISIBLE_DEVICES.",
    )
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    baton_dir = tempfile.mkdtemp(prefix="gpu_baton_")
    print(f"[orchestrator] baton dir = {baton_dir}", flush=True)

    # Seed the token so ping starts as the holder.
    with open(os.path.join(baton_dir, "token.tmp"), "w") as f:
        f.write("ping")
    os.replace(os.path.join(baton_dir, "token.tmp"), os.path.join(baton_dir, "token"))

    common = ["--baton-dir", baton_dir, "--iters", str(args.iters)]
    ping_cmd = [sys.executable, "-u", os.path.join(here, "demo_ping.py"), *common]
    pong_cmd = [sys.executable, "-u", os.path.join(here, "demo_pong.py"), *common]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(
        f"[orchestrator] launching ping: {' '.join(ping_cmd)} "
        f"(CUDA_VISIBLE_DEVICES={args.gpu})",
        flush=True,
    )
    print(
        f"[orchestrator] launching pong: {' '.join(pong_cmd)} "
        f"(CUDA_VISIBLE_DEVICES={args.gpu})",
        flush=True,
    )

    # Both share this stdout/stderr so we see output interleaved.
    ping = subprocess.Popen(ping_cmd, env=env)
    pong = subprocess.Popen(pong_cmd, env=env)

    rc_ping = ping.wait()
    rc_pong = pong.wait()

    print(f"[orchestrator] ping rc={rc_ping}, pong rc={rc_pong}", flush=True)

    if not args.keep_dir:
        shutil.rmtree(baton_dir, ignore_errors=True)

    sys.exit(0 if (rc_ping == 0 and rc_pong == 0) else 1)


if __name__ == "__main__":
    main()
