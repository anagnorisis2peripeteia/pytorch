"""Orchestrator for the coordinator-backed ping-pong demo.

Starts the coordinator subprocess, reads its bound address from stdout,
exports it via COORD_ADDR, then launches ping+pong pinned to the same GPU
via CUDA_VISIBLE_DEVICES. Kills the coordinator on exit.
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "checkpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Pin ping and pong to this GPU via CUDA_VISIBLE_DEVICES.",
    )
    ap.add_argument(
        "--tcp",
        action="store_true",
        help="Use localhost TCP instead of UDS (default is UDS).",
    )
    args = ap.parse_args()

    # Launch coordinator with ping pre-seeded as initial holder.
    base = [
        sys.executable,
        "-u",
        os.path.join(LIB, "coordinator.py"),
        "--initial-rank",
        "0",
    ]
    if args.tcp:
        coord_cmd = base + ["--tcp-port", "0"]
    else:
        sock_path = os.path.join(tempfile.mkdtemp(prefix="coord_"), "coord.sock")
        coord_cmd = base + ["--socket", sock_path]

    print(f"[orchestrator] launching coordinator: {' '.join(coord_cmd)}", flush=True)
    coord = subprocess.Popen(coord_cmd, stdout=subprocess.PIPE, text=True)

    # First line of coord stdout is "ADDR <addr>".
    addr_line = coord.stdout.readline().strip()
    if not addr_line.startswith("ADDR "):
        print(
            f"[orchestrator] coordinator failed to start: got {addr_line!r}",
            file=sys.stderr,
        )
        coord.terminate()
        sys.exit(1)
    coord_addr = addr_line[len("ADDR ") :]
    print(f"[orchestrator] coord addr: {coord_addr}", flush=True)

    # Drain coord stdout in the background so it doesn't block.
    def _drain():
        for line in coord.stdout:
            sys.stdout.write("[coord-stdout] " + line)
            sys.stdout.flush()

    import threading

    threading.Thread(target=_drain, daemon=True).start()

    env = os.environ.copy()
    env["COORD_ADDR"] = coord_addr
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    common = ["--iters", str(args.iters)]
    ping_cmd = [sys.executable, "-u", os.path.join(HERE, "demo_coord_ping.py"), *common]
    pong_cmd = [sys.executable, "-u", os.path.join(HERE, "demo_coord_pong.py"), *common]

    print(f"[orchestrator] launching ping: {' '.join(ping_cmd)}", flush=True)
    print(f"[orchestrator] launching pong: {' '.join(pong_cmd)}", flush=True)

    # Coordinator was started with --initial-holder ping, so startup order
    # between ping and pong no longer matters.
    ping = subprocess.Popen(ping_cmd, env=env)
    pong = subprocess.Popen(pong_cmd, env=env)

    rc_ping = ping.wait()
    rc_pong = pong.wait()

    print(f"[orchestrator] ping rc={rc_ping}, pong rc={rc_pong}", flush=True)

    # Shut coord down.
    try:
        coord.send_signal(signal.SIGTERM)
        coord.wait(timeout=3)
    except subprocess.TimeoutExpired:
        coord.kill()

    sys.exit(0 if (rc_ping == 0 and rc_pong == 0) else 1)


if __name__ == "__main__":
    main()
