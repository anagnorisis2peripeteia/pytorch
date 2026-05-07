"""Benchmark FlashAttention-2 varlen vs padded SDPA on MPS.

Uses torch.utils.benchmark.Timer + blocked_autorange.
Compares varlen kernel against padded F.scaled_dot_product_attention.

Usage:
    PYTHONPATH=~/pytorch python3 benchmarks/bench_flash_attn_varlen.py
    PYTHONPATH=~/pytorch python3 benchmarks/bench_flash_attn_varlen.py --backward
"""
import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

assert torch.backends.mps.is_available(), "MPS not available"

_fa_op = torch.ops.aten._scaled_dot_product_flash_attention_varlen_for_mps


def fmt(m):
    return f"{m.median * 1e6:>8.1f} +/- {m.iqr * 1e6:>5.1f}"


def bench(stmt, glob, min_run_time=2.0):
    t = Timer(stmt=stmt, globals=glob)
    return t.blocked_autorange(min_run_time=min_run_time)


def make_varlen_tensors(B, H, S, D, dtype, gqa=1):
    kH = H // gqa
    q = torch.randn(B * S, H, D, device="mps", dtype=dtype) * 0.1
    k = torch.randn(B * S, kH, D, device="mps", dtype=dtype) * 0.1
    v = torch.randn(B * S, kH, D, device="mps", dtype=dtype) * 0.1
    sl = torch.tensor([0] + [S] * B, device="mps", dtype=torch.int32)
    cu = torch.cumsum(sl, dim=0, dtype=torch.int32)
    return q, k, v, cu, S


def make_padded_tensors(B, H, S, D, dtype, gqa=1):
    q = torch.randn(B, H, S, D, device="mps", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, D, device="mps", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, D, device="mps", dtype=dtype) * 0.1
    return q, k, v


configs = [
    # label,             B,  H,  S,    D,  gqa
    ("B4 S=512 H32 D128 GQA4", 4, 32, 512, 128, 4),
    ("B1 S=4096 H32 D128 GQA4", 1, 32, 4096, 128, 4),
    ("B1 S=8192 H32 D128 GQA4", 1, 32, 8192, 128, 4),
    ("B4 S=512 H8 D64 GQA1", 4, 8, 512, 64, 1),
    ("B2 S=1024 H32 D128 GQA4", 2, 32, 1024, 128, 4),
    ("B1 S=2048 H32 D128 GQA4", 1, 32, 2048, 128, 4),
]


def run_forward(args):
    print("=" * 80)
    print("FORWARD ONLY")
    print("=" * 80)

    header = f"{'Config':<32} {'varlen (us)':>20} {'padded SDPA (us)':>20} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for label, B, H, S, D, gqa in configs:
        scale = 1.0 / math.sqrt(D)

        q, k, v, cu, ms = make_varlen_tensors(B, H, S, D, torch.float16, gqa)
        r_vl = bench(
            "_fa_op(q, k, v, cu, cu, ms, ms, 0.0, True, scale=scale); torch.mps.synchronize()",
            {"_fa_op": _fa_op, "q": q, "k": k, "v": v, "cu": cu, "ms": ms,
             "scale": scale, "torch": torch},
            min_run_time=args.min_time,
        )

        qp, kp, vp = make_padded_tensors(B, H, S, D, torch.float16, gqa)
        r_pad = bench(
            "F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale); torch.mps.synchronize()",
            {"F": F, "q": qp, "k": kp, "v": vp, "scale": scale, "torch": torch},
            min_run_time=args.min_time,
        )

        speedup = r_pad.median / r_vl.median
        print(f"{label:<32} {fmt(r_vl):>20} {fmt(r_pad):>20} {speedup:>7.2f}x")


def run_fwd_bwd(args):
    print()
    print("=" * 80)
    print("FORWARD + BACKWARD")
    print("=" * 80)

    header = f"{'Config':<32} {'varlen (us)':>20} {'padded SDPA (us)':>20} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for label, B, H, S, D, gqa in configs:
        scale = 1.0 / math.sqrt(D)

        q, k, v, cu, ms = make_varlen_tensors(B, H, S, D, torch.float16, gqa)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        r_vl = bench(
            "out, _ = _fa_op(q, k, v, cu, cu, ms, ms, 0.0, True, scale=scale)\n"
            "out.sum().backward()\n"
            "q.grad = k.grad = v.grad = None\n"
            "torch.mps.synchronize()",
            {"_fa_op": _fa_op, "q": q, "k": k, "v": v, "cu": cu, "ms": ms,
             "scale": scale, "torch": torch},
            min_run_time=args.min_time,
        )

        qp, kp, vp = make_padded_tensors(B, H, S, D, torch.float16, gqa)
        qp.requires_grad_(True)
        kp.requires_grad_(True)
        vp.requires_grad_(True)

        try:
            r_pad = bench(
                "out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)\n"
                "out.sum().backward()\n"
                "q.grad = k.grad = v.grad = None\n"
                "torch.mps.synchronize()",
                {"F": F, "q": qp, "k": kp, "v": vp, "scale": scale, "torch": torch},
                min_run_time=args.min_time,
            )
            speedup = r_pad.median / r_vl.median
            print(f"{label:<32} {fmt(r_vl):>20} {fmt(r_pad):>20} {speedup:>7.2f}x")
        except Exception as e:
            print(f"{label:<32} {fmt(r_vl):>20} {'BASELINE FAILS':>20} {'N/A':>8}")
            print(f"  Baseline error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MPS FlashAttention-2 varlen vs padded SDPA")
    parser.add_argument("--min-time", type=float, default=2.0)
    parser.add_argument("--backward", action="store_true",
                        help="Include forward+backward timing")
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}")
    print(f"Benchmark: torch.utils.benchmark.Timer + blocked_autorange")
    print(f"dtype: float16")
    print()

    run_forward(args)
    if args.backward:
        run_fwd_bwd(args)


if __name__ == "__main__":
    main()

