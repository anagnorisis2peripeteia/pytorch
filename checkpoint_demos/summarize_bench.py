"""Summarize a bench_baton CSV as an ASCII table.

Reads the CSV produced by run_bench.py and prints median / p95 / stddev of
``elapsed_ms`` grouped by (sweep, config_id, role, phase), excluding iter 0
(driver warmup) unless ``--include-warmup`` is passed.
"""

import argparse
import csv
import statistics
from collections import defaultdict


def _p95(xs):
    s = sorted(xs)
    if not s:
        return float("nan")
    # nearest-rank
    k = max(0, int(round(0.95 * (len(s) - 1))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="bench_results.csv")
    ap.add_argument(
        "--include-warmup",
        action="store_true",
        help="Include iter 0 (otherwise treated as warmup).",
    )
    args = ap.parse_args()

    # key -> list of elapsed_ms
    groups = defaultdict(list)
    group_order = []  # preserve first-seen ordering of keys
    for row in csv.DictReader(open(args.csv)):
        it = int(row["iter"])
        if it == 0 and not args.include_warmup:
            continue
        key = (row["sweep"], row["config_id"], row["role"], row["phase"])
        if key not in groups:
            group_order.append(key)
        groups[key].append(float(row["elapsed_ms"]))

    # Column widths
    sweep_w = max(5, max(len(k[0]) for k in group_order))
    cfg_w = max(10, max(len(k[1]) for k in group_order))
    role_w = 4
    phase_w = 10

    header = (
        f"{'sweep':<{sweep_w}}  {'config':<{cfg_w}}  "
        f"{'role':<{role_w}}  {'phase':<{phase_w}}  "
        f"{'n':>3}  {'median_ms':>10}  {'p95_ms':>10}  {'stddev_ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for key in group_order:
        xs = groups[key]
        med = statistics.median(xs)
        p95 = _p95(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        sweep, cfg, role, phase = key
        print(
            f"{sweep:<{sweep_w}}  {cfg:<{cfg_w}}  "
            f"{role:<{role_w}}  {phase:<{phase_w}}  "
            f"{len(xs):>3}  {med:>10.2f}  {p95:>10.2f}  {sd:>10.2f}"
        )


if __name__ == "__main__":
    main()
