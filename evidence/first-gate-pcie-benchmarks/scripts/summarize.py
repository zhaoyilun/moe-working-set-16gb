#!/usr/bin/env python3
"""Aggregate repeated benchmark rounds without selecting a best run."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


COPY_KEYS = ("pool_gb", "host_memory", "layout", "batch_experts", "bytes", "iterations")
COPY_METRICS = (
    "wall_mean_ms", "wall_p50_ms", "wall_p95_ms", "event_mean_ms", "effective_gbps"
)
OVERLAP_KEYS = ("pool_gb", "batch_experts", "bytes", "pipeline_cycles", "pipeline_rounds")
OVERLAP_METRICS = (
    "copy_only_ms", "compute_only_ms", "serial_ms", "double_buffer_serial_ms",
    "overlap_mean_ms", "overlap_p50_ms", "overlap_p95_ms", "exposed_copy_ms",
    "hidden_copy_percent", "equivalent_gemm_flops",
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(round_dirs, filename, keys, metrics):
    grouped = defaultdict(list)
    for directory in round_dirs:
        for row in read_rows(directory / filename):
            grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        record = dict(zip(keys, key))
        record["rounds"] = len(rows)
        for metric in metrics:
            record[metric] = statistics.median(float(row[metric]) for row in rows)
        output.append(record)
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    copy_rows = aggregate(args.round_dirs, "copy_results.csv", COPY_KEYS, COPY_METRICS)
    overlap_rows = aggregate(args.round_dirs, "overlap_results.csv", OVERLAP_KEYS, OVERLAP_METRICS)
    write_csv(args.output / "aggregate_copy_results.csv", copy_rows)
    write_csv(args.output / "aggregate_overlap_results.csv", overlap_rows)

    pinned = [
        row for row in copy_rows
        if row["host_memory"] == "pinned" and row["layout"] == "contiguous_record"
    ]
    one = [row for row in pinned if int(row["batch_experts"]) == 1]
    sustained = [row for row in pinned if int(row["batch_experts"]) >= 8]
    nine = [
        row for row in copy_rows
        if row["host_memory"] == "pinned"
        and row["layout"] == "nine_pieces_per_expert"
        and int(row["batch_experts"]) >= 8
    ]
    pageable = [
        row for row in copy_rows
        if row["host_memory"] == "pageable"
        and row["layout"] == "contiguous_record"
        and int(row["batch_experts"]) >= 8
    ]
    top10 = next(row for row in overlap_rows if int(row["batch_experts"]) == 10)
    bandwidth = statistics.median(row["effective_gbps"] for row in sustained)
    bytes_per_token = 2_764_800 * 10 * 48
    compute_token_ms = top10["compute_only_ms"] * 48
    hit_rows = []
    for hit_rate in (0, 0.25, 0.50, 0.75, 0.90, 0.95):
        h2d_gb = bytes_per_token * (1 - hit_rate) / 1e9
        h2d_ms = h2d_gb / bandwidth * 1000
        hit_rows.append({
            "hit_rate": hit_rate,
            "h2d_gb_per_token": h2d_gb,
            "h2d_ms_per_token": h2d_ms,
            "raw_h2d_ceiling_tps": 1000 / h2d_ms if h2d_ms else None,
            "surrogate_moe_compute_ms": compute_token_ms,
            "ideal_exposed_copy_ms": max(0, h2d_ms - compute_token_ms),
        })

    summary = {
        "rounds": [str(path) for path in args.round_dirs],
        "single_expert_pinned_contiguous": {
            "p50_ms": statistics.median(row["wall_p50_ms"] for row in one),
            "p95_ms": statistics.median(row["wall_p95_ms"] for row in one),
            "effective_gbps": statistics.median(row["effective_gbps"] for row in one),
        },
        "sustained_pinned_contiguous_batches_8_10_20": {
            "effective_gbps": bandwidth,
            "range": [
                min(row["effective_gbps"] for row in sustained),
                max(row["effective_gbps"] for row in sustained),
            ],
        },
        "sustained_pinned_nine_piece_batches_8_10_20": {
            "effective_gbps": statistics.median(row["effective_gbps"] for row in nine),
        },
        "pageable_contiguous_batches_8_10_20": {
            "effective_gbps": statistics.median(row["effective_gbps"] for row in pageable),
        },
        "top10_overlap": top10,
        "bytes_per_token_at_zero_hit": bytes_per_token,
        "cache_hit_table": hit_rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
