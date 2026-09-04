#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from trace_format import read_meta, read_trace


TARGETS = (0.50, 0.80, 0.90, 0.95, 0.99)
WINDOWS = (2, 4, 8, 16, 32, 64, 128)


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), q))


def working_set(counts: np.ndarray) -> dict[str, int]:
    ranked = np.sort(counts[counts > 0])[::-1]
    if not len(ranked):
        return {f"W{int(target * 100)}": 0 for target in TARGETS}
    cumulative = np.cumsum(ranked)
    total = cumulative[-1]
    return {
        f"W{int(target * 100)}": int(np.searchsorted(cumulative, total * target, side="left") + 1)
        for target in TARGETS
    }


def entropy(counts: np.ndarray) -> tuple[float, float]:
    nz = counts[counts > 0].astype(np.float64)
    if not len(nz):
        return 0.0, 0.0
    p = nz / nz.sum()
    bits = float(-(p * np.log2(p)).sum())
    return bits, bits / math.log2(len(counts))


def trace_cube(records: np.ndarray, phase: int, n_layers: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    selected = records[records["phase"] == phase]
    tokens = np.unique(selected["token_index"])
    token_pos = {int(token): i for i, token in enumerate(tokens)}
    cube = np.full((len(tokens), n_layers, top_k), -1, dtype=np.int16)
    weights = np.zeros((len(tokens), n_layers, top_k), dtype=np.float32)
    for record in selected:
        token = token_pos[int(record["token_index"])]
        layer = int(record["layer_id"])
        cube[token, layer] = record["expert_ids"]
        weights[token, layer] = record["weights"]
    if len(cube) and np.any(cube < 0):
        missing = int(np.count_nonzero(cube < 0))
        raise ValueError(f"trace cube has {missing} missing expert ids")
    return cube, weights


def adjacent_overlap(cube: np.ndarray) -> dict:
    top_k = cube.shape[2]
    histogram = np.zeros(top_k + 1, dtype=np.int64)
    per_layer_mean = []
    per_layer_histogram = []
    for layer in range(cube.shape[1]):
        values = []
        layer_histogram = np.zeros(top_k + 1, dtype=np.int64)
        for token in range(1, cube.shape[0]):
            overlap = len(set(map(int, cube[token - 1, layer])) & set(map(int, cube[token, layer])))
            histogram[overlap] += 1
            layer_histogram[overlap] += 1
            values.append(overlap)
        per_layer_mean.append(float(np.mean(values)) if values else 0.0)
        layer_total = int(layer_histogram.sum())
        per_layer_histogram.append(
            {
                "layer": layer,
                "histogram": {str(i): int(layer_histogram[i]) for i in range(top_k + 1)},
                "distribution": {str(i): float(layer_histogram[i] / layer_total) if layer_total else 0.0 for i in range(top_k + 1)},
            }
        )
    total = int(histogram.sum())
    return {
        "histogram": {str(i): int(histogram[i]) for i in range(top_k + 1)},
        "distribution": {str(i): float(histogram[i] / total) if total else 0.0 for i in range(top_k + 1)},
        "mean_intersection": float(sum(i * histogram[i] for i in range(top_k + 1)) / total) if total else 0.0,
        "per_layer_mean_intersection": per_layer_mean,
        "per_layer": per_layer_histogram,
    }


def window_locality(cube: np.ndarray) -> dict:
    result = {}
    top_k = cube.shape[2]
    for window in WINDOWS:
        if cube.shape[0] < window:
            result[str(window)] = {"samples": 0}
            continue
        distinct = []
        repeated_fraction = []
        per_layer = []
        for layer in range(cube.shape[1]):
            layer_distinct = []
            layer_repeated = []
            for start in range(cube.shape[0] - window + 1):
                count = len(np.unique(cube[start : start + window, layer]))
                distinct.append(count)
                repeated_fraction.append(1.0 - count / (window * top_k))
                layer_distinct.append(count)
                layer_repeated.append(1.0 - count / (window * top_k))
            per_layer.append(
                {
                    "layer": layer,
                    "distinct_experts_mean": float(np.mean(layer_distinct)),
                    "distinct_experts_p50": percentile(layer_distinct, 50),
                    "distinct_experts_p95": percentile(layer_distinct, 95),
                    "repeated_request_fraction_mean": float(np.mean(layer_repeated)),
                }
            )
        result[str(window)] = {
            "samples": len(distinct),
            "distinct_experts_mean": float(np.mean(distinct)),
            "distinct_experts_p50": percentile(distinct, 50),
            "distinct_experts_p95": percentile(distinct, 95),
            "repeated_request_fraction_mean": float(np.mean(repeated_fraction)),
            "per_layer": per_layer,
        }
    return result


def reuse_distance(cube: np.ndarray) -> dict:
    last_request: dict[tuple[int, int], int] = {}
    last_token: dict[tuple[int, int], int] = {}
    request_distances: list[int] = []
    token_distances: list[int] = []
    first = 0
    request_index = 0
    per_layer_request_index = [0] * cube.shape[1]
    per_layer_last_request = [dict() for _ in range(cube.shape[1])]
    per_layer_last_token = [dict() for _ in range(cube.shape[1])]
    per_layer_request_distances = [[] for _ in range(cube.shape[1])]
    per_layer_token_distances = [[] for _ in range(cube.shape[1])]
    for token in range(cube.shape[0]):
        for layer in range(cube.shape[1]):
            for expert in cube[token, layer]:
                key = (layer, int(expert))
                if key in last_request:
                    request_distances.append(request_index - last_request[key])
                    token_distances.append(token - last_token[key])
                else:
                    first += 1
                last_request[key] = request_index
                last_token[key] = token
                request_index += 1
                local_request = per_layer_request_index[layer]
                if int(expert) in per_layer_last_request[layer]:
                    per_layer_request_distances[layer].append(local_request - per_layer_last_request[layer][int(expert)])
                    per_layer_token_distances[layer].append(token - per_layer_last_token[layer][int(expert)])
                per_layer_last_request[layer][int(expert)] = local_request
                per_layer_last_token[layer][int(expert)] = token
                per_layer_request_index[layer] += 1
    def describe(values: list[int]) -> dict:
        return {
            "count": len(values),
            "p50": percentile(values, 50),
            "p80": percentile(values, 80),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "mean": float(np.mean(values)) if values else None,
        }
    return {
        "first_touch_requests": first,
        "repeat_requests": len(request_distances),
        "expert_request_distance": describe(request_distances),
        "token_distance": describe(token_distances),
        "per_layer": [
            {
                "layer": layer,
                "expert_request_distance": describe(per_layer_request_distances[layer]),
                "token_distance": describe(per_layer_token_distances[layer]),
            }
            for layer in range(cube.shape[1])
        ],
    }


def frequency_and_working_set(cube: np.ndarray) -> dict:
    layers = cube.shape[1]
    global_counts = np.zeros(layers * 512, dtype=np.int64)
    per_layer = []
    for layer in range(layers):
        counts = np.bincount(cube[:, layer].reshape(-1), minlength=512)
        global_counts[layer * 512 : (layer + 1) * 512] = counts
        bits, normalized = entropy(counts)
        per_layer.append(
            {
                "layer": layer,
                "distinct_experts": int(np.count_nonzero(counts)),
                "entropy_bits": bits,
                "entropy_normalized": normalized,
                "working_set": working_set(counts),
                "top_experts": [
                    {"expert": int(expert), "requests": int(counts[expert]), "fraction": float(counts[expert] / counts.sum())}
                    for expert in np.argsort(-counts)[:20]
                    if counts[expert] > 0
                ],
            }
        )
    bits, normalized = entropy(global_counts)
    return {
        "global": {
            "distinct_pages": int(np.count_nonzero(global_counts)),
            "entropy_bits": bits,
            "entropy_normalized": normalized,
            "working_set": working_set(global_counts),
        },
        "per_layer": per_layer,
        "counts": global_counts,
    }


def workload_comparison(counts: dict[str, np.ndarray]) -> list[dict]:
    def compare(a: np.ndarray, b: np.ndarray) -> dict:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        pa = a / max(a.sum(), 1)
        pb = b / max(b.sum(), 1)
        mean = (pa + pb) / 2
        mask_a = pa > 0
        mask_b = pb > 0
        js = 0.5 * float((pa[mask_a] * np.log2(pa[mask_a] / mean[mask_a])).sum())
        js += 0.5 * float((pb[mask_b] * np.log2(pb[mask_b] / mean[mask_b])).sum())
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        hot_a = set(np.argsort(-a)[: max(1, int(np.count_nonzero(a) * 0.1))].tolist())
        hot_b = set(np.argsort(-b)[: max(1, int(np.count_nonzero(b) * 0.1))].tolist())
        return {
            "frequency_cosine": float(np.dot(a, b) / denom) if denom else 0.0,
            "jensen_shannon_bits": js,
            "top_10pct_jaccard": len(hot_a & hot_b) / len(hot_a | hot_b),
        }

    rows = []
    names = sorted(counts)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = counts[left]
            b = counts[right]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    **compare(a, b),
                    "per_layer": [
                        {"layer": layer, **compare(a[layer * 512 : (layer + 1) * 512], b[layer * 512 : (layer + 1) * 512])}
                        for layer in range(len(a) // 512)
                    ],
                }
            )
    return rows


def analyze(trace_path: Path) -> tuple[dict, np.ndarray]:
    header, records = read_trace(trace_path)
    meta_path = trace_path.with_suffix(".meta.json")
    meta = read_meta(meta_path) if meta_path.exists() else {}
    cube, weights = trace_cube(records, 1, header.n_layers, header.top_k)
    frequencies = frequency_and_working_set(cube)
    weight_sums = weights.sum(axis=-1)
    result = {
        "trace": str(trace_path),
        "metadata": meta,
        "decode_tokens": int(cube.shape[0]),
        "layers": header.n_layers,
        "top_k": header.top_k,
        "expert_requests": int(cube.size),
        "routing_weight_sum_error_max": float(np.max(np.abs(weight_sums - 1.0))) if len(cube) else None,
        "frequency": {key: value for key, value in frequencies.items() if key != "counts"},
        "adjacent_token_overlap": adjacent_overlap(cube),
        "window_locality": window_locality(cube),
        "reuse_distance": reuse_distance(cube),
    }
    return result, frequencies["counts"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    analyses = []
    counts = {}
    for path in args.traces:
        result, frequency = analyze(path)
        name = result["metadata"].get("workload", path.stem)
        analyses.append(result)
        counts[name] = counts.get(name, np.zeros_like(frequency)) + frequency
    output = {
        "format": "slotstream-routing-summary-v1",
        "source": "real decode routing traces",
        "traces": analyses,
        "cross_workload": workload_comparison(counts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "traces": len(analyses), "decode_tokens": sum(x["decode_tokens"] for x in analyses)}, indent=2))


if __name__ == "__main__":
    main()
