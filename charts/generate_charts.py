from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "results"
OUT = ROOT / "charts"

COLORS = {4: "#9fb3c8", 6: "#4f86c6", 8: "#0b4f6c"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finish(path: Path) -> None:
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def decode_progression() -> None:
    rows = read_csv("decode_progression.csv")
    rows = [row for row in rows if row["configuration"] != "Hybrid fixed cache" or row["cuda_graph"] == "on"]
    labels = []
    values = []
    for row in rows:
        if row["configuration"] == "Hybrid fixed cache":
            labels.append(f"Hybrid\n{row['expert_pool_gib']} GiB")
        elif row["configuration"] == "Native expert-only CPU offload":
            labels.append("Native\nsame path")
        else:
            labels.append("Golden Native\nhistorical")
        values.append(float(row["decode_tok_s_mean"]))

    plt.figure(figsize=(8.8, 4.8))
    bars = plt.bar(labels, values, color=["#64748b", "#9fb3c8", "#4f86c6", "#0b4f6c", "#c89b3c"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}", ha="center", fontsize=9)
    plt.ylabel("Decode tokens/s")
    plt.title("Decode progression (Measured)")
    plt.ylim(0, max(values) * 1.18)
    finish(OUT / "decode_progression.png")


def context_charts() -> None:
    rows = read_csv("context_pool_matrix.csv")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["pool_gib"])].append(row)

    for metric, ylabel, filename, scale in (
        ("decode_tok_s", "Decode tokens/s", "context_vs_decode.png", 1.0),
        ("hit_rate", "Fixed-cache hit rate (%)", "context_vs_hit_rate.png", 100.0),
    ):
        plt.figure(figsize=(8.4, 4.8))
        for pool in (4, 6, 8):
            series = sorted(grouped[pool], key=lambda row: int(row["actual_context_tokens"]))
            x = [int(row["actual_context_tokens"]) for row in series]
            y = [float(row[metric]) * scale for row in series]
            plt.plot(x, y, marker="o", linewidth=2.2, color=COLORS[pool], label=f"{pool} GiB")
        plt.xlabel("Actual context tokens")
        plt.ylabel(ylabel)
        plt.title(f"Context vs {ylabel} (Measured)")
        plt.xticks([2027, 8171, 16363, 32747], ["2K", "8K", "16K", "32K"])
        plt.legend(frameon=False)
        finish(OUT / filename)


def prefill_chart() -> None:
    rows = [
        row
        for row in read_csv("prefill_ubatch_sweep.csv")
        if row["series"] == "prefill_v2_same_prompt" and row["context"] == "16K"
    ]
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["n_ubatch"])].append(float(row["prefill_tok_s"]))
    x = sorted(grouped)
    y = [mean(grouped[value]) for value in x]
    low = [value - min(grouped[key]) for key, value in zip(x, y)]
    high = [max(grouped[key]) - value for key, value in zip(x, y)]

    plt.figure(figsize=(7.6, 4.8))
    plt.errorbar(x, y, yerr=[low, high], marker="o", linewidth=2.4, capsize=5, color="#0b4f6c")
    for ubatch, value in zip(x, y):
        plt.text(ubatch, value + 14, f"{value:.1f}", ha="center")
    plt.axhline(500, color="#c2410c", linestyle="--", linewidth=1.4, label="500 tok/s gate")
    plt.xlabel("Physical ubatch")
    plt.ylabel("Cold Prefill tokens/s")
    plt.title("16K Prefill scaling, same prompt (Measured)")
    plt.xticks(x, ["2K", "4K", "8K"])
    plt.legend(frameon=False)
    plt.ylim(0, max(y) * 1.18)
    finish(OUT / "prefill_ubatch_scaling.png")


def threshold_chart() -> None:
    rows = read_csv("selective_h2d_thresholds.csv")
    hit = [float(row["hit_rate"]) * 100 for row in rows]
    speed = [float(row["decode_tok_s"]) for row in rows]
    thresholds = [row["threshold"] for row in rows]

    plt.figure(figsize=(7.6, 4.8))
    plt.plot(hit, speed, marker="o", linewidth=2.2, color="#9f1239")
    for x, y, threshold in zip(hit, speed, thresholds):
        plt.annotate(f"T{threshold}", (x, y), xytext=(5, 6), textcoords="offset points")
    plt.xlabel("Runtime hit rate (%)")
    plt.ylabel("Decode tokens/s")
    plt.title("Selective H2D: higher hit rate, lower throughput")
    finish(OUT / "selective_h2d_tradeoff.png")


if __name__ == "__main__":
    decode_progression()
    context_charts()
    prefill_chart()
    threshold_chart()
