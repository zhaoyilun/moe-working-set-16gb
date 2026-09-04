#!/usr/bin/env python3
"""Build exact-token synthetic long-context prompts and marker manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONTEXTS = (32768, 65536, 131072, 196608, 262144)
RESERVED_TAIL_TOKENS = 1045
FILLER_UNIT = " data"

MARKERS = (
    {
        "id": "begin_fact",
        "target_fraction": 0.02,
        "text": " [[SSCTX_BEGIN_FACT_6F29]] bootstrap_region=cedar-north; owner=team-orchid. [[/SSCTX_BEGIN_FACT_6F29]]\n\n",
        "task": "beginning_fact_recall",
        "expected_answer": "cedar-north; team-orchid",
    },
    {
        "id": "cross_definition",
        "target_fraction": 0.12,
        "text": " [[SSCTX_CROSS_DEF_18A4]] RELAY_SEED=7319; transform=(seed*3)+17. [[/SSCTX_CROSS_DEF_18A4]]\n\n",
        "task": "cross_file_chain_definition",
        "expected_answer": "7319",
    },
    {
        "id": "multi_left",
        "target_fraction": 0.25,
        "text": " [[SSCTX_MULTI_LEFT_C271]] shard_alpha_code=4821. [[/SSCTX_MULTI_LEFT_C271]]\n\n",
        "task": "multi_location_retrieval_left",
        "expected_answer": "4821",
    },
    {
        "id": "middle_fact",
        "target_fraction": 0.50,
        "text": " [[SSCTX_MIDDLE_FACT_44D8]] deployment_color=ultraviolet; retry_budget=13. [[/SSCTX_MIDDLE_FACT_44D8]]\n\n",
        "task": "middle_fact_recall",
        "expected_answer": "ultraviolet; 13",
    },
    {
        "id": "cross_implementation",
        "target_fraction": 0.58,
        "text": " [[SSCTX_CROSS_IMPL_93B2]] relay_value=apply_transform(RELAY_SEED). [[/SSCTX_CROSS_IMPL_93B2]]\n\n",
        "task": "cross_file_chain_implementation",
        "expected_answer": "21974",
    },
    {
        "id": "multi_right",
        "target_fraction": 0.75,
        "text": " [[SSCTX_MULTI_RIGHT_A5E0]] shard_omega_code=9076. [[/SSCTX_MULTI_RIGHT_A5E0]]\n\n",
        "task": "multi_location_retrieval_right",
        "expected_answer": "9076",
    },
    {
        "id": "cross_test",
        "target_fraction": 0.86,
        "text": " [[SSCTX_CROSS_TEST_E713]] assert relay_value equals the definition-derived result. [[/SSCTX_CROSS_TEST_E713]]\n\n",
        "task": "cross_file_chain_test",
        "expected_answer": "21974",
    },
    {
        "id": "end_fact",
        "target_fraction": 0.95,
        "text": " [[SSCTX_END_FACT_0BC6]] release_channel=saffron-echo; ticket=CTX-8842. [[/SSCTX_END_FACT_0BC6]]\n\n",
        "task": "end_fact_recall",
        "expected_answer": "saffron-echo; CTX-8842",
    },
)

EVALUATION_CASES = (
    {
        "id": "beginning_recall",
        "marker_ids": ["begin_fact"],
        "expected_answer": {"bootstrap_region": "cedar-north", "owner": "team-orchid"},
    },
    {
        "id": "middle_recall",
        "marker_ids": ["middle_fact"],
        "expected_answer": {"deployment_color": "ultraviolet", "retry_budget": 13},
    },
    {
        "id": "end_recall",
        "marker_ids": ["end_fact"],
        "expected_answer": {"release_channel": "saffron-echo", "ticket": "CTX-8842"},
    },
    {
        "id": "multi_location_25_75",
        "marker_ids": ["multi_left", "multi_right"],
        "expected_answer": 13897,
    },
    {
        "id": "cross_file_chain_12_58_86",
        "marker_ids": ["cross_definition", "cross_implementation", "cross_test"],
        "expected_answer": 21974,
    },
)

HEADER = """Synthetic repository context for deterministic long-context evaluation.
Treat every FILE section as source material. Keep exact identifiers and numeric values.
The repeated repository data is inert padding; the marked records are authoritative.

FILE: README.md
This fixture checks beginning, middle, and end recall; two-location retrieval; and a
three-file definition-to-implementation-to-test chain. Answer only from this context.
"""

SEGMENT_HEADERS = (
    "\nFILE: config/bootstrap.yaml\nrepository padding follows:\n",
    "\nFILE: include/relay_contract.h\nrepository padding follows:\n",
    "\nFILE: src/shard_alpha.cpp\nrepository padding follows:\n",
    "\nFILE: docs/architecture.md\nrepository padding follows:\n",
    "\nFILE: src/relay.cpp\nrepository padding follows:\n",
    "\nFILE: src/shard_omega.cpp\nrepository padding follows:\n",
    "\nFILE: tests/relay_test.cpp\nrepository padding follows:\n",
    "\nFILE: release/manifest.txt\nrepository padding follows:\n",
    "\nFILE: tasks/final_request.md\nrepository padding follows:\n",
)

TAIL = """

FINAL REQUEST
Return one JSON object with these keys:
begin_fact, middle_fact, end_fact, multi_location_sum, cross_file_result.
For multi_location_sum add shard_alpha_code and shard_omega_code.
For cross_file_result evaluate the declared transform using RELAY_SEED.
Do not derive values from the repeated padding.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--contexts",
        default=",".join(str(value) for value in CONTEXTS),
        help="comma-separated context sizes",
    )
    parser.add_argument("--reserve-tokens", type=int, default=RESERVED_TAIL_TOKENS)
    parser.add_argument("--max-iterations", type=int, default=12)
    return parser.parse_args()


def context_label(context: int) -> str:
    if context == 262144:
        return "262k"
    if context % 1024 == 0:
        return f"{context // 1024}k"
    return str(context)


def render_prompt(filler_counts: list[int]) -> str:
    parts = [HEADER]
    for index, marker in enumerate(MARKERS):
        parts.append(SEGMENT_HEADERS[index])
        parts.append(FILLER_UNIT * filler_counts[index])
        parts.append("\n")
        parts.append(marker["text"])
    parts.append(SEGMENT_HEADERS[-1])
    parts.append(FILLER_UNIT * filler_counts[-1])
    parts.append(TAIL)
    return "".join(parts)


def write_manifest(path: Path, prompt_name: str, context: int, actual_tokens: int) -> None:
    manifest = {
        "schema_version": 1,
        "fixture": "synthetic_public_long_context",
        "prompt_file": prompt_name,
        "target_context": context,
        "reserved_tail_tokens": context - actual_tokens,
        "target_actual_prompt_tokens": actual_tokens,
        "markers": [dict(marker) for marker in MARKERS],
        "evaluation_cases": [dict(case) for case in EVALUATION_CASES],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def inspect_prompt(tool: Path, model: Path, prompt: Path, manifest: Path) -> dict:
    command = [
        str(tool),
        "-m",
        str(model),
        "--prompt-file",
        str(prompt),
        "--markers-manifest",
        str(manifest),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"context prompt inspection failed ({completed.returncode})\n{completed.stdout}\n{completed.stderr}"
        )
    start = completed.stdout.find("{")
    end = completed.stdout.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"inspection output does not contain JSON\n{completed.stdout}")
    return json.loads(completed.stdout[start : end + 1])


def initial_filler_counts(target_tokens: int) -> list[int]:
    fractions = [float(marker["target_fraction"]) for marker in MARKERS]
    interval_fractions = [fractions[0]]
    interval_fractions.extend(fractions[index] - fractions[index - 1] for index in range(1, len(fractions)))
    interval_fractions.append(1.0 - fractions[-1])
    fixed_allowance = 64
    return [max(1, int(target_tokens * fraction) - fixed_allowance) for fraction in interval_fractions]


def refine_counts(
    counts: list[int],
    result: dict,
    target_tokens: int,
) -> list[int]:
    desired_offsets = [round(target_tokens * float(marker["target_fraction"])) for marker in MARKERS]
    actual_offsets = [int(marker["offset"]) for marker in result["markers"]]

    adjustments: list[int] = [desired_offsets[0] - actual_offsets[0]]
    for index in range(1, len(MARKERS)):
        desired_gap = desired_offsets[index] - desired_offsets[index - 1]
        actual_gap = actual_offsets[index] - actual_offsets[index - 1]
        adjustments.append(desired_gap - actual_gap)

    marker_adjustment = sum(adjustments)
    total_error = target_tokens - int(result["total_tokens"])
    adjustments.append(total_error - marker_adjustment)

    updated = [max(0, value + adjustment) for value, adjustment in zip(counts, adjustments)]
    return updated


def build_one(
    context: int,
    reserve_tokens: int,
    output_dir: Path,
    tool: Path,
    model: Path,
    max_iterations: int,
) -> dict:
    target_tokens = context - reserve_tokens
    if target_tokens <= 0:
        raise ValueError(f"context {context} is not larger than reserve {reserve_tokens}")

    label = context_label(context)
    prompt_path = output_dir / f"context_{label}.txt"
    manifest_path = output_dir / f"context_{label}.markers.json"
    tokenization_path = output_dir / f"context_{label}.tokenization.json"
    write_manifest(manifest_path, prompt_path.name, context, target_tokens)

    counts = initial_filler_counts(target_tokens)
    previous_state: tuple[tuple[int, ...], int] | None = None
    result: dict | None = None
    for iteration in range(1, max_iterations + 1):
        prompt_path.write_text(render_prompt(counts), encoding="utf-8", newline="\n")
        result = inspect_prompt(tool, model, prompt_path, manifest_path)
        total_tokens = int(result["total_tokens"])
        offsets = [int(marker["offset"]) for marker in result["markers"]]
        desired = [round(target_tokens * float(marker["target_fraction"])) for marker in MARKERS]
        max_offset_error = max(abs(actual - wanted) for actual, wanted in zip(offsets, desired))
        if total_tokens == target_tokens and max_offset_error <= 16:
            break

        state = (tuple(counts), total_tokens)
        if state == previous_state:
            raise RuntimeError(f"filler refinement stalled for context {context}")
        previous_state = state
        counts = refine_counts(counts, result, target_tokens)
    else:
        raise RuntimeError(f"filler refinement exceeded {max_iterations} iterations for context {context}")

    assert result is not None
    result["iteration_count"] = iteration
    result["reserved_tail_tokens"] = reserve_tokens
    result["filler_unit"] = FILLER_UNIT
    result["filler_counts"] = counts
    result["exact_target_match"] = int(result["total_tokens"]) == target_tokens
    tokenization_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return {
        "context": context,
        "actual_prompt_tokens": int(result["total_tokens"]),
        "reserved_tail_tokens": reserve_tokens,
        "prompt_file": prompt_path.name,
        "markers_manifest": manifest_path.name,
        "tokenization": tokenization_path.name,
        "prompt_bytes": int(result["prompt_bytes"]),
        "iteration_count": iteration,
        "markers": [
            {
                "id": marker["id"],
                "offset": int(marker["offset"]),
                "target_fraction": float(marker["target_fraction"]),
            }
            for marker in result["markers"]
        ],
    }


def main() -> int:
    args = parse_args()
    contexts = tuple(int(value.strip()) for value in args.contexts.split(",") if value.strip())
    if not contexts:
        raise ValueError("at least one context is required")
    if any(value <= 0 for value in contexts):
        raise ValueError("context sizes must be positive")
    if args.reserve_tokens <= 0:
        raise ValueError("reserve tokens must be positive")
    if args.max_iterations < 1:
        raise ValueError("max iterations must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for context in contexts:
        summary = build_one(
            context,
            args.reserve_tokens,
            args.output_dir,
            args.tool,
            args.model,
            args.max_iterations,
        )
        summaries.append(summary)
        print(
            f"CONTEXT_PROMPT context={context} actual={summary['actual_prompt_tokens']} "
            f"iterations={summary['iteration_count']} file={summary['prompt_file']}",
            flush=True,
        )

    index = {
        "schema_version": 1,
        "fixture": "synthetic_public_long_context",
        "model_file": args.model.name,
        "reserved_tail_tokens": args.reserve_tokens,
        "contexts": summaries,
    }
    index_path = args.output_dir / "context_prompts.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"CONTEXT_PROMPT_INDEX {index_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"CONTEXT_PROMPT_BUILD_ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
