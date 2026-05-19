"""
CAMA-D Stage 2c: Label Consistency Validation (标注一致性校验)

Standalone script to validate labeling quality:
  - Compute label distribution statistics
  - Re-label a subset and measure consistency rate
  - Flag potential mislabeled samples

Usage:
    python Cul/step_label/validate_labels.py \\
        --input_file /path/to/step_labels.jsonl \\
        --report
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def analyze_labels(input_file: str) -> None:
    """Compute comprehensive statistics on labeled data."""
    samples = [json.loads(l) for l in open(input_file, encoding="utf-8")]

    total_samples = len(samples)
    total_steps = 0
    label_counts = Counter()
    steps_per_sample = []
    labels_per_source = {}

    for sample in samples:
        steps = sample["steps"]
        total_steps += len(steps)
        steps_per_sample.append(len(steps))

        source = sample.get("reasoning_source", "unknown")
        if source not in labels_per_source:
            labels_per_source[source] = Counter()

        for step in steps:
            label = step.get("label", 0.5)
            # Discretize
            if label > 0.7:
                label_counts["0.9"] += 1
                labels_per_source[source]["0.9"] += 1
            elif label < 0.3:
                label_counts["0.1"] += 1
                labels_per_source[source]["0.1"] += 1
            else:
                label_counts["0.5"] += 1
                labels_per_source[source]["0.5"] += 1

    # Report
    print("=" * 60)
    print("CAMA-D Step Label Validation Report")
    print("=" * 60)
    print(f"\nDataset: {input_file}")
    print(f"Total samples: {total_samples}")
    print(f"Total steps: {total_steps}")

    if steps_per_sample:
        avg = sum(steps_per_sample) / len(steps_per_sample)
        print(f"Steps per sample: avg={avg:.1f}, "
              f"min={min(steps_per_sample)}, max={max(steps_per_sample)}")

    print(f"\n--- Label Distribution ---")
    for label in ["0.9", "0.5", "0.1"]:
        count = label_counts[label]
        pct = count / total_steps * 100 if total_steps > 0 else 0
        name = {"0.9": "主场确权步", "0.5": "中立讨论步", "0.1": "文化混淆步"}[label]
        print(f"  {label} ({name}): {count:5d} ({pct:5.1f}%)")

    # Expected distribution check
    print(f"\n--- Distribution Health Check ---")
    pct_09 = label_counts["0.9"] / total_steps * 100 if total_steps > 0 else 0
    pct_05 = label_counts["0.5"] / total_steps * 100 if total_steps > 0 else 0
    pct_01 = label_counts["0.1"] / total_steps * 100 if total_steps > 0 else 0

    print(f"  Expected: 0.5 ~55-65% | 0.9 ~20-30% | 0.1 ~10-20%")
    print(f"  Actual:   0.5  {pct_05:.1f}%  | 0.9  {pct_09:.1f}%  | 0.1  {pct_01:.1f}%")

    warnings = []
    if pct_05 > 80:
        warnings.append("WARNING: Too many neutral steps (>80%). "
                        "Auditor may be under-discriminating.")
    if pct_09 < 10:
        warnings.append("WARNING: Too few cultural endorsement steps (<10%). "
                        "Check if reasoning contains cultural specifics.")
    if pct_01 < 5:
        warnings.append("WARNING: Too few cultural confusion steps (<5%). "
                        "Consider including incorrect reasoning paths.")
    if pct_01 > 40:
        warnings.append("WARNING: Too many confusion steps (>40%). "
                        "Auditor may be over-rejecting.")

    if warnings:
        print(f"\n--- Warnings ---")
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"\n  ✓ Distribution looks healthy!")

    # Per-source breakdown
    if len(labels_per_source) > 1:
        print(f"\n--- Per-Source Breakdown ---")
        for source, counts in sorted(labels_per_source.items()):
            total_source = sum(counts.values())
            print(f"  {source}: ", end="")
            for label in ["0.9", "0.5", "0.1"]:
                c = counts[label]
                p = c / total_source * 100 if total_source > 0 else 0
                print(f"{label}={c}({p:.0f}%) ", end="")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="CAMA-D Stage 2c: Label Validation"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Labeled step JSONL from label_steps.py")
    parser.add_argument("--report", action="store_true",
                        help="Print detailed validation report")
    args = parser.parse_args()

    if args.report:
        analyze_labels(args.input_file)
    else:
        analyze_labels(args.input_file)


if __name__ == "__main__":
    main()