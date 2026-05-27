"""
Split step_labels.jsonl into train/val sets for PRM training.

Usage:
    python Cul/step_label/split_step_labels.py \
        --input_file /path/to/normad_step_labels.jsonl \
        --output_dir /path/to/output/ \
        --val_ratio 0.2 \
        --seed 42

Outputs:
    {output_dir}/normad_step_labels_train.jsonl
    {output_dir}/normad_step_labels_val.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Split step-labeled JSONL into train/val for PRM training"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input step_labels.jsonl from label_steps.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input file)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    samples = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {args.input_file}")

    # Shuffle and split
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filenames from input filename
    stem = Path(args.input_file).stem  # e.g., "normad_step_labels"
    train_file = output_dir / f"{stem}_train.jsonl"
    val_file = output_dir / f"{stem}_val.jsonl"

    # Write train
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write val
    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Split complete:")
    print(f"  Train: {len(train_samples)} samples → {train_file}")
    print(f"  Val:   {len(val_samples)} samples → {val_file}")


if __name__ == "__main__":
    main()
