"""
Step 0: Split MAS inference data into PRM train / PRM val / GRPO train sets.
Split ratio 5:2:3 by question dimension (no question appears in two splits).

Usage:
    python Cul/prm/split_dataset.py \
        --input_file /autodl-fs/data/CulturalBench_mas_inference_20260510_192023.jsonl \
        --output_dir /autodl-fs/data/splits \
        --seed 42
"""

import json
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="MAS inference jsonl file (1227 samples)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save split files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = [json.loads(l) for l in open(args.input_file, encoding="utf-8")]
    print(f"Loaded {len(data)} samples")

    # Shuffle and split 5:2:3
    indices = list(range(len(data)))
    random.shuffle(indices)
    n = len(data)
    n_prm_train = int(n * 0.5)
    n_prm_val   = int(n * 0.2)
    # rest goes to GRPO

    splits = {
        "prm_train": [data[i] for i in indices[:n_prm_train]],
        "prm_val":   [data[i] for i in indices[n_prm_train:n_prm_train + n_prm_val]],
        "grpo_train":[data[i] for i in indices[n_prm_train + n_prm_val:]],
    }

    for name, subset in splits.items():
        out = Path(args.output_dir) / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(subset)} samples → {out}")


if __name__ == "__main__":
    main()
