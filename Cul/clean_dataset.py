"""
Clean a JSON dataset by keeping only specified fields.

Usage:
    python Cul/clean_dataset.py \
        --input_file Cul/data/CulturalBench_merge_gen.json \
        --output_file Cul/data/CulturalBench_mas.json \
        --keep_fields instruction input output
"""

import json
import argparse
from pathlib import Path


def clean_dataset(input_file, output_file, keep_fields):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = [
        {k: item[k] for k in keep_fields if k in item}
        for item in data
    ]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(cleaned)} samples saved to {output_file}")
    print(f"Fields kept: {keep_fields}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--keep_fields", nargs="+", default=["instruction", "input", "output"])
    args = parser.parse_args()

    clean_dataset(args.input_file, args.output_file, args.keep_fields)
