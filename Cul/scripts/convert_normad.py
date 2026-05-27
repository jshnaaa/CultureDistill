"""
Convert raw datasets to HF-CAC MAS format (JSON).

Supported datasets:
  1. NormAD (JSONL) → normad_mas.json
     Input:  Cul/data/normad.jsonl
     Output: Cul/data/normad_mas.json

  2. CultureAtlas (JSON) → cultureAtlas_mas.json
     Input:  Cul/data/cultureAtlas.json
     Output: Cul/data/cultureAtlas_mas.json

Usage:
    python Cul/scripts/convert_normad.py              # Convert NormAD only
    python Cul/scripts/convert_normad.py --atlas      # Convert CultureAtlas only
    python Cul/scripts/convert_normad.py --all        # Convert both
"""

import json
import os
import random
import argparse
from collections import Counter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Cul/
NORMAD_INPUT = os.path.join(PROJECT_ROOT, "data", "normad.jsonl")
NORMAD_OUTPUT = os.path.join(PROJECT_ROOT, "data", "normad_mas.json")
ATLAS_INPUT = os.path.join(PROJECT_ROOT, "data", "cultureAtlas.json")
ATLAS_OUTPUT = os.path.join(PROJECT_ROOT, "data", "cultureAtlas_mas.json")

# Label mapping: yes -> 1, no -> 2, neutral -> 3
LABEL_MAP = {
    "yes": "1",
    "no": "2",
    "neutral": "3",
}

# Fixed instruction (no role assignment, concise task description)
INSTRUCTION = (
    "Read the following cultural background information and scenario carefully. "
    "Determine whether the behavior described in the scenario is socially acceptable "
    "in the specified country's cultural context. "
    "Answer with a single number only:\n"
    "1 = acceptable\n"
    "2 = unacceptable\n"
    "3 = neutral/indeterminate\n"
    "Do not provide any explanation. Only output the number."
)


def build_input(sample: dict) -> str:
    """Concatenate Country + Background + Story into the input field."""
    country = sample["Country"]
    background = sample["Background"].strip()
    story = sample["Story"].strip()

    return (
        f"Country: {country}\n\n"
        f"Cultural Background:\n{background}\n\n"
        f"Scenario:\n{story}"
    )


def convert_normad():
    """Convert NormAD JSONL to MAS format."""
    print("=" * 60)
    print("Converting NormAD dataset")
    print("=" * 60)

    if not os.path.exists(NORMAD_INPUT):
        print(f"ERROR: Input file not found: {NORMAD_INPUT}")
        return

    results = []
    skipped = 0

    with open(NORMAD_INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            gold_label = sample["Gold Label"].strip().lower()
            output = LABEL_MAP.get(gold_label)
            if output is None:
                print(f"WARNING: Unknown label '{gold_label}' at ID={sample.get('ID')}, skipping.")
                skipped += 1
                continue

            converted = {
                "instruction": INSTRUCTION,
                "input": build_input(sample),
                "output": output,
                "country": sample["Country"],
            }
            results.append(converted)

    # Save as JSON array
    os.makedirs(os.path.dirname(NORMAD_OUTPUT), exist_ok=True)
    with open(NORMAD_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(results)} samples successfully.")
    if skipped:
        print(f"Skipped {skipped} samples due to unknown labels.")
    print(f"Saved to: {NORMAD_OUTPUT}")

    # Print label distribution
    label_dist = Counter(r["output"] for r in results)
    print(f"\nLabel distribution:")
    print(f"  1 (acceptable):      {label_dist.get('1', 0)}")
    print(f"  2 (unacceptable):    {label_dist.get('2', 0)}")
    print(f"  3 (neutral):         {label_dist.get('3', 0)}")

    # Print country distribution
    country_dist = Counter(r["country"] for r in results)
    print(f"\nCountry distribution ({len(country_dist)} countries):")
    for country, count in country_dist.most_common():
        print(f"  {country}: {count}")

    # Print a sample for verification
    if results:
        print("\n--- Sample (first entry) ---")
        print(json.dumps(results[0], ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# CultureAtlas conversion
# ---------------------------------------------------------------------------

ATLAS_INSTRUCTION = (
    "Below is a question about a specific country's culture, along with two "
    "candidate responses. One response reflects deeper, more culturally specific "
    "knowledge (e.g., unique traditions, lesser-known practices, or nuanced "
    "cultural significance), while the other is more generic or surface-level.\n\n"
    "Choose the response that demonstrates more culturally specific and insightful "
    "knowledge about the target country. Answer with a single number only:\n"
    "1 = Response 1 is more culturally specific\n"
    "2 = Response 2 is more culturally specific\n"
    "Do not provide any explanation. Only output the number."
)


def convert_cultureatlas(seed: int = 42):
    """Convert CultureAtlas JSON to MAS format with randomized A/B option order."""
    print("\n" + "=" * 60)
    print("Converting CultureAtlas dataset")
    print("=" * 60)

    if not os.path.exists(ATLAS_INPUT):
        print(f"ERROR: Input file not found: {ATLAS_INPUT}")
        return

    random.seed(seed)

    with open(ATLAS_INPUT, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_data = list(raw_data.values())[0]

    results = []
    skipped = 0

    for item in raw_data:
        # Validate required fields
        if not all(k in item for k in ("instruction", "positive_sample",
                                        "negative_sample", "country")):
            skipped += 1
            continue

        question = item["instruction"].strip()
        positive = item["positive_sample"].strip()
        negative = item["negative_sample"].strip()
        country = item["country"].strip()

        if not question or not positive or not negative or not country:
            skipped += 1
            continue

        # Randomly decide order: positive first (output="1") or second (output="2")
        if random.random() < 0.5:
            option_1 = positive
            option_2 = negative
            output = "1"  # positive is Response 1
        else:
            option_1 = negative
            option_2 = positive
            output = "2"  # positive is Response 2

        # Build input field
        input_text = (
            f"Country: {country}\n\n"
            f"Question: {question}\n\n"
            f"Response 1: {option_1}\n\n"
            f"Response 2: {option_2}"
        )

        converted = {
            "instruction": ATLAS_INSTRUCTION,
            "input": input_text,
            "output": output,
            "country": country,
        }
        results.append(converted)

    # Save as JSON array
    os.makedirs(os.path.dirname(ATLAS_OUTPUT), exist_ok=True)
    with open(ATLAS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(results)} samples successfully.")
    if skipped:
        print(f"Skipped {skipped} samples due to missing fields.")
    print(f"Saved to: {ATLAS_OUTPUT}")

    # Print label distribution (should be ~50/50 due to randomization)
    label_dist = Counter(r["output"] for r in results)
    print(f"\nLabel distribution (should be ~50/50):")
    print(f"  1 (positive first):  {label_dist.get('1', 0)} "
          f"({label_dist.get('1', 0)/max(len(results),1)*100:.1f}%)")
    print(f"  2 (positive second): {label_dist.get('2', 0)} "
          f"({label_dist.get('2', 0)/max(len(results),1)*100:.1f}%)")

    # Print country distribution
    country_dist = Counter(r["country"] for r in results)
    print(f"\nCountry distribution ({len(country_dist)} countries):")
    for country, count in country_dist.most_common(10):
        print(f"  {country}: {count}")
    if len(country_dist) > 10:
        print(f"  ... and {len(country_dist) - 10} more countries")

    # Print a sample for verification
    if results:
        print("\n--- Sample (first entry) ---")
        print(json.dumps(results[0], ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw datasets to HF-CAC MAS format"
    )
    parser.add_argument("--normad", action="store_true",
                        help="Convert NormAD dataset only")
    parser.add_argument("--atlas", action="store_true",
                        help="Convert CultureAtlas dataset only")
    parser.add_argument("--all", action="store_true",
                        help="Convert all datasets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for CultureAtlas option shuffling")
    args = parser.parse_args()

    # Default: if no flag specified, convert NormAD (backward compatible)
    if not args.normad and not args.atlas and not args.all:
        args.normad = True

    if args.normad or args.all:
        convert_normad()
    if args.atlas or args.all:
        convert_cultureatlas(seed=args.seed)
