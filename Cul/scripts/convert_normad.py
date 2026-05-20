"""
Convert raw NormAD dataset (JSONL) to HF-CAC MAS format (JSON).

Input:  Cul/data/normad.jsonl  (one JSON object per line)
Output: Cul/data/normad_mas.json (JSON array, SFT-style with instruction/input/output/country)
"""

import json
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Cul/
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "normad.jsonl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "normad_mas.json")

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


def convert():
    results = []
    skipped = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
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
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(results)} samples successfully.")
    if skipped:
        print(f"Skipped {skipped} samples due to unknown labels.")
    print(f"Saved to: {OUTPUT_PATH}")

    # Print label distribution
    from collections import Counter
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


if __name__ == "__main__":
    convert()
