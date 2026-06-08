"""
Convert Social Chemistry 101 dataset (TSV) to HF-CAC MAS format (JSON).

Social Chemistry 101 is a large-scale corpus of social/moral norms expressed
as Rules-of-Thumb (RoTs). Each entry contains a situation, an action, and
a moral judgment score (action-moral-judgment: -2 to +2).

We convert it to NormAD-style 3-way classification:
  - 1 = socially acceptable (action-moral-judgment >= 1)
  - 2 = socially unacceptable (action-moral-judgment <= -1)
  - 3 = neutral/indeterminate (action-moral-judgment == 0)

Key design decisions:
  - Use only "agency" samples (actor performs the action themselves), skip
    "experience" samples for cleaner evaluation.
  - Use only train/dev/test splits (skip analysis/extra splits).
  - Require valid action-moral-judgment (skip NaN).
  - The "country" field uses rot-moral-foundations as the cultural/normative
    dimension, since Social Chemistry is not country-based but rather
    organized by moral foundations (care-harm, fairness-cheating, etc.).
  - Input combines: moral foundation context + situation + action to judge.

Input:  Cul/data/social-chem-101.tsv
Output: Cul/data/social_mas.json

Usage:
    python Cul/scripts/convert_social_chem.py
"""

import json
import os
import math
import random
from collections import Counter

import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Cul/
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "social-chem-101.tsv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "social_mas.json")

# Moral judgment mapping: 5-point scale -> 3-way classification
# -2 (very bad), -1 (bad) -> 2 (unacceptable)
#  0 (neutral)            -> 3 (neutral/indeterminate)
# +1 (good), +2 (very good) -> 1 (acceptable)
def map_judgment(score: float) -> str:
    """Map 5-point moral judgment to 3-way classification."""
    if score >= 1:
        return "1"  # acceptable
    elif score <= -1:
        return "2"  # unacceptable
    else:
        return "3"  # neutral/indeterminate


# Moral foundation display names (for building context)
MORAL_FOUNDATION_NAMES = {
    "care-harm": "Care/Harm",
    "fairness-cheating": "Fairness/Cheating",
    "loyalty-betrayal": "Loyalty/Betrayal",
    "authority-subversion": "Authority/Subversion",
    "sanctity-degradation": "Sanctity/Degradation",
}


def get_primary_foundation(foundations_str: str) -> str:
    """Extract the primary (first) moral foundation from a pipe-separated string."""
    if not foundations_str or pd.isna(foundations_str):
        return "general"
    return foundations_str.split("|")[0].strip()


# Instruction text (NormAD-style, adapted for social norms without country)
INSTRUCTION = (
    "Read the following social situation and action carefully. "
    "Determine whether the described action is socially acceptable "
    "based on common social norms and moral reasoning. "
    "Answer with a single number only:\n"
    "1 = acceptable\n"
    "2 = unacceptable\n"
    "3 = neutral/indeterminate\n"
    "Do not provide any explanation. Only output the number."
)


def build_input(row: pd.Series) -> str:
    """
    Build input text from a Social Chemistry row.

    Format:
      Moral Foundation: <foundation>
      Situation: <situation>
      Action: <action>
      Rule of Thumb: <rot>
    """
    foundation = get_primary_foundation(row.get("rot-moral-foundations", ""))
    foundation_display = MORAL_FOUNDATION_NAMES.get(foundation, foundation.replace("-", "/").title())

    situation = str(row["situation"]).strip()
    action = str(row["action"]).strip()
    rot = str(row["rot"]).strip()

    return (
        f"Moral Foundation: {foundation_display}\n\n"
        f"Situation:\n{situation}\n\n"
        f"Action to judge:\n{action}\n\n"
        f"Rule of Thumb:\n{rot}"
    )


def convert_social_chem(seed: int = 42):
    """Convert Social Chemistry 101 TSV to MAS format."""
    print("=" * 60)
    print("Converting Social Chemistry 101 dataset")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    # Load data
    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"  Loaded {len(df)} total rows")

    # Filter: only agency actions (not experience)
    df = df[df["action-agency"] == "agency"]
    print(f"  After agency filter: {len(df)} rows")

    # Filter: only standard splits (train/dev/test)
    df = df[df["split"].isin(["train", "dev", "test"])]
    print(f"  After split filter (train/dev/test): {len(df)} rows")

    # Filter: valid moral judgment score
    df = df[df["action-moral-judgment"].notna()]
    print(f"  After valid judgment filter: {len(df)} rows")

    # Filter: non-empty situation and action
    df = df[df["situation"].notna() & df["action"].notna() & df["rot"].notna()]
    df = df[df["situation"].str.strip().ne("") & df["action"].str.strip().ne("")]
    print(f"  After non-empty filter: {len(df)} rows")

    # Convert
    results = []
    for _, row in df.iterrows():
        output = map_judgment(row["action-moral-judgment"])
        foundation = get_primary_foundation(row.get("rot-moral-foundations", ""))

        converted = {
            "instruction": INSTRUCTION,
            "input": build_input(row),
            "output": output,
            "country": foundation,  # Use moral foundation as the "country" dimension
        }
        results.append(converted)

    # Shuffle for balanced sampling
    random.seed(seed)
    random.shuffle(results)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nConverted {len(results)} samples successfully.")
    print(f"Saved to: {OUTPUT_FILE}")

    # Label distribution
    label_dist = Counter(r["output"] for r in results)
    print(f"\nLabel distribution:")
    print(f"  1 (acceptable):      {label_dist.get('1', 0)} "
          f"({label_dist.get('1', 0)/len(results)*100:.1f}%)")
    print(f"  2 (unacceptable):    {label_dist.get('2', 0)} "
          f"({label_dist.get('2', 0)/len(results)*100:.1f}%)")
    print(f"  3 (neutral):         {label_dist.get('3', 0)} "
          f"({label_dist.get('3', 0)/len(results)*100:.1f}%)")

    # "Country" (moral foundation) distribution
    country_dist = Counter(r["country"] for r in results)
    print(f"\nMoral Foundation distribution ({len(country_dist)} foundations):")
    for foundation, count in country_dist.most_common():
        print(f"  {foundation}: {count} ({count/len(results)*100:.1f}%)")

    # Print samples
    if results:
        print("\n--- Sample (first 3 entries) ---")
        for i, sample in enumerate(results[:3]):
            print(f"\n[{i+1}]")
            print(json.dumps(sample, ensure_ascii=False, indent=2))

    return results


if __name__ == "__main__":
    convert_social_chem()
