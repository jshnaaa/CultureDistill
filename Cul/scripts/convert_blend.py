"""
Convert BLEnD dataset (CSV) to HF-CAC MAS format (JSON).

BLEnD is a multi-choice cultural QA benchmark with 4 options (A/B/C/D).
We convert it to the same format as CulturalBench:
  - instruction: fixed task instruction
  - input: question + numbered options (1/2/3/4)
  - output: correct answer number (1/2/3/4)
  - country: target country

Input:
  Cul/data/blend/mc_questions_file-1.csv
  Cul/data/blend/mc_questions_file-2.csv

Output:
  Cul/data/blend_mas.json

Usage:
    python Cul/scripts/convert_blend.py
"""

import csv
import json
import os
import re
from collections import Counter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Cul/
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "blend")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "blend_mas.json")

INPUT_FILES = [
    os.path.join(DATA_DIR, "mc_questions_file-1.csv"),
    os.path.join(DATA_DIR, "mc_questions_file-2.csv"),
]

# Letter to number mapping
LETTER_TO_NUM = {"A": "1", "B": "2", "C": "3", "D": "4"}

# Fixed instruction (same style as CulturalBench)
INSTRUCTION = (
    "Please answer the following cultural knowledge question "
    "by selecting the correct option number."
)

# Country name normalization (BLEnD uses some non-standard names)
COUNTRY_MAP = {
    "UK": "United Kingdom",
    "US": "United States",
    "USA": "United States",
    "South_Korea": "South Korea",
    "North_Korea": "North Korea",
    "West_Java": "Indonesia",
    "New_Zealand": "New Zealand",
    "Saudi_Arabia": "Saudi Arabia",
    "South_Africa": "South Africa",
    "Sri_Lanka": "Sri Lanka",
    "Costa_Rica": "Costa Rica",
    "El_Salvador": "El Salvador",
    "Puerto_Rico": "Puerto Rico",
    "Hong_Kong": "Hong Kong",
}


def normalize_country(country: str) -> str:
    """Normalize country name: replace underscores, apply mapping."""
    country = country.strip()
    if country in COUNTRY_MAP:
        return COUNTRY_MAP[country]
    # Replace underscores with spaces for any remaining cases
    return country.replace("_", " ")


def extract_question_and_options(prompt: str) -> tuple:
    """
    Extract the core question and options from BLEnD prompt.

    BLEnD prompt format:
      "What is ...? Without any explanation, choose only one from the given
       alphabet choices(e.g., A, B, C). Provide as JSON format: ...

       A. option1
       B. option2
       C. option3
       D. option4

       Answer:"

    Returns: (question_text, [option1, option2, option3, option4])
    """
    # Split at "Without any explanation" to get the pure question
    parts = prompt.split("Without any explanation")
    if len(parts) >= 2:
        question = parts[0].strip()
    else:
        # Fallback: try to split at the first newline after the question mark
        question = prompt.split("\n")[0].strip()

    # Extract options using regex: "A. ...", "B. ...", etc.
    options = []
    for letter in ["A", "B", "C", "D"]:
        # Match "A. something" pattern (option text ends at next letter or end)
        pattern = rf'{letter}\.\s*(.+?)(?=\n[A-D]\.|(?:\n\s*\n)|(?:\nAnswer:)|$)'
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            option_text = match.group(1).strip()
            options.append(option_text)

    return question, options


def parse_blend_csv(filepath: str) -> list:
    """
    Parse BLEnD CSV file which has multi-line fields (JSON in choices column).

    Returns list of dicts with keys: MCQID, ID, country, prompt, choices, 
    choice_countries, answer_idx
    """
    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    return records


def convert_blend():
    """Convert BLEnD CSV files to MAS format."""
    print("=" * 60)
    print("Converting BLEnD dataset")
    print("=" * 60)

    all_records = []
    for filepath in INPUT_FILES:
        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            continue
        records = parse_blend_csv(filepath)
        print(f"  Loaded {len(records)} records from {os.path.basename(filepath)}")
        all_records.extend(records)

    print(f"  Total raw records: {len(all_records)}")

    results = []
    skipped = 0
    skip_reasons = Counter()

    for record in all_records:
        prompt = record.get("prompt", "")
        country = record.get("country", "")
        answer_idx = record.get("answer_idx", "").strip()

        # Validate answer_idx
        if answer_idx not in LETTER_TO_NUM:
            skip_reasons["invalid_answer_idx"] += 1
            skipped += 1
            continue

        if not prompt or not country:
            skip_reasons["missing_prompt_or_country"] += 1
            skipped += 1
            continue

        # Extract question and options
        question, options = extract_question_and_options(prompt)

        if len(options) != 4:
            skip_reasons[f"options_count_{len(options)}"] += 1
            skipped += 1
            continue

        if not question:
            skip_reasons["empty_question"] += 1
            skipped += 1
            continue

        # Build input in CulturalBench format: question + numbered options
        input_text = (
            f"{question}\n"
            f"1. {options[0]}\n"
            f"2. {options[1]}\n"
            f"3. {options[2]}\n"
            f"4. {options[3]}"
        )

        # Convert answer letter to number
        output = LETTER_TO_NUM[answer_idx]

        # Normalize country name
        normalized_country = normalize_country(country)

        converted = {
            "instruction": INSTRUCTION,
            "input": input_text,
            "output": output,
            "country": normalized_country,
        }
        results.append(converted)

    # Save as JSON array
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nConverted {len(results)} samples successfully.")
    if skipped:
        print(f"Skipped {skipped} samples:")
        for reason, count in skip_reasons.most_common():
            print(f"  {reason}: {count}")
    print(f"Saved to: {OUTPUT_FILE}")

    # Print label distribution
    label_dist = Counter(r["output"] for r in results)
    print(f"\nLabel distribution:")
    for label in sorted(label_dist.keys()):
        print(f"  {label}: {label_dist[label]} ({label_dist[label]/len(results)*100:.1f}%)")

    # Print country distribution
    country_dist = Counter(r["country"] for r in results)
    print(f"\nCountry distribution ({len(country_dist)} countries):")
    for country, count in country_dist.most_common():
        print(f"  {country}: {count}")

    # Print sample for verification
    if results:
        print("\n--- Sample (first 3 entries) ---")
        for i, sample in enumerate(results[:3]):
            print(f"\n[{i+1}]")
            print(json.dumps(sample, ensure_ascii=False, indent=2))

    return results


if __name__ == "__main__":
    convert_blend()
