#!/usr/bin/env python3
"""
Convert cultureLLM dataset (WVQ JSONL files) to the unified _mas.json format.

Source format (JSONL, one per line):
    {"text": "### Question: <question_text>\n ### Answer: <answer>"}

Target format (JSON array):
    [
      {
        "instruction": "...",
        "input": "<question_text>",
        "output": "<answer>",
        "country": "<country_name>"
      },
      ...
    ]

Usage:
    python convert_cultureLLM.py
"""

import json
import os
import re
import glob


# Mapping from filename patterns to country/culture names
FILENAME_TO_COUNTRY = {
    "WVQ_Arabic_Iraq_Jordan_llama.jsonl": "Arabic (Iraq & Jordan)",
    "WVQ_Bengali_llama.jsonl": "Bangladesh",
    "WVQ_China_llama.jsonl": "China",
    "WVQ_English_llama.jsonl": "United States",
    "WVQ_Germany_llama.jsonl": "Germany",
    "WVQ_Greece_llama.jsonl": "Greece",
    "WVQ_Korean_llama.jsonl": "South Korea",
    "WVQ_Portuguese_llama.jsonl": "Portugal",
    "WVQ_Spanish_llama.jsonl": "Spain",
    "WVQ_Turkey_llama.jsonl": "Turkey",
}

INSTRUCTION = (
    "Answer the following World Values Survey question by selecting the option number "
    "that best represents the cultural perspective of the specified country. "
    "Only output the number."
)


def parse_text_field(text: str):
    """
    Parse the 'text' field from cultureLLM JSONL format.
    Expected format: "### Question: <question>\n ### Answer: <answer>"
    Returns (question, answer) tuple.
    """
    # Split on "### Answer:" to get question and answer parts
    parts = re.split(r'\s*###\s*Answer:\s*', text, maxsplit=1)
    if len(parts) != 2:
        return None, None

    question_part = parts[0]
    answer = parts[1].strip()

    # Remove the "### Question:" prefix
    question = re.sub(r'^###\s*Question:\s*', '', question_part).strip()

    return question, answer


def convert_file(filepath: str, country: str) -> list:
    """Convert a single JSONL file to a list of records in the target format."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping line {line_num} in {filepath}: {e}")
                continue

            text = data.get("text", "")
            question, answer = parse_text_field(text)

            if question is None or answer is None:
                print(f"  WARNING: Could not parse line {line_num} in {filepath}")
                continue

            record = {
                "instruction": INSTRUCTION,
                "input": f"Country: {country}\n\nQuestion: {question}",
                "output": answer,
                "country": country,
            }
            records.append(record)

    return records


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, "cultureLLM")
    output_path = os.path.join(script_dir, "cultureLLM_mas.json")

    all_records = []
    files_processed = 0

    print("Converting cultureLLM dataset to _mas.json format...")
    print(f"Source directory: {source_dir}")
    print(f"Output file: {output_path}")
    print()

    for filename, country in sorted(FILENAME_TO_COUNTRY.items()):
        filepath = os.path.join(source_dir, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: File not found: {filepath}")
            continue

        records = convert_file(filepath, country)
        all_records.extend(records)
        files_processed += 1
        print(f"  Processed {filename}: {len(records)} records (country: {country})")

    print()
    print(f"Total files processed: {files_processed}")
    print(f"Total records: {len(all_records)}")

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
