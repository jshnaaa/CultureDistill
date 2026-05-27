"""
MAD (Multi-Agent Debate) - Shared utilities.

Prompt templates from:
  "Multiple LLM Agents Debate for Equitable Cultural Alignment" (Ki et al., 2024)
  Appendix A.3 (Debate-Only) and A.4 (Self-Reflect+Debate)

All prompts use the NO-rule-of-thumb variant: Rule lines removed from original templates.
"""

import os
import re
import json
from collections import Counter

# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------
MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

ANSWER_MAP = {"yes": "1", "no": "2", "neither": "3"}
REVERSE_ANSWER_MAP = {"1": "Yes", "2": "No", "3": "Neither"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())[0]
    return data


def parse_input(input_text):
    """
    Parse the 'input' field to extract Country and Scenario.
    Strips Cultural Background section (rule-of-thumb equivalent).

    Returns: (country: str, scenario: str)
    """
    m = re.search(r'Country:\s*(.+?)(?:\n|$)', input_text, re.IGNORECASE)
    country = m.group(1).strip() if m else ""

    m = re.search(r'Scenario:\s*\n(.+)$', input_text, re.DOTALL)
    if m:
        scenario = m.group(1).strip()
    else:
        si = input_text.rfind("Scenario:")
        if si >= 0:
            scenario = input_text[si:].strip()
        else:
            scenario = input_text.strip()

    return country, scenario


def extract_answer(text):
    """
    Extract final answer from model output.
    Returns: "1" (Yes), "2" (No), "3" (Neither), or None.
    """
    tl = text.strip().lower()

    for pat in [
        r'answer\s*[:\(]\s*(yes|no|neither)',
        r'^\s*(yes|no|neither)\s*$',
        r'\b(yes|no|neither)\s*\.?\s*$',
    ]:
        m = re.search(pat, tl, re.MULTILINE)
        if m:
            return ANSWER_MAP.get(m.group(1))

    for word in ["neither", "no", "yes"]:
        idx = tl.rfind(word)
        if idx >= 0:
            before = tl[idx - 1] if idx > 0 else " "
            after = tl[idx + len(word)] if idx + len(word) < len(tl) else " "
            if not before.isalnum() and not after.isalnum():
                return ANSWER_MAP[word]
    return None


def extract_choice(text):
    """
    For Self-Reflect+Debate: extract choice (A) or (B).
    Returns: "A" (self-reflect) or "B" (debate).
    """
    tc = text.strip().upper()
    m = re.search(r'\b([AB])\b', tc)
    if m:
        return m.group(1)
    tl = tc.lower()
    if "reflect" in tl:
        return "A"
    if "debate" in tl or "respond" in tl or "feedback" in tl:
        return "B"
    return "A"


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def infer_output_path(input_file: str, method: str, variant: str, model_name: str,
                      output_dir: str = None) -> tuple:
    """
    Infer output file paths following naming convention:
      {dataset}_{method}_{variant}_{model}.json
      {dataset}_{method}_{variant}_{model}_metrics.json

    Returns: (output_json_path, metrics_json_path)
    """
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    # Strip trailing _mas if present
    if dataset_name.endswith("_mas"):
        dataset_name = dataset_name[:-4]

    base_name = f"{dataset_name}_{method}_{variant}_{model_name}"

    if output_dir is None:
        output_dir = "/autodl-fs/data/mad"

    json_path = os.path.join(output_dir, f"{base_name}.json")
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
    return json_path, metrics_path


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list) -> dict:
    """
    Compute accuracy metrics from inference results.

    Each entry in results should have:
      - "gt": ground truth label (string "1"/"2"/"3")
      - "country": country name
      - "final_answer": predicted answer
    """
    total = 0
    correct = 0
    country_stats = {}
    answer_dist = Counter()

    for r in results:
        gt = str(r.get("gt", r.get("output", ""))).strip()
        if not gt:
            continue
        country = r.get("country", "unknown")
        final_ans = r.get("final_answer", "")

        total += 1
        answer_dist[final_ans] += 1

        if final_ans == gt:
            correct += 1

        if country not in country_stats:
            country_stats[country] = {"total": 0, "correct": 0}
        country_stats[country]["total"] += 1
        if final_ans == gt:
            country_stats[country]["correct"] += 1

    # Per-country accuracy
    per_country = {}
    for country, stats in sorted(country_stats.items()):
        per_country[country] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": (stats["correct"] / stats["total"]
                         if stats["total"] > 0 else 0.0),
        }

    # GT distribution
    gt_dist = dict(Counter(
        str(r.get("gt", r.get("output", ""))).strip()
        for r in results if r.get("gt") or r.get("output")
    ))

    return {
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "gt_distribution": gt_dist,
        "prediction_distribution": dict(answer_dist),
        "per_country": per_country,
    }
