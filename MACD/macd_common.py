"""
MACD (Multi-Agent Cultural Debate) - Shared utilities.

Prompt templates from:
  "Mitigating Cultural Bias in LLMs via Multi-Agent Cultural Debate" (Tan et al., 2026)
  Appendix A (Meta prompt), B (Cultural Persona), C (SCGRD strategy)

Adapted for NormAD cultural acceptability judgment task.
"""

import os
import re
import json
from collections import Counter

# ---------------------------------------------------------------------------
# Model aliases (same as MAD for consistency)
# ---------------------------------------------------------------------------
MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

ANSWER_MAP = {"yes": "1", "no": "2", "neither": "3"}
REVERSE_ANSWER_MAP = {"1": "Yes", "2": "No", "3": "Neither"}


# ---------------------------------------------------------------------------
# Cultural Personas (Appendix B, verbatim from paper)
# ---------------------------------------------------------------------------

CULTURAL_PERSONAS = {
    "Western": (
        "You are a 29-year-old woman living in Amsterdam, the Netherlands. "
        "You speak English and Dutch, hold an MSc in Urban Planning, and work "
        "at a municipal planning agency. You cycle to work and spend weekends "
        "at museums or running outdoors. Living independently with your partner, "
        "you value privacy and contractual norms, prefer data- and evidence-based "
        "analysis at work, and make decisions that emphasize individual choice, "
        "equality, and transparent public rules while seeking defensible trade-offs "
        "between efficiency and fairness."
    ),
    "East Asian": (
        "You are a 22-year-old man from Guangzhou, China, now a computer science "
        "master's student and part-time teaching assistant. You speak Mandarin and "
        "Cantonese and keep close contact with your parents. Your daily routine is "
        "tightly scheduled, planful, and self-disciplined; your communication is "
        "restrained and context-sensitive. In team settings, you aim for harmony "
        "and prudent solutions, respect elders and institutions, and attend to "
        "practicality and cost."
    ),
    "African": (
        "You are a 30-year-old woman in Nairobi, Kenya, fluent in Swahili and "
        "English. Trained in public health, you work on community health programs "
        "and often collaborate with neighborhood organizations on outreach and "
        "services. Close to your siblings, you take part in community events and "
        "music during festivals."
    ),
    "Middle Eastern": (
        "You are a 32-year-old woman from Amman, Jordan, who speaks Arabic and "
        "English. You run a small catering business while managing family "
        "responsibilities. Daily life emphasizes hospitality and etiquette, with "
        "respect for tradition and legal norms."
    ),
    "South Asian": (
        "You are a 27-year-old man living in Chennai, India, who speaks Tamil "
        "and English. You hold a B.E. in Electrical Engineering and work as an "
        "engineer in manufacturing, living with your parents and valuing festivals "
        "and family rituals. Your manner is polite and measured."
    ),
}

# Cultural values associated with each persona (from Appendix B)
CULTURAL_VALUES = {
    "Western": "individual rights, freedom, rational analysis, utilitarianism",
    "East Asian": "social harmony, collective well-being, filial piety, face-saving",
    "African": "community, Ubuntu (I am because we are), collective responsibility, respect for elders",
    "Middle Eastern": "family honor, tradition, religious duty, hospitality",
    "South Asian": "dharma (moral duty), karma, spiritual growth, respect for hierarchy",
}


# ---------------------------------------------------------------------------
# SCGRD Strategy Prompt (Appendix C, verbatim from paper)
# ---------------------------------------------------------------------------

SCGRD_PROMPT = (
    "Adjust your response to align with your agents' examples, seeking a general "
    "answer to the question, trying to find common ground and maximize overall agreement."
)


# ---------------------------------------------------------------------------
# Data helpers (reuse from MAD common)
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

    # Pattern 1: "answer: yes/no/neither" or "answer (yes/no/neither)"
    for pat in [
        r'answer\s*[:\(]\s*(yes|no|neither)',
        r'^\s*(yes|no|neither)\s*$',
        r'\b(yes|no|neither)\s*\.?\s*$',
    ]:
        m = re.search(pat, tl, re.MULTILINE)
        if m:
            return ANSWER_MAP.get(m.group(1))

    # Pattern 2: starts with "yes/no/neither" (e.g., "No. Based on the...")
    m = re.match(r'\s*(yes|no|neither)\b', tl)
    if m:
        return ANSWER_MAP.get(m.group(1))

    # Pattern 3: find standalone word boundary matches (search all occurrences)
    for word in ["neither", "no", "yes"]:
        pattern = r'\b' + word + r'\b'
        matches = list(re.finditer(pattern, tl))
        if matches:
            return ANSWER_MAP[word]

    return None


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def infer_output_path(input_file: str, model_name: str,
                      output_dir: str = None) -> tuple:
    """
    Infer output file paths following naming convention:
      {dataset}_MACD_{model}.json
      {dataset}_MACD_{model}_metrics.json

    Returns: (output_json_path, metrics_json_path)
    """
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    # Strip trailing _mas if present
    if dataset_name.endswith("_mas"):
        dataset_name = dataset_name[:-4]

    base_name = f"{dataset_name}_MACD_{model_name}"

    if output_dir is None:
        output_dir = "/autodl-fs/data/macd"

    json_path = os.path.join(output_dir, f"{base_name}.json")
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
    return json_path, metrics_path


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list) -> dict:
    """
    Compute accuracy metrics from inference results.
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


# ---------------------------------------------------------------------------
# Incremental output helpers
# ---------------------------------------------------------------------------

def init_jsonl(jsonl_path):
    """Create/truncate JSONL file, ensure dir exists."""
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w") as f:
        pass  # truncate


def append_jsonl(jsonl_path, record):
    """Append one record to JSONL file."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def jsonl_to_json(jsonl_path, json_path):
    """Convert JSONL file to JSON array file."""
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records
