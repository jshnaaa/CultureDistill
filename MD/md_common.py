"""
MD (Multiagent Debate) - Shared utilities.

Prompt templates from:
  "Improving Factuality and Reasoning in Language Models through
   Multiagent Debate" (Du et al., 2023), Appendix Figure 15.

MD uses N copies of the SAME language model as agents. Each agent first
solves the problem independently ("Starting" prompt). Then over R rounds of
debate, each agent receives the concatenation of the OTHER agents' most
recent responses as additional advice and produces an updated answer
("Debate" prompt). The final answer is decided by majority vote over the
agents' last-round answers (ties broken by Agent 1's answer).

The Figure-15 MMLU template is the closest match to our multiple-choice
cultural tasks; we keep its "use other agents' reasoning as advice -> give an
updated answer" logic and only adapt the surface task wording / answer format
to each dataset.
"""

import os
import re
import json
from collections import Counter

# ---------------------------------------------------------------------------
# Model aliases (consistent with MAD / MACD / OG-MAR baselines)
# ---------------------------------------------------------------------------
MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

ANSWER_MAP = {"yes": "1", "no": "2", "neither": "3"}
REVERSE_ANSWER_MAP = {"1": "Yes", "2": "No", "3": "Neither"}

# Dataset types
DATASET_NORMAD = "normad"          # Yes/No/Neither social acceptability
DATASET_MCQ = "mcq"                # Multiple-choice (1/2/3/4) - CulturalBench
DATASET_BLEND = "blend"            # Multiple-choice (1/2/3/4) - BLEND
DATASET_CULTURELLM = "culturellm"  # World Values Survey, variable options


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())[0]
    return data


def detect_dataset_type(input_file: str):
    """Auto-detect dataset type from input file name."""
    basename = os.path.basename(input_file).lower()
    if "culturalbench" in basename:
        return DATASET_MCQ
    if "blend" in basename:
        return DATASET_BLEND
    if "culturellm" in basename:
        return DATASET_CULTURELLM
    return DATASET_NORMAD


def parse_input(input_text):
    """
    Parse the 'input' field to extract Country, Cultural Background, Scenario.
    Returns: (country: str, scenario: str, cultural_context: str)
    """
    m = re.search(r'Country:\s*(.+?)(?:\n|$)', input_text, re.IGNORECASE)
    country = m.group(1).strip() if m else ""

    cultural_context = ""
    bg_match = re.search(
        r'Cultural Background:\s*\n(.*?)(?=\nScenario:)',
        input_text, re.DOTALL | re.IGNORECASE
    )
    if bg_match:
        cultural_context = bg_match.group(1).strip()

    m = re.search(r'Scenario:\s*\n(.+)$', input_text, re.DOTALL)
    if m:
        scenario = m.group(1).strip()
    else:
        si = input_text.rfind("Scenario:")
        if si >= 0:
            scenario = input_text[si + len("Scenario:"):].strip()
        else:
            scenario = input_text.strip()

    return country, scenario, cultural_context


def get_culturellm_option_range(input_text: str):
    """Extract (min_option, max_option) from a CultureLLM input field."""
    m = re.search(r'from (\d+) to (\d+)', input_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1, 4  # default fallback


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text):
    """
    Extract final answer (NormAD: Yes/No/Neither -> 1/2/3) from model output.
    Returns: "1", "2", "3", or None.
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

    m = re.match(r'\s*(yes|no|neither)\b', tl)
    if m:
        return ANSWER_MAP.get(m.group(1))

    # Last-mention fallback
    last_pos, last_word = -1, None
    for word in ["yes", "no", "neither"]:
        matches = list(re.finditer(r'\b' + word + r'\b', tl))
        if matches and matches[-1].start() > last_pos:
            last_pos = matches[-1].start()
            last_word = word
    if last_word:
        return ANSWER_MAP[last_word]

    return None


def extract_answer_mcq(text):
    """Extract answer for MCQ (1/2/3/4). Returns "1"-"4" or None."""
    tl = text.strip()

    m = re.search(r'(?:answer|option)\s*(?:is|:)?\s*([1-4])\b', tl, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'^\s*([1-4])\s*[\.\)]?\s*$', tl, re.MULTILINE)
    if m:
        return m.group(1)
    m = re.match(r'\s*([1-4])\b', tl)
    if m:
        return m.group(1)
    matches = re.findall(r'\b([1-4])\b', tl)
    if matches:
        return matches[-1]
    return None


def extract_answer_culturellm(text, max_option: int):
    """
    Extract answer for CultureLLM (variable option range 0-2/1-4/1-5/1-10).
    Returns a string digit or None.
    """
    min_option = 0 if max_option == 2 else 1
    valid_set = set(str(i) for i in range(min_option, max_option + 1))
    tl = text.strip()

    m = re.search(r'(?:answer|option)\s*(?:is|:)?\s*(\d+)\b', tl, re.IGNORECASE)
    if m and m.group(1) in valid_set:
        return m.group(1)
    m = re.search(r'^\s*(\d+)\s*[\.\)]?\s*$', tl, re.MULTILINE)
    if m and m.group(1) in valid_set:
        return m.group(1)
    m = re.match(r'\s*(\d+)\b', tl)
    if m and m.group(1) in valid_set:
        return m.group(1)
    matches = re.findall(r'\b(\d+)\b', tl)
    valid_matches = [x for x in matches if x in valid_set]
    if valid_matches:
        return valid_matches[-1]
    return None


# ---------------------------------------------------------------------------
# Majority vote (final decision aggregation)
# ---------------------------------------------------------------------------

def majority_vote(answers):
    """
    Majority vote over a list of agent answers (strings; None/"" ignored).
    Ties are broken by the FIRST agent's answer (agents[0]).

    Returns: the winning answer string, or "" if no valid answer.
    """
    valid = [a for a in answers if a]
    if not valid:
        return ""

    counts = Counter(valid)
    top = counts.most_common()
    best_count = top[0][1]
    winners = [a for a, c in top if c == best_count]

    if len(winners) == 1:
        return winners[0]

    # Tie: prefer the earliest agent (agents[0]) whose answer is among winners
    for a in answers:
        if a in winners:
            return a
    return winners[0]


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def infer_output_path(input_file: str, model_name: str, output_dir: str = None) -> tuple:
    """
    Infer output file paths:
      {dataset}_MD_{model}_{timestamp}.json + _metrics.json
    """
    from datetime import datetime

    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    # Strip trailing _mas if present (consistent with MAD / MACD / OG-MAR)
    if dataset_name.endswith("_mas"):
        dataset_name = dataset_name[:-4]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dataset_name}_MD_{model_name}_{timestamp}"

    if output_dir is None:
        output_dir = "/autodl-fs/data/md"

    json_path = os.path.join(output_dir, f"{base_name}.json")
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
    return json_path, metrics_path


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list) -> dict:
    """Compute accuracy metrics from inference results."""
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

    per_country = {}
    for country, stats in sorted(country_stats.items()):
        per_country[country] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": (stats["correct"] / stats["total"]
                         if stats["total"] > 0 else 0.0),
        }

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
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w"):
        pass  # truncate


def append_jsonl(jsonl_path, record):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def jsonl_to_json(jsonl_path, json_path):
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records
