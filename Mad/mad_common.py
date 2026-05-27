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
import threading

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
# I/O helpers
# ---------------------------------------------------------------------------

_io_lock = threading.Lock()


def write_to_jsonl(file_name, data):
    with _io_lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')


def reserve_unprocessed_queries(output_file, dataset):
    if not os.path.exists(output_file):
        return dataset
    processed = set()
    with open(output_file, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                processed.add(d.get("input", "").strip())
            except Exception:
                continue
    return [d for d in dataset if d.get("input", "").strip() not in processed]
