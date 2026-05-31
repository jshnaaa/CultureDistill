"""
Generate cultural alignment reasoning data using the HF-CAC framework.

HF-CAC (Home-Field Culture-Activated Collaboration):
  - Extends RECONCILE with dynamic authority activation based on target country
  - Host-Culture Guardian generates authoritative cultural claims
  - Cross-Cultural Auditors provide contrastive perspectives
  - Judge weights Guardian's claims with veto mechanism

Supports three dataset types (auto-detected from data format):
  - NormAD: behavior acceptability judgment (1/2/3) — uses hf_cac_config.yaml
  - CultureAtlas: comparative cultural depth (1/2) — uses hf_cac_config_cultureatlas.yaml
  - CulturalBench: multiple-choice cultural knowledge QA (1/2/3/4) — uses hf_cac_config_culturalbench.yaml

Output format mirrors AgentArk LLM Debate so that the existing
label.py / split_solutions pipeline can be reused directly.

Usage:
    # NormAD dataset (auto-detected)
    python Cul/generate_hf_cac_data.py \\
        --input_file /autodl-fs/data/normad_mas.json \\
        --output_file /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \\
        --model_name qwen \\
        --use_vllm --tensor_parallel_size 2 \\
        --max_samples 5 --negotiation_rounds 1 \\
        --include_judge true

    # CultureAtlas dataset (auto-detected)
    python Cul/generate_hf_cac_data.py \\
        --input_file /autodl-fs/data/cultureAtlas_mas.json \\
        --output_file /autodl-fs/data/qwen/cultureatlas_hf_cac_inference.jsonl \\
        --model_name qwen \\
        --use_vllm --tensor_parallel_size 2 \\
        --max_samples 5 --negotiation_rounds 1 \\
        --include_judge true

    # CulturalBench dataset (auto-detected)
    python Cul/generate_hf_cac_data.py \\
        --input_file /autodl-fs/data/culturalBench_mas.json \\
        --output_file /autodl-fs/data/qwen/culturalbench_hf_cac_inference.jsonl \\
        --model_name qwen \\
        --use_vllm --tensor_parallel_size 2 \\
        --max_samples 5 --negotiation_rounds 1 \\
        --include_judge true

    # Explicit config override (skip auto-detection)
    python Cul/generate_hf_cac_data.py \\
        --input_file /autodl-fs/data/cultureAtlas_mas.json \\
        --config_path Cul/configs/hf_cac_config_cultureatlas.yaml \\
        --model_name qwen \\
        --use_vllm --tensor_parallel_size 2 \\
        --max_samples 0 --negotiation_rounds 1 \\
        --include_judge true
"""

import os
import sys
import re
import json
import argparse
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import write_to_jsonl, reserve_unprocessed_queries


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())[0]
    return data


def detect_dataset_type(data: list) -> str:
    """
    Auto-detect dataset type from data content.

    Returns:
        "cultureatlas" if data matches CultureAtlas format (comparative, binary 1/2)
        "culturalbench" if data matches CulturalBench format (4-way knowledge QA, 1/2/3/4)
        "normad" otherwise (behavior acceptability, 3-way 1/2/3)

    Detection heuristics (in priority order):
      1. Instruction content: CultureAtlas mentions "more culturally specific" or
         "Response 1"/"Response 2"; NormAD mentions "acceptable"/"unacceptable";
         CulturalBench mentions "cultural knowledge question" or "correct option number"
      2. Input field: CultureAtlas has "Response 1:" and "Response 2:" patterns;
         CulturalBench has numbered options ("1. "/"2. "/"3. "/"4. ")
      3. Output distribution (full dataset): CultureAtlas only has 1/2, NormAD has 1/2/3,
         CulturalBench has 1/2/3/4
    """
    if not data:
        return "normad"

    # Sample first few items for instruction-based detection (most reliable)
    sample = data[:min(10, len(data))]

    # Check instruction content — most reliable signal
    for item in sample:
        instruction = item.get("instruction", "")
        instr_lower = instruction.lower()
        # CulturalBench markers
        if "cultural knowledge question" in instr_lower:
            return "culturalbench"
        if "correct option number" in instr_lower:
            return "culturalbench"
        # CultureAtlas markers
        if "more culturally specific" in instr_lower:
            return "cultureatlas"
        if "response 1" in instr_lower and "response 2" in instr_lower:
            return "cultureatlas"
        # NormAD markers
        if "acceptable" in instr_lower or "unacceptable" in instr_lower:
            return "normad"
        if "determine whether the behavior" in instr_lower:
            return "normad"

    # Check if input contains "Response 1" / "Response 2" pattern (CultureAtlas)
    for item in sample:
        inp = item.get("input", "")
        if "Response 1:" in inp and "Response 2:" in inp:
            return "cultureatlas"

    # Check if input contains numbered options pattern (CulturalBench)
    for item in sample:
        inp = item.get("input", "")
        if "\n1. " in inp and "\n2. " in inp and "\n3. " in inp and "\n4. " in inp:
            return "culturalbench"

    # Fallback: check output distribution across a larger sample
    # Use up to 100 samples to avoid false positives from small samples
    check_size = min(100, len(data))
    outputs = set(str(item.get("output", "")).strip()
                  for item in data[:check_size])
    if "4" in outputs:
        return "culturalbench"
    if "3" in outputs and "4" not in outputs:
        return "normad"
    if outputs and outputs <= {"1", "2"}:
        return "cultureatlas"

    return "normad"


def convert_sample(item):
    """
    Convert dataset sample to internal format.

    Supports two input formats:
      - normad_mas.json (new): {instruction, input, output, country}
        → input contains concatenated "Country: ...\nCultural Background: ...\nScenario: ..."
        → output is "1"/"2"/"3"
      - CulturalBench / legacy NormAD: {instruction (with ### Answer:), output/label, Country}

    Returns: {"query": str, "gt": str, "country": str}
    """
    # New format: normad_mas.json (has separate 'input' field with structured content)
    if "input" in item and item["input"] and "country" in item:
        # The 'input' field contains the full scenario text
        # The 'instruction' is the task description (answer 1/2/3)
        # Combine them as the query for the agent
        query = item["input"].strip()
        country = item["country"].strip()
        gt = str(item.get("output", "")).strip()
        return {"query": query, "gt": gt, "country": country}

    # Legacy format: CulturalBench / old NormAD
    instruction = item["instruction"]
    query = instruction.split("### Answer:")[0].strip()

    if "Country" in item and item["Country"]:
        country = item["Country"].strip()
    else:
        m = re.search(r"country or language that is (.+?)\.", instruction)
        country = m.group(1).strip() if m else ""

    gt = str(item.get("output", item.get("label", ""))).strip()
    return {"query": query, "gt": gt, "country": country}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate cultural reasoning data via HF-CAC MAS"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL path (default: auto-generated beside input)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama / qwen) or full local path")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to hf_cac_config.yaml (auto-detected if not specified)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM batch inference (recommended)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Samples per vLLM batch")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Number of samples to process. 0 = all samples.")
    parser.add_argument("--negotiation_rounds", type=int, default=1,
                        help="Rounds of structured negotiation. "
                             "0 = independent (Guardian+Auditors don't see each other). "
                             "1 = standard (Auditors see Guardian's response). "
                             "Default: 1.")
    parser.add_argument("--include_judge", type=str, default="true",
                        choices=["true", "false"],
                        help="Whether to include Judge reasoning. Default: true.")
    parser.add_argument("--eval_accuracy", type=str, default="true",
                        choices=["true", "false"],
                        help="Whether to compute and save accuracy metrics "
                             "after inference. Results saved as JSON beside "
                             "the output file. Default: true.")

    args = parser.parse_args()
    args.include_judge = args.include_judge.lower() == "true"
    args.eval_accuracy = args.eval_accuracy.lower() == "true"

    # ------------------------------------------------------------------
    # Model alias resolution
    # ------------------------------------------------------------------
    MODEL_ALIASES = {
        "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
        "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
    }
    args.model_name = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {args.model_name}")

    # ------------------------------------------------------------------
    # Output path: append timestamp before extension
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_file is None:
        stem = Path(args.input_file).stem
        args.output_file = str(
            Path(args.input_file).parent / f"{stem}_hf_cac_{timestamp}.jsonl"
        )
    else:
        p = Path(args.output_file)
        args.output_file = str(p.parent / f"{p.stem}_{timestamp}{p.suffix}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    print(f"Output file: {args.output_file}")

    # ------------------------------------------------------------------
    # Load and convert dataset
    # ------------------------------------------------------------------
    raw_data = load_dataset(args.input_file)
    dataset = [convert_sample(item) for item in raw_data]
    print(f"Loaded {len(dataset)} samples from {args.input_file}")

    # ------------------------------------------------------------------
    # Auto-detect dataset type and resolve config path
    # ------------------------------------------------------------------
    dataset_type = detect_dataset_type(raw_data)
    print(f"Detected dataset type: {dataset_type}")

    if args.config_path is None:
        # Auto-select config based on dataset type
        config_dir = os.path.join(os.path.dirname(__file__), "configs")
        if dataset_type == "cultureatlas":
            args.config_path = os.path.join(config_dir, "hf_cac_config_cultureatlas.yaml")
        elif dataset_type == "culturalbench":
            args.config_path = os.path.join(config_dir, "hf_cac_config_culturalbench.yaml")
        else:
            args.config_path = os.path.join(config_dir, "hf_cac_config.yaml")
        print(f"Auto-selected config: {args.config_path}")

    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]
        print(f"Using first {args.max_samples} samples")

    # ------------------------------------------------------------------
    # Resume: skip already-processed samples
    # ------------------------------------------------------------------
    dataset = reserve_unprocessed_queries(args.output_file, dataset)
    print(f"After resume filter: {len(dataset)} samples remaining")

    if len(dataset) == 0:
        print("All samples already processed.")
        return

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    from Cul.hf_cac_mas import HF_CAC_MAS

    mas = HF_CAC_MAS(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        config_path=args.config_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        include_judge=args.include_judge,
        negotiation_rounds=args.negotiation_rounds,
    )
    print(f"HF-CAC initialized:")
    print(f"  Task type: {mas.task_type}")
    print(f"  Include Judge: {args.include_judge}")
    print(f"  Negotiation rounds: {args.negotiation_rounds}")

    lock = threading.Lock()

    if args.use_vllm:
        for start in tqdm(range(0, len(dataset), args.batch_size), desc="Batches"):
            batch = dataset[start: start + args.batch_size]
            results = mas.inference_batch(batch)
            for sample, result in zip(batch, results):
                output = {**sample, **result, "task_type": mas.task_type}
                write_to_jsonl(lock, args.output_file, output)
    else:
        for sample in tqdm(dataset, desc="Samples"):
            result = mas.inference(sample)
            output = {**sample, **result, "task_type": mas.task_type}
            write_to_jsonl(lock, args.output_file, output)

    print(f"\nDone. Results saved to: {args.output_file}")

    # ------------------------------------------------------------------
    # Accuracy evaluation
    # ------------------------------------------------------------------
    if args.eval_accuracy:
        metrics = compute_accuracy(args.output_file)
        metrics_file = str(Path(args.output_file).with_suffix(".metrics.json"))
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n--- Accuracy Metrics ---")
        print(f"Task type: {metrics['task_type']}")
        print(f"Judge accuracy:    {metrics['judge_accuracy']:.4f} "
              f"({metrics['judge_correct']}/{metrics['judge_total']})")
        print(f"Guardian accuracy: {metrics['guardian_accuracy']:.4f} "
              f"({metrics['guardian_correct']}/{metrics['guardian_total']})")
        print(f"Metrics saved to: {metrics_file}")


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def detect_task_type_from_output(data: list) -> str:
    """Detect task type from inference output data (for accuracy computation).

    Detection priority:
      1. If any record has 'task_type' field, use it directly.
      2. Check query structure: if queries contain 4 numbered options ("\n1. "..."\n4. "),
         it's culturalbench (even if sampled GT doesn't include "4").
      3. Fallback to GT distribution analysis.
    """
    # Priority 1: explicit task_type field in output records
    for d in data[:10]:
        if d.get("task_type"):
            return d["task_type"]

    # Priority 2: check query structure for 4-option multiple choice
    sample = data[:min(20, len(data))]
    four_option_count = 0
    for d in sample:
        query = d.get("query", "")
        if "\n1. " in query and "\n2. " in query and "\n3. " in query and "\n4. " in query:
            four_option_count += 1
        elif "\n1." in query and "\n2." in query and "\n3." in query and "\n4." in query:
            four_option_count += 1
    if four_option_count >= len(sample) * 0.5:
        return "culturalbench"

    # Priority 3: GT distribution analysis
    outputs = set(d.get("gt", "").strip() for d in data[:100] if d.get("gt"))
    if "4" in outputs:
        return "culturalbench"
    if outputs and outputs <= {"1", "2"}:
        return "cultureatlas"
    return "normad"


def _extract_first_digit(text: str, pattern: str) -> str | None:
    """Extract answer using answer-first format (first line is the number)."""
    first_line = text.strip().split("\n")[0].strip()
    m = re.match(rf'^({pattern})$', first_line)
    if m:
        return m.group(1)
    # Fallback: "Answer: X" pattern
    m = re.search(rf'Answer\s*:\s*({pattern})', text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Last resort: last digit in valid range
    digits = re.findall(rf'\b({pattern})\b', text)
    return digits[-1] if digits else None


def extract_judge_answer(response_text: str, max_choice: int = 3,
                         question: str = ""):
    """Extract Judge's final answer from response text."""
    judge_match = re.search(
        r'===== Solution \d+ \[JUDGE.*?\] =====\n(.*?)$',
        response_text, re.DOTALL
    )
    if not judge_match:
        return None
    judge_text = judge_match.group(1)
    pattern = f'[1-{max_choice}]'
    return _extract_first_digit(judge_text, pattern)


def extract_guardian_answer(response_text: str, max_choice: int = 3,
                            question: str = ""):
    """Extract Guardian's answer from response text."""
    guardian_match = re.search(
        r'===== Solution \d+ \[GUARDIAN\] =====\n(.*?)(?=\n===== Solution)',
        response_text, re.DOTALL
    )
    if not guardian_match:
        return None
    guardian_text = guardian_match.group(1)
    pattern = f'[1-{max_choice}]'
    return _extract_first_digit(guardian_text, pattern)


def compute_accuracy(output_file: str) -> dict:
    """
    Compute Judge and Guardian accuracy from inference output JSONL.

    Auto-detects task type (NormAD 3-way vs CultureAtlas 2-way) from GT labels.
    Returns a dict with overall metrics and per-country breakdown.
    """
    data = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Auto-detect task type for answer extraction
    task_type = detect_task_type_from_output(data)
    if task_type == "cultureatlas":
        max_choice = 2
    elif task_type == "culturalbench":
        max_choice = 4
    else:
        max_choice = 3
    print(f"Accuracy evaluation — detected task type: {task_type} (max_choice={max_choice})")

    judge_correct = 0
    judge_total = 0
    guardian_correct = 0
    guardian_total = 0
    country_stats = {}

    for d in data:
        gt = d.get("gt", "").strip()
        if not gt:
            continue
        country = d.get("country", "unknown")
        response = d.get("response", "")
        question = d.get("query", "")

        judge_ans = extract_judge_answer(response, max_choice, question)
        guardian_ans = extract_guardian_answer(response, max_choice, question)

        if judge_ans:
            judge_total += 1
            if judge_ans == gt:
                judge_correct += 1

        if guardian_ans:
            guardian_total += 1
            if guardian_ans == gt:
                guardian_correct += 1

        if country not in country_stats:
            country_stats[country] = {
                "total": 0, "judge_correct": 0, "guardian_correct": 0
            }
        country_stats[country]["total"] += 1
        if judge_ans == gt:
            country_stats[country]["judge_correct"] += 1
        if guardian_ans == gt:
            country_stats[country]["guardian_correct"] += 1

    # Per-country accuracy
    per_country = {}
    for country, stats in sorted(country_stats.items()):
        per_country[country] = {
            "total": stats["total"],
            "judge_correct": stats["judge_correct"],
            "judge_accuracy": (stats["judge_correct"] / stats["total"]
                               if stats["total"] > 0 else 0.0),
            "guardian_correct": stats["guardian_correct"],
            "guardian_accuracy": (stats["guardian_correct"] / stats["total"]
                                  if stats["total"] > 0 else 0.0),
        }

    # GT and prediction distributions
    gt_dist = dict(Counter(d.get("gt", "").strip() for d in data if d.get("gt")))
    judge_ans_dist = dict(Counter(
        extract_judge_answer(d.get("response", ""), max_choice, d.get("query", ""))
        for d in data
    ))

    return {
        "total_samples": len(data),
        "task_type": task_type,
        "judge_total": judge_total,
        "judge_correct": judge_correct,
        "judge_accuracy": judge_correct / judge_total if judge_total > 0 else 0.0,
        "guardian_total": guardian_total,
        "guardian_correct": guardian_correct,
        "guardian_accuracy": (guardian_correct / guardian_total
                              if guardian_total > 0 else 0.0),
        "gt_distribution": gt_dist,
        "judge_answer_distribution": judge_ans_dist,
        "per_country": per_country,
    }


if __name__ == "__main__":
    main()
