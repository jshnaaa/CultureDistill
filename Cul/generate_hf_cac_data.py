"""
Generate cultural alignment reasoning data using the HF-CAC framework.

HF-CAC (Home-Field Culture-Activated Collaboration):
  - Extends RECONCILE with dynamic authority activation based on target country
  - Host-Culture Guardian generates authoritative cultural claims
  - Cross-Cultural Auditors provide contrastive perspectives
  - Judge weights Guardian's claims with veto mechanism

Output format mirrors AgentArk LLM Debate so that the existing
label.py / split_solutions pipeline can be reused directly.

Usage:
    # Quick test (5 samples, with negotiation)
    python Cul/generate_hf_cac_data.py \\
        --input_file /autodl-fs/data/normad_mas.json \\
        --output_file /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \\
        --model_name qwen \\
        --use_vllm --tensor_parallel_size 2 \\
        --max_samples 5 --negotiation_rounds 1 \\
        --include_judge true

    # Full dataset
    python Cul/generate_hf_cac_data.py \\
        --input_file /autodl-fs/data/normad_mas.json \\
        --output_file /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \\
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
                        help="Path to hf_cac_config.yaml")
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
    print(f"  Include Judge: {args.include_judge}")
    print(f"  Negotiation rounds: {args.negotiation_rounds}")

    lock = threading.Lock()

    if args.use_vllm:
        for start in tqdm(range(0, len(dataset), args.batch_size), desc="Batches"):
            batch = dataset[start: start + args.batch_size]
            results = mas.inference_batch(batch)
            for sample, result in zip(batch, results):
                output = {**sample, **result}
                write_to_jsonl(lock, args.output_file, output)
    else:
        for sample in tqdm(dataset, desc="Samples"):
            result = mas.inference(sample)
            output = {**sample, **result}
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
        print(f"Judge accuracy:    {metrics['judge_accuracy']:.4f} "
              f"({metrics['judge_correct']}/{metrics['judge_total']})")
        print(f"Guardian accuracy: {metrics['guardian_accuracy']:.4f} "
              f"({metrics['guardian_correct']}/{metrics['guardian_total']})")
        print(f"Metrics saved to: {metrics_file}")


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def extract_judge_answer(response_text: str):
    """Extract Judge's final answer (1/2/3) from response text."""
    judge_match = re.search(
        r'===== Solution \d+ \[JUDGE.*?\] =====\n(.*?)$',
        response_text, re.DOTALL
    )
    if not judge_match:
        return None
    judge_text = judge_match.group(1)
    m = re.search(r'Answer\s*:\s*([1-3])', judge_text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(r'\b([1-3])\b', judge_text)
    return digits[-1] if digits else None


def extract_guardian_answer(response_text: str):
    """Extract Guardian's answer (1/2/3) from response text."""
    guardian_match = re.search(
        r'===== Solution \d+ \[GUARDIAN\] =====\n(.*?)(?=\n===== Solution)',
        response_text, re.DOTALL
    )
    if not guardian_match:
        return None
    guardian_text = guardian_match.group(1)
    m = re.search(r'Answer\s*:\s*([1-3])', guardian_text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(r'\b([1-3])\b', guardian_text)
    return digits[-1] if digits else None


def compute_accuracy(output_file: str) -> dict:
    """
    Compute Judge and Guardian accuracy from inference output JSONL.

    Returns a dict with overall metrics and per-country breakdown.
    """
    data = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

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

        judge_ans = extract_judge_answer(response)
        guardian_ans = extract_guardian_answer(response)

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
        extract_judge_answer(d.get("response", "")) for d in data
    ))

    return {
        "total_samples": len(data),
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
