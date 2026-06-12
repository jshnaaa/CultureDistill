"""
Resume HF-CAC inference from where it stopped.

This script reads the existing output JSONL, skips already-processed samples,
and continues inference for the remaining ones. It writes to the SAME file
(no timestamp appended).

Usage:
    python Cul/resume_hf_cac.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/llama/normad_hf_cac_inference_XXXXXXXX_XXXXXX.jsonl \
        --model_name llama \
        --use_vllm --tensor_parallel_size 2 \
        --batch_size 8 \
        --negotiation_rounds 1 \
        --include_judge true
"""

import os
import sys
import re
import json
import argparse
import threading
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import write_to_jsonl


# ---------------------------------------------------------------------------
# Data helpers (same as generate_hf_cac_data.py)
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
    Kept in sync with generate_hf_cac_data.detect_dataset_type so that resume
    uses the SAME config (and therefore the same task_type) as the original run.
    """
    if not data:
        return "normad"

    sample = data[:min(10, len(data))]

    # Check instruction content — most reliable signal
    for item in sample:
        instruction = item.get("instruction", "")
        instr_lower = instruction.lower()
        if "world values survey" in instr_lower:
            return "culturellm"
        if "cultural knowledge question" in instr_lower:
            return "culturalbench"
        if "correct option number" in instr_lower:
            return "culturalbench"
        if "more culturally specific" in instr_lower:
            return "cultureatlas"
        if "response 1" in instr_lower and "response 2" in instr_lower:
            return "cultureatlas"
        if "acceptable" in instr_lower or "unacceptable" in instr_lower:
            return "normad"
        if "determine whether the behavior" in instr_lower:
            return "normad"

    # Check input content for CultureLLM (WVS survey pattern)
    for item in sample:
        inp = item.get("input", "")
        if "Give me the answer from" in inp and "You can only choose one option" in inp:
            return "culturellm"

    for item in sample:
        inp = item.get("input", "")
        if "Response 1:" in inp and "Response 2:" in inp:
            return "cultureatlas"

    for item in sample:
        inp = item.get("input", "")
        if "\n1. " in inp and "\n2. " in inp and "\n3. " in inp and "\n4. " in inp:
            return "culturalbench"

    # Fallback: check output distribution
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
    """Convert dataset sample to internal format."""
    if "input" in item and item["input"] and "country" in item:
        query = item["input"].strip()
        country = item["country"].strip()
        gt = str(item.get("output", "")).strip()
        return {"query": query, "gt": gt, "country": country}

    instruction = item["instruction"]
    query = instruction.split("### Answer:")[0].strip()
    if "Country" in item and item["Country"]:
        country = item["Country"].strip()
    else:
        m = re.search(r"country or language that is (.+?)\.", instruction)
        country = m.group(1).strip() if m else ""
    gt = str(item.get("output", item.get("label", ""))).strip()
    return {"query": query, "gt": gt, "country": country}


def get_processed_queries(output_path):
    """Read already-processed queries from existing output file."""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        processed.add(obj["query"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resume HF-CAC inference from existing output file"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to original dataset JSON file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to EXISTING output JSONL (will append to it)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama / qwen) or full local path")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--negotiation_rounds", type=int, default=1)
    parser.add_argument("--include_judge", type=str, default="true",
                        choices=["true", "false"])
    parser.add_argument("--num_agents", type=int, default=6,
                        choices=[2, 3, 4, 5, 6],
                        help="Number of cultural agents. MUST match the original "
                             "run. Default: 6.")

    args = parser.parse_args()
    args.include_judge = args.include_judge.lower() == "true"

    # Model alias resolution
    MODEL_ALIASES = {
        "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
        "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
    }
    args.model_name = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {args.model_name}")
    print(f"Output file (append mode): {args.output_file}")

    # Load full dataset
    raw_data = load_dataset(args.input_file)
    dataset = [convert_sample(item) for item in raw_data]
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples}")

    # Auto-detect dataset type and resolve config path (same logic as generate)
    dataset_type = detect_dataset_type(raw_data)
    print(f"Detected dataset type: {dataset_type}")
    if args.config_path is None:
        config_dir = os.path.join(os.path.dirname(__file__), "configs")
        if dataset_type == "cultureatlas":
            args.config_path = os.path.join(config_dir, "hf_cac_config_cultureatlas.yaml")
        elif dataset_type == "culturalbench":
            args.config_path = os.path.join(config_dir, "hf_cac_config_culturalbench.yaml")
        elif dataset_type == "culturellm":
            args.config_path = os.path.join(config_dir, "hf_cac_config_culturellm.yaml")
        else:
            args.config_path = os.path.join(config_dir, "hf_cac_config.yaml")
        print(f"Auto-selected config: {args.config_path}")

    # Filter out already-processed
    processed_queries = get_processed_queries(args.output_file)
    print(f"Already processed: {len(processed_queries)} samples")

    remaining = [s for s in dataset if s["query"] not in processed_queries]
    print(f"Remaining to process: {len(remaining)} samples")

    if len(remaining) == 0:
        print("All samples already processed. Nothing to do.")
        return

    # Initialize MAS
    from Cul.hf_cac_mas import HF_CAC_MAS

    mas = HF_CAC_MAS(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        config_path=args.config_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        include_judge=args.include_judge,
        negotiation_rounds=args.negotiation_rounds,
        num_agents=args.num_agents,
    )
    print(f"HF-CAC initialized:")
    print(f"  Num agents: {mas.num_agents}")
    print(f"  Task type: {mas.task_type}")
    print(f"  Include Judge: {args.include_judge}")
    print(f"  Negotiation rounds: {args.negotiation_rounds}")
    print(f"  Batch size: {args.batch_size}")
    print(f"\nResuming inference...")

    lock = threading.Lock()

    if args.use_vllm:
        for start in tqdm(range(0, len(remaining), args.batch_size), desc="Batches"):
            batch = remaining[start: start + args.batch_size]
            results = mas.inference_batch(batch)
            for sample, result in zip(batch, results):
                output = {**sample, **result, "task_type": mas.task_type,
                          "num_agents": mas.num_agents}
                write_to_jsonl(lock, args.output_file, output)
    else:
        for sample in tqdm(remaining, desc="Samples"):
            result = mas.inference(sample)
            output = {**sample, **result, "task_type": mas.task_type,
                      "num_agents": mas.num_agents}
            write_to_jsonl(lock, args.output_file, output)

    # Final count
    final_processed = get_processed_queries(args.output_file)
    print(f"\nDone! Total processed: {len(final_processed)}/{total_samples}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
