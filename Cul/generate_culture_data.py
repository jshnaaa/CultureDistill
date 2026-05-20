"""
Generate cultural alignment reasoning data using the RECONCILE MAS framework.

Output format mirrors AgentArk LLM Debate so that the existing
label.py / split_solutions pipeline can be reused directly.

Usage:
    # Run on 5 samples (quick test)
    python Cul/generate_culture_data.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/normad_mas_inference.jsonl \
        --model_name qwen \
        --use_vllm --tensor_parallel_size 2 --max_samples 5

    # Full dataset (max_samples 0 = all)
    python Cul/generate_culture_data.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/normad_mas_inference.jsonl \
        --model_name qwen \
        --use_vllm --tensor_parallel_size 2 --max_samples 0
"""

import os
import sys
import re
import json
import argparse
import threading
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
    Convert dataset sample to internal query/gt/country format.

    Supports two input formats:
      - normad_mas.json (new): {instruction, input, output, country}
        → input contains concatenated "Country: ...\nCultural Background: ...\nScenario: ..."
        → output is "1"/"2"/"3"
      - CulturalBench / legacy format: {instruction (with ### Answer:), output/label, Country}

    Returns: {"query": str, "gt": str, "country": str}
    """
    # New format: normad_mas.json (has separate 'input' field with structured content)
    if "input" in item and item["input"] and "country" in item:
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
        description="Generate cultural reasoning data via RECONCILE MAS"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to CulturalBench JSON dataset")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL path (default: auto-generated beside input)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama / qwen) or full local path")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to reconcile_config.yaml")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM batch inference (recommended)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Samples per vLLM batch")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Number of samples to process. 0 = all samples.")
    parser.add_argument("--num_debate_rounds", type=int, default=None,
                        help="Number of debate rounds. Overrides config value if specified.")
    parser.add_argument("--include_judge", type=str, default="true",
                        choices=["true", "false"],
                        help="Whether to include Judge reasoning in output. "
                             "'true' = Solution N+1 includes Judge (default). "
                             "'false' = only Agent solutions, no Judge.")
    args = parser.parse_args()
    args.include_judge = args.include_judge.lower() == "true"

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
            Path(args.input_file).parent / f"{stem}_reconcile_infer_{timestamp}.jsonl"
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

    # max_samples=0 means all; otherwise take first N
    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]
        print(f"Using first {args.max_samples} samples (max_samples={args.max_samples})")

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
    from Cul.reconcile_mas import ReconcileMAS

    mas = ReconcileMAS(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        config_path=args.config_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_debate_rounds=args.num_debate_rounds,
        include_judge=args.include_judge,
    )
    print(f"Include Judge: {args.include_judge}")

    lock = threading.Lock()

    if args.use_vllm:
        for start in tqdm(range(0, len(dataset), args.batch_size), desc="Batches"):
            batch = dataset[start: start + args.batch_size]
            results = mas.inference_batch(batch)
            for sample, result in zip(batch, results):
                write_to_jsonl(lock, args.output_file, {**sample, **result})
    else:
        for sample in tqdm(dataset, desc="Samples"):
            result = mas.inference(sample)
            write_to_jsonl(lock, args.output_file, {**sample, **result})

    print(f"\nDone. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
