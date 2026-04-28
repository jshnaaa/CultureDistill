"""
Entry point for generating cultural alignment reasoning data
using the RECONCILE multi-agent framework.

Output format mirrors AgentArk LLM Debate so that the existing
label.py / split_solutions pipeline can be reused directly.

Usage:
    # Debug with sample data
    python Cul/generate_culture_data.py \
        --input_file Cul/data/sample.json \
        --output_file Cul/data/sample_generated.jsonl \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --use_vllm --tensor_parallel_size 1 --debug

    # Full dataset
    python Cul/generate_culture_data.py \
        --input_file /path/to/culturellm.json \
        --output_file results/CultureLLM/reconcile_infer.jsonl \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --use_vllm --tensor_parallel_size 4
"""

import os
import sys
import json
import argparse
import threading
from pathlib import Path
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import write_to_jsonl, reserve_unprocessed_queries


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def convert_sample(item):
    """
    Convert CultureLLM format to the internal query/gt/country format.

    Input:
        {
            "instruction": "### Question: ... ### Answer: ",
            "output": "1",
            "Country": "Arabic"
        }
    Output:
        {"query": "### Question: ...", "gt": "1", "country": "Arabic"}
    """
    instruction = item["instruction"]
    # Strip trailing '### Answer:' prompt suffix so query is clean
    query = instruction.split("### Answer:")[0].strip()
    return {
        "query": query,
        "gt": str(item["output"]).strip(),
        "country": item.get("Country", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate cultural reasoning data via RECONCILE MAS")

    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to CultureLLM JSON dataset")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL path (default: auto-generated beside input)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to reconcile_config.yaml (default: Cul/configs/reconcile_config.yaml)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM batch inference (recommended)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of samples per vLLM batch")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--debug", action="store_true",
                        help="Run one sample and print output without writing")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------
    if args.output_file is None:
        stem = Path(args.input_file).stem
        args.output_file = str(Path(args.input_file).parent / f"{stem}_reconcile_infer.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # ------------------------------------------------------------------
    # Load and convert dataset
    # ------------------------------------------------------------------
    raw_data = load_dataset(args.input_file)
    if isinstance(raw_data, dict):
        # Some datasets wrap the list in a dict
        raw_data = list(raw_data.values())[0]

    dataset = [convert_sample(item) for item in raw_data]
    print(f"Loaded {len(dataset)} samples from {args.input_file}")

    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]

    # ------------------------------------------------------------------
    # Debug mode: run one sample, print, exit
    # ------------------------------------------------------------------
    if args.debug:
        from Cul.reconcile_mas import ReconcileMAS
        mas = ReconcileMAS(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            config_path=args.config_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        sample = dataset[0]
        print(f"\n[DEBUG] Query: {sample['query']}")
        print(f"[DEBUG] GT: {sample['gt']} | Country: {sample['country']}\n")
        result = mas.inference(sample)
        output = {**sample, **result}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    # ------------------------------------------------------------------
    # Filter already-processed samples (resume support)
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
    )

    lock = threading.Lock()

    if args.use_vllm:
        # Batch processing
        batch_size = args.batch_size
        for start in tqdm(range(0, len(dataset), batch_size), desc="Batches"):
            batch = dataset[start: start + batch_size]
            results = mas.inference_batch(batch)
            for sample, result in zip(batch, results):
                write_to_jsonl(lock, args.output_file, {**sample, **result})
    else:
        # Sequential (fallback, no vLLM)
        for sample in tqdm(dataset, desc="Samples"):
            result = mas.inference(sample)
            write_to_jsonl(lock, args.output_file, {**sample, **result})

    print(f"\nDone. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
