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
    )
    print(f"HF-CAC initialized:")
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
                output = {**sample, **result}
                write_to_jsonl(lock, args.output_file, output)
    else:
        for sample in tqdm(remaining, desc="Samples"):
            result = mas.inference(sample)
            output = {**sample, **result}
            write_to_jsonl(lock, args.output_file, output)

    # Final count
    final_processed = get_processed_queries(args.output_file)
    print(f"\nDone! Total processed: {len(final_processed)}/{total_samples}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
