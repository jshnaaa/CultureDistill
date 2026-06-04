"""
CGM-GRPO Evaluation Script

Evaluates a trained CGM-GRPO model on the test set with dual-GPU support.
Supports three modes:
  - rl:     base model + CGM-GRPO LoRA adapter (no SFT)
  - sft_rl: base model + SFT LoRA (merged) + CGM-GRPO LoRA adapter
  - cgm:    alias for sft_rl (recommended for CGM-GRPO evaluation)

Outputs:
  - Overall accuracy
  - Per-country accuracy
  - Per-culture-circle accuracy (aggregated by 6 culture circles)
  - Optional: detailed JSON results with per-sample predictions

Dual-GPU: Policy model on cuda:0, batch inference optimized.

Usage:
    # Evaluate CGM-GRPO model (SFT+RL mode)
    python Cul/grpo/eval_cgm_grpo.py \\
        --mode sft_rl \\
        --model_name qwen \\
        --data_pkl /path/to/normad_splits.pkl \\
        --sft_adapter /path/to/camad_sft_qwen/best \\
        --grpo_adapter /path/to/camad_cgm_grpo_qwen/best \\
        --output_json /path/to/results/cgm_grpo_eval.json

    # Evaluate RL-only CGM-GRPO model
    python Cul/grpo/eval_cgm_grpo.py \\
        --mode rl \\
        --model_name qwen \\
        --data_pkl /path/to/normad_splits.pkl \\
        --grpo_adapter /path/to/camad_cgm_grpo_qwen/best \\
        --output_json /path/to/results/cgm_grpo_rl_only_eval.json
"""

import re
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cgm_grpo import get_culture_circle


MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

CIRCLE_NAMES = [
    "Western & Anglo-Saxon",
    "Latin American",
    "Sub-Saharan African",
    "East-Asian",
    "Islamic & Middle-Eastern",
    "South & Southeast Asian",
]

MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 512


def load_model(args):
    """Load model according to the specified mode."""
    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)
    print(f"Base model: {model_path}")
    print(f"Mode: {args.mode}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Determine device placement
    if torch.cuda.device_count() >= 2:
        device = torch.device("cuda:0")
        print(f"  Using dual-GPU setup (model on cuda:0)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    if args.mode == "rl":
        if not args.grpo_adapter:
            raise ValueError("--grpo_adapter required for mode=rl")
        print(f"Loading CGM-GRPO adapter: {args.grpo_adapter}")
        model = PeftModel.from_pretrained(model, args.grpo_adapter)
        print(f"  CGM-GRPO LoRA loaded (no SFT)")

    elif args.mode in ("sft_rl", "cgm"):
        if not args.sft_adapter:
            raise ValueError("--sft_adapter required for mode=sft_rl/cgm")
        if not args.grpo_adapter:
            raise ValueError("--grpo_adapter required for mode=sft_rl/cgm")
        print(f"Loading SFT adapter: {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()
        print(f"  SFT LoRA merged into base")
        print(f"Loading CGM-GRPO adapter: {args.grpo_adapter}")
        model = PeftModel.from_pretrained(model, args.grpo_adapter)
        print(f"  CGM-GRPO LoRA loaded on top of SFT-merged base")

    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use rl/sft_rl/cgm")

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def extract_answer(text: str):
    """Extract answer from generated response."""
    m = re.search(r"Answer\s*:\s*([1-4])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"answer\s+is\s*:?\s*([1-4])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    digits = re.findall(r"\b([1-4])\b", text)
    return digits[-1] if digits else None


@torch.no_grad()
def evaluate_on_test(model, tokenizer, test_samples, device,
                     max_samples=None, batch_size=1):
    """
    Evaluate model on test set.

    Returns dict with overall accuracy, per-country, per-circle metrics.
    """
    correct = 0
    total = 0
    country_correct = defaultdict(int)
    country_total = defaultdict(int)
    circle_correct = defaultdict(int)
    circle_total = defaultdict(int)
    results = []

    n_samples = (len(test_samples) if max_samples is None
                 else min(max_samples, len(test_samples)))

    for i, obj in enumerate(test_samples[:n_samples]):
        query = obj["query"]
        country = obj.get("country", "unknown")
        gold = str(obj["gt"]).strip()
        circle_id = get_culture_circle(country)

        # Build prompt
        input_text = f"[{country}]\n{query}"
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            prompt, return_tensors="pt",
            max_length=MAX_SEQ_LEN, truncation=True
        ).to(device)

        # Generate
        outs = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        response = tokenizer.decode(outs[0][prompt_len:], skip_special_tokens=True)

        # Extract answer
        pred = extract_answer(response)
        is_correct = (pred == gold)

        if is_correct:
            correct += 1
            country_correct[country] += 1
            if circle_id >= 0:
                circle_correct[circle_id] += 1
        total += 1
        country_total[country] += 1
        if circle_id >= 0:
            circle_total[circle_id] += 1

        results.append({
            "query": query[:100],
            "country": country,
            "culture_circle": CIRCLE_NAMES[circle_id] if circle_id >= 0 else "Unknown",
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response": response[:200],
        })

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples} "
                  f"(acc={correct/total:.4f})")

    # Compute metrics
    overall_acc = correct / total if total > 0 else 0.0

    per_country = {}
    for c in sorted(country_total.keys()):
        c_acc = country_correct[c] / country_total[c]
        per_country[c] = {
            "accuracy": c_acc,
            "correct": country_correct[c],
            "total": country_total[c],
        }

    per_circle = {}
    for cid in sorted(circle_total.keys()):
        c_acc = circle_correct[cid] / circle_total[cid]
        per_circle[CIRCLE_NAMES[cid]] = {
            "accuracy": c_acc,
            "correct": circle_correct[cid],
            "total": circle_total[cid],
        }

    return {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "per_country": per_country,
        "per_culture_circle": per_circle,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CGM-GRPO Evaluation: Test trained model on held-out test set"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["rl", "sft_rl", "cgm"],
                        help="Model mode: rl, sft_rl, or cgm (alias for sft_rl)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="'llama', 'qwen', or full model path")
    parser.add_argument("--data_pkl", type=str, required=True,
                        help="Path to splits pkl file (uses test set)")
    parser.add_argument("--sft_adapter", type=str, default=None,
                        help="SFT LoRA adapter path (for sft_rl/cgm modes)")
    parser.add_argument("--grpo_adapter", type=str, default=None,
                        help="CGM-GRPO LoRA adapter path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples (default: all)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save detailed results as JSON")
    args = parser.parse_args()

    # Load test data
    print(f"Loading data splits from: {args.data_pkl}")
    with open(args.data_pkl, "rb") as f:
        splits = pickle.load(f)
    test_data = splits["test"]
    print(f"  Test set: {len(test_data)} samples")

    # Load model
    model, tokenizer, device = load_model(args)

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating CGM-GRPO [{args.mode}] on test set...")
    print(f"{'='*60}\n")

    metrics = evaluate_on_test(
        model, tokenizer, test_data, device,
        max_samples=args.max_samples
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS [CGM-GRPO {args.mode}]")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")

    print(f"\nPer-Culture-Circle Accuracy:")
    for circle, info in sorted(metrics["per_culture_circle"].items(),
                                key=lambda x: -x[1]["accuracy"]):
        print(f"  {circle:30s}: {info['accuracy']:.4f} "
              f"({info['correct']}/{info['total']})")

    print(f"\nPer-Country Accuracy (top 20):")
    sorted_countries = sorted(metrics["per_country"].items(),
                              key=lambda x: -x[1]["accuracy"])
    for country, info in sorted_countries[:20]:
        print(f"  {country:20s}: {info['accuracy']:.4f} "
              f"({info['correct']}/{info['total']})")

    if len(sorted_countries) > 20:
        print(f"  ... ({len(sorted_countries) - 20} more countries)")

    # Save detailed results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "mode": args.mode,
                "model_name": args.model_name,
                "sft_adapter": args.sft_adapter,
                "grpo_adapter": args.grpo_adapter,
                "overall_accuracy": metrics["overall_accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "per_country": metrics["per_country"],
                "per_culture_circle": metrics["per_culture_circle"],
                "results": metrics["results"],
            }, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
