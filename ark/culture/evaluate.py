"""
AgentArk Baseline — Evaluation Script

Evaluates trained models on test set from pkl splits.

Supports three modes:
  - sft:    base model + SFT LoRA adapter
  - rl:     base model + GRPO LoRA adapter (no SFT)
  - sft_rl: base model + SFT LoRA (merged) + GRPO LoRA adapter

Key difference from CAMA-D evaluate.py:
  - AgentArk prompt does NOT include country prefix [country]
  - Culture-agnostic policy: model must answer without explicit cultural cue
  - Per-country accuracy still computed (for analysis, not model input)

Usage:
    # Evaluate SFT model
    python ark/culture/evaluate.py \\
        --mode sft \\
        --model_name qwen \\
        --data_pkl /path/to/normad_agentark_splits.pkl \\
        --sft_adapter /path/to/agentark_sft_qwen/best

    # Evaluate SFT+RL model
    python ark/culture/evaluate.py \\
        --mode sft_rl \\
        --model_name qwen \\
        --data_pkl /path/to/normad_agentark_splits.pkl \\
        --sft_adapter /path/to/agentark_sft_qwen/best \\
        --grpo_adapter /path/to/agentark_grpo_qwen/best

    # Evaluate RL-only model
    python ark/culture/evaluate.py \\
        --mode rl \\
        --model_name qwen \\
        --data_pkl /path/to/normad_agentark_splits.pkl \\
        --grpo_adapter /path/to/agentark_grpo_qwen/best \\
        --data_source reconcile
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


MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 512


def load_model(args):
    """Load model according to the specified mode."""
    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)
    print(f"Base model: {model_path}")
    print(f"Mode: {args.mode}")
    print(f"Data source: {args.data_source}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    if args.mode == "sft":
        if not args.sft_adapter:
            raise ValueError("--sft_adapter required for mode=sft")
        print(f"Loading SFT adapter: {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)

    elif args.mode == "rl":
        if not args.grpo_adapter:
            raise ValueError("--grpo_adapter required for mode=rl")
        print(f"Loading GRPO adapter: {args.grpo_adapter}")
        model = PeftModel.from_pretrained(model, args.grpo_adapter)

    elif args.mode == "sft_rl":
        if not args.sft_adapter:
            raise ValueError("--sft_adapter required for mode=sft_rl")
        if not args.grpo_adapter:
            raise ValueError("--grpo_adapter required for mode=sft_rl")
        print(f"Loading SFT adapter: {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()
        print(f"  SFT LoRA merged")
        print(f"Loading GRPO adapter: {args.grpo_adapter}")
        model = PeftModel.from_pretrained(model, args.grpo_adapter)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def evaluate_on_test(model, tokenizer, test_samples: list[dict], device,
                     max_samples: int = None) -> dict:
    """
    Evaluate model on test set.

    AgentArk: NO country prefix in prompt (culture-agnostic policy).
    Per-country accuracy is still computed for analysis.
    """
    correct = 0
    total = 0
    country_correct = defaultdict(int)
    country_total = defaultdict(int)
    results = []

    n_samples = (len(test_samples) if max_samples is None
                 else min(max_samples, len(test_samples)))

    for i, obj in enumerate(test_samples[:n_samples]):
        query = obj["query"]
        country = obj.get("country", "unknown")
        gold = str(obj["gt"]).strip()

        # AgentArk: NO [country] prefix — culture-agnostic prompt
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            prompt, return_tensors="pt",
            max_length=MAX_SEQ_LEN, truncation=True
        ).to(device)

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

        pred = extract_answer(response)
        is_correct = (pred == gold)

        if is_correct:
            correct += 1
            country_correct[country] += 1
        total += 1
        country_total[country] += 1

        results.append({
            "query": query,
            "country": country,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response": response[:200],
        })

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples} "
                  f"(acc={correct/total:.4f})")

    overall_acc = correct / total if total > 0 else 0.0

    per_country = {}
    for c in sorted(country_total.keys()):
        c_acc = country_correct[c] / country_total[c]
        per_country[c] = {
            "accuracy": c_acc,
            "correct": country_correct[c],
            "total": country_total[c],
        }

    return {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "per_country": per_country,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AgentArk Baseline — Evaluation on Test Set"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["sft", "rl", "sft_rl"],
                        help="Model mode: sft, rl, or sft_rl")
    parser.add_argument("--model_name", type=str, required=True,
                        help="'llama', 'qwen', or full model path")
    parser.add_argument("--data_source", type=str, default="reconcile",
                        choices=["reconcile", "hf_cac"],
                        help="Data source used for training (for logging)")
    parser.add_argument("--data_pkl", type=str, required=True,
                        help="Path to splits pkl file")
    parser.add_argument("--sft_adapter", type=str, default=None,
                        help="SFT LoRA adapter path")
    parser.add_argument("--grpo_adapter", type=str, default=None,
                        help="GRPO LoRA adapter path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples (default: all)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save detailed results to JSON")
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
    print(f"AgentArk [{args.mode}] | data_source={args.data_source}")
    print(f"{'='*60}\n")

    metrics = evaluate_on_test(
        model, tokenizer, test_data, device,
        max_samples=args.max_samples
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS — AgentArk [{args.mode}] (data_source={args.data_source})")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"\nPer-Country Accuracy:")
    for country, info in sorted(metrics["per_country"].items(),
                                 key=lambda x: -x[1]["accuracy"]):
        print(f"  {country:20s}: {info['accuracy']:.4f} "
              f"({info['correct']}/{info['total']})")

    # Save results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "method": "agentark",
                "mode": args.mode,
                "data_source": args.data_source,
                "model_name": args.model_name,
                "sft_adapter": args.sft_adapter,
                "grpo_adapter": args.grpo_adapter,
                "overall_accuracy": metrics["overall_accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "per_country": metrics["per_country"],
                "results": metrics["results"],
            }, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
