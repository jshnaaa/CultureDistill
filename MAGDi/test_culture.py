"""
MAGDi evaluation script for cultural alignment tasks (NormAD & CultureBench).

Evaluates a MAGDi-distilled model on the test split of cultural alignment data.
Uses the same test data format as CAMAD (from split_data.py pkl files) to ensure
fair comparison between MAGDi baseline and CAMAD.

The distilled model performs zero-shot inference (no GCN at test time),
same as the original MAGDi paper.

Usage:
    # Evaluate MAGDi on CultureBench test set (using CAMAD's test split)
    python test_culture.py \
        --dataset culturalbench \
        --data_source hf_cac \
        --data_pkl /path/to/culturalbench_splits.pkl \
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \
        --lora_model checkpoints/MAGDi_culturalbench_hf_cac \
        --output_json results/magdi_culturalbench_hf_cac.json

    # Evaluate MAGDi on NormAD test set
    python test_culture.py \
        --dataset normad \
        --data_source reconcile \
        --data_pkl /path/to/normad_splits.pkl \
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \
        --lora_model checkpoints/MAGDi_normad_reconcile \
        --output_json results/magdi_normad_reconcile.json

    # Evaluate base model (no distillation, zero-shot baseline)
    python test_culture.py \
        --dataset culturalbench \
        --data_source hf_cac \
        --data_pkl /path/to/culturalbench_splits.pkl \
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \
        --no_lora \
        --output_json results/baseline_culturalbench.json
"""

import os
import re
import json
import pickle
import argparse
import datetime
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer_culturalbench(text: str):
    """
    Extract answer (1-4) from generated text for CultureBench.
    Matches patterns like "the answer is 3", "Answer: 2", or last digit 1-4.
    """
    # Pattern 1: "answer is X" or "Answer: X"
    m = re.search(r"[Aa]nswer\s*(?:is|:)\s*([1-4])", text)
    if m:
        return m.group(1)
    # Pattern 2: "So the answer is X"
    m = re.search(r"[Ss]o the answer is\s*([1-4])", text)
    if m:
        return m.group(1)
    # Pattern 3: last occurrence of a standalone digit 1-4
    digits = re.findall(r"\b([1-4])\b", text)
    return digits[-1] if digits else None


def extract_answer_normad(text: str):
    """
    Extract answer (1-3) from generated text for NormAD.
    1 = acceptable, 2 = unacceptable, 3 = neutral
    """
    # Pattern 1: "answer is X" or "Answer: X"
    m = re.search(r"[Aa]nswer\s*(?:is|:)\s*([1-3])", text)
    if m:
        return m.group(1)
    # Pattern 2: "So the answer is X"
    m = re.search(r"[Ss]o the answer is\s*([1-3])", text)
    if m:
        return m.group(1)
    # Pattern 3: last occurrence of a standalone digit 1-3
    digits = re.findall(r"\b([1-3])\b", text)
    return digits[-1] if digits else None


def extract_answer(dataset: str, text: str):
    """Route to appropriate answer extractor."""
    if dataset == 'culturalbench':
        return extract_answer_culturalbench(text)
    elif dataset == 'normad':
        return extract_answer_normad(text)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt_magdi(query: str, country: str, dataset: str) -> str:
    """
    Format prompt for MAGDi inference.
    Uses the same [INST] format as training for consistency.
    """
    return f"[INST] ### Question: {query}[/INST] ### Answer:"


def format_prompt_chat(query: str, country: str, dataset: str, tokenizer) -> str:
    """
    Format prompt using chat template (for models that support it).
    Includes country context like CAMAD evaluation does.
    """
    input_text = f"[{country}]\n{query}"
    messages = [{"role": "user", "content": input_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback to simple format
        prompt = f"[INST] ### Question: {query}[/INST] ### Answer:"
    return prompt


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, test_samples, dataset, device,
             batch_size=16, max_new_tokens=300, temperature=0.7,
             prompt_format='magdi'):
    """
    Evaluate model on test samples.
    
    Returns dict with overall accuracy, per-country accuracy, and detailed results.
    """
    correct = 0
    total = 0
    country_correct = defaultdict(int)
    country_total = defaultdict(int)
    results = []
    
    for idx in range(0, len(test_samples), batch_size):
        batch_samples = test_samples[idx:idx + batch_size]
        
        # Format prompts
        prompts = []
        for sample in batch_samples:
            query = sample.get('query', '')
            country = sample.get('country', 'unknown')
            
            if prompt_format == 'chat':
                prompt = format_prompt_chat(query, country, dataset, tokenizer)
            else:
                prompt = format_prompt_magdi(query, country, dataset)
            prompts.append(prompt)
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)
        
        # Generate
        output_tokens = model.generate(
            **inputs,
            do_sample=(temperature > 0),
            top_p=0.9 if temperature > 0 else None,
            top_k=50 if temperature > 0 else None,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Decode and evaluate
        generated_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        
        for i, (sample, gen_text) in enumerate(zip(batch_samples, generated_texts)):
            gt = str(sample.get('gt', '')).strip()
            country = sample.get('country', 'unknown')
            
            # Extract only the generated part (after prompt)
            prompt_text = prompts[i] if i < len(prompts) else ""
            response = gen_text[len(prompt_text):] if gen_text.startswith(prompt_text) else gen_text
            
            pred = extract_answer(dataset, response)
            is_correct = (pred == gt)
            
            if is_correct:
                correct += 1
                country_correct[country] += 1
            total += 1
            country_total[country] += 1
            
            results.append({
                'query': sample.get('query', '')[:100],
                'country': country,
                'gold': gt,
                'pred': pred,
                'correct': is_correct,
                'response': response[:200],
            })
        
        # Progress
        acc = correct / total if total > 0 else 0
        print(f"  {datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"samples: {total}/{len(test_samples)} | acc: {acc:.4f}")
    
    # Compute metrics
    overall_acc = correct / total if total > 0 else 0.0
    
    per_country = {}
    for c in sorted(country_total.keys()):
        c_acc = country_correct[c] / country_total[c]
        per_country[c] = {
            'accuracy': c_acc,
            'correct': country_correct[c],
            'total': country_total[c],
        }
    
    return {
        'overall_accuracy': overall_acc,
        'correct': correct,
        'total': total,
        'per_country': per_country,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MAGDi evaluation for cultural alignment tasks"
    )
    # Dataset and data source
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['normad', 'culturalbench'],
                        help="Dataset type")
    parser.add_argument('--data_source', type=str, required=True,
                        choices=['hf_cac', 'reconcile'],
                        help="Data source used for training")
    
    # Data
    parser.add_argument('--data_pkl', type=str, default='',
                        help="Path to splits pkl file (from split_data.py). "
                             "If provided, uses the 'test' split.")
    parser.add_argument('--test_file', type=str, default='',
                        help="Alternative: direct path to test JSONL file")
    
    # Model
    parser.add_argument('--base_model', type=str, required=True,
                        choices=['llama', 'qwen'],
                        help="Base model alias (llama or qwen)")
    parser.add_argument('--lora_model', type=str, default='',
                        help="Path to MAGDi LoRA checkpoint")
    parser.add_argument('--no_lora', action='store_true',
                        help="Evaluate base model without LoRA (zero-shot baseline)")
    parser.add_argument('--cache_dir', type=str, default='',
                        help="Model cache directory")
    
    # Generation
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="Generation temperature (0 = greedy)")
    parser.add_argument('--max_new_tokens', type=int, default=300,
                        help="Max new tokens to generate")
    parser.add_argument('--prompt_format', type=str, default='magdi',
                        choices=['magdi', 'chat'],
                        help="Prompt format: 'magdi' ([INST] format) or 'chat' (chat template)")
    
    # Output
    parser.add_argument('--output_json', type=str, default='',
                        help="Path to save detailed results as JSON")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="Max test samples (0 = all)")
    
    args = parser.parse_args()
    
    # Resolve model alias to full path
    model_path = MODEL_ALIASES.get(args.base_model, args.base_model)
    
    # Validate args
    if not args.data_pkl and not args.test_file:
        raise ValueError("Must provide either --data_pkl or --test_file")
    if not args.no_lora and not args.lora_model:
        raise ValueError("Must provide --lora_model or use --no_lora for baseline")
    
    print(f"{'=' * 60}")
    print(f"MAGDi Cultural Alignment Evaluation")
    print(f"{'=' * 60}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data source: {args.data_source}")
    print(f"  Base model: {args.base_model} -> {model_path}")
    print(f"  LoRA model: {args.lora_model if not args.no_lora else '(none - zero-shot)'}")
    print(f"  Prompt format: {args.prompt_format}")
    print(f"{'=' * 60}")
    
    # Load test data
    if args.data_pkl:
        print(f"\nLoading test data from pkl: {args.data_pkl}")
        with open(args.data_pkl, 'rb') as f:
            splits = pickle.load(f)
        test_samples = splits['test']
    else:
        print(f"\nLoading test data from: {args.test_file}")
        test_samples = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_samples.append(json.loads(line))
    
    if args.max_samples > 0:
        test_samples = test_samples[:args.max_samples]
    print(f"  Test samples: {len(test_samples)}")
    
    # Load model
    print(f"\nLoading model...")
    cache_dir = args.cache_dir if args.cache_dir else None
    
    if args.no_lora:
        # Zero-shot baseline
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
    else:
        # MAGDi distilled model (base + LoRA)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        # Resolve LoRA path: check for adapter_config.json, try 'best/' subdirectory
        lora_path = args.lora_model
        if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            # Try 'best/' subdirectory (created by train_culture.py)
            best_path = os.path.join(lora_path, "best")
            if os.path.exists(os.path.join(best_path, "adapter_config.json")):
                lora_path = best_path
                print(f"  [INFO] adapter_config.json not found at root, using: {lora_path}")
            else:
                # Try finding any checkpoint-* subdirectory
                import glob
                ckpts = sorted(glob.glob(os.path.join(lora_path, "checkpoint-*")))
                for ckpt in reversed(ckpts):
                    if os.path.exists(os.path.join(ckpt, "adapter_config.json")):
                        lora_path = ckpt
                        print(f"  [INFO] Using checkpoint: {lora_path}")
                        break
        
        print(f"  Loading LoRA from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print(f"  LoRA merged into base model")
    
    model.eval()
    device = next(model.parameters()).device
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        add_eos_token=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Evaluate
    print(f"\nEvaluating...")
    metrics = evaluate(
        model, tokenizer, test_samples, args.dataset, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prompt_format=args.prompt_format
    )
    
    # Print results
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"\nPer-Country Accuracy:")
    for country, info in sorted(metrics['per_country'].items(),
                                 key=lambda x: -x[1]['accuracy']):
        print(f"  {country:20s}: {info['accuracy']:.4f} "
              f"({info['correct']}/{info['total']})")
    
    # Save results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'method': 'MAGDi',
            'dataset': args.dataset,
            'data_source': args.data_source,
            'base_model': model_path,
            'lora_model': args.lora_model if not args.no_lora else None,
            'prompt_format': args.prompt_format,
            'temperature': args.temperature,
            'overall_accuracy': metrics['overall_accuracy'],
            'correct': metrics['correct'],
            'total': metrics['total'],
            'per_country': metrics['per_country'],
            'results': metrics['results'],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
