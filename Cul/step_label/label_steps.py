"""
CAMA-D Stage 2b: Open-Book Step Labeling (开卷式步骤标注)

Labels each reasoning step with one of three values:
  - 0.9: 主场确权步 (culturally specific evidence supporting correct answer)
  - 0.5: 中立讨论步 (neutral/generic reasoning, format transitions)
  - 0.1: 文化混淆步 (cultural confusion, points toward wrong answer)

"Open-Book" means the auditor receives the Ground Truth answer as prior knowledge,
reducing the task from open-ended quality assessment to simple semantic alignment
matching: "Does this step support the correct answer?"

Usage:
    # Using vLLM for batch inference
    python Cul/step_label/label_steps.py \\
        --input_file  /path/to/steps_split.jsonl \\
        --output_file /path/to/step_labels.jsonl \\
        --model_name  qwen \\
        --batch_size  64 \\
        --tensor_parallel_size 1

    # Validate labeling consistency (10% re-label)
    python Cul/step_label/label_steps.py \\
        --input_file  /path/to/steps_split.jsonl \\
        --output_file /path/to/step_labels.jsonl \\
        --model_name  qwen \\
        --validate_consistency --consistency_sample_rate 0.1
"""

import re
import json
import random
import argparse
from pathlib import Path
from typing import Optional


MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}


# ---------------------------------------------------------------------------
# Prompt template for open-book step labeling
# ---------------------------------------------------------------------------

LABEL_PROMPT_TEMPLATE = """You are evaluating a single reasoning step for cultural alignment.

Context:
- Question: {question}
- Target Culture: {country}
- Correct Answer: {ground_truth_answer}

Reasoning Step to Evaluate:
{step_text}

Task: Does this step support the correct answer within the target culture's context?

Label definitions:
- 0.9: This step provides culturally specific evidence that directly supports the correct answer (e.g., cites specific customs, traditions, values unique to the target culture). The model strongly endorses this step.
- 0.5: This step is neutral — it provides generic reasoning, format transitions, or universal logic that neither supports nor contradicts the correct answer in a culturally meaningful way. Neither reward nor penalty.
- 0.1: This step introduces cultural confusion — it points toward a wrong option, applies values from a different culture, or contains misconceptions about the target culture. The model strongly rejects this step.

Respond with ONLY one of: 0.9, 0.5, 0.1"""


def build_label_prompt(question: str, country: str,
                       ground_truth: str, step_text: str) -> str:
    """Build the prompt for labeling a single step."""
    return LABEL_PROMPT_TEMPLATE.format(
        question=question,
        country=country,
        ground_truth_answer=ground_truth,
        step_text=step_text,
    )


def parse_label_response(response: str) -> Optional[float]:
    """Parse LLM response into label value."""
    text = response.strip()
    # Try exact match first
    if text in ("0.9", "0.5", "0.1"):
        return float(text)
    # Try to find a valid label anywhere in the response
    for label in ["0.9", "0.5", "0.1"]:
        if label in text:
            return float(label)
    return None


# ---------------------------------------------------------------------------
# Batch labeling with vLLM
# ---------------------------------------------------------------------------

def label_steps_vllm(
    samples: list[dict],
    model_path: str,
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    temperature: float = 0.0,
    seed: int = 42,
) -> list[dict]:
    """
    Label all steps in all samples using vLLM batch inference.

    Each step is labeled independently (one LLM call per step).

    Args:
        samples: List of step-split samples from split_steps.py
        model_path: Path to auditor model
        batch_size: vLLM batch size
        tensor_parallel_size: Number of GPUs for tensor parallelism
        temperature: Sampling temperature (0.0 for deterministic)
        seed: Random seed for reproducibility

    Returns:
        Updated samples with 'label' field added to each step
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading auditor model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=8,
        stop=["<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>", "\n"],
    )

    # Build all prompts
    all_prompts = []
    prompt_index_map = []  # (sample_idx, step_idx) for each prompt

    for si, sample in enumerate(samples):
        for step in sample["steps"]:
            prompt_text = build_label_prompt(
                question=sample["question"],
                country=sample["country"],
                ground_truth=sample["gt"],
                step_text=step["text"],
            )
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(formatted)
            prompt_index_map.append((si, step["step_idx"] - 1))

    print(f"Total steps to label: {len(all_prompts)}")

    # Batch inference
    parse_failures = 0
    labels_assigned = 0

    for start in range(0, len(all_prompts), batch_size):
        batch = all_prompts[start:start + batch_size]
        outputs = llm.generate(batch, sampling_params)

        for i, out in enumerate(outputs):
            global_idx = start + i
            raw_response = out.outputs[0].text.strip()
            label = parse_label_response(raw_response)

            si, step_i = prompt_index_map[global_idx]

            if label is not None:
                samples[si]["steps"][step_i]["label"] = label
                labels_assigned += 1
            else:
                # Default to neutral if parsing fails
                samples[si]["steps"][step_i]["label"] = 0.5
                parse_failures += 1

        # Progress reporting
        done = min(start + batch_size, len(all_prompts))
        if (start // batch_size) % 10 == 0:
            print(f"  Labeled {done}/{len(all_prompts)} steps...")

    print(f"\nLabeling complete:")
    print(f"  Labels assigned: {labels_assigned}")
    print(f"  Parse failures (defaulted to 0.5): {parse_failures}")

    return samples


# ---------------------------------------------------------------------------
# Consistency validation
# ---------------------------------------------------------------------------

def validate_consistency(
    samples: list[dict],
    model_path: str,
    sample_rate: float = 0.1,
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    seed: int = 123,
) -> float:
    """
    Re-label a subset of steps with a different seed and compute consistency.

    Args:
        samples: Already-labeled samples
        sample_rate: Fraction of steps to re-label
        seed: Different seed for re-labeling

    Returns:
        Consistency rate (fraction of labels that match)
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\nValidating consistency (sample_rate={sample_rate}, seed={seed})...")

    # Select random subset of steps
    all_steps = []
    for si, sample in enumerate(samples):
        for step in sample["steps"]:
            all_steps.append((si, step))

    random.seed(seed)
    n_validate = max(1, int(len(all_steps) * sample_rate))
    validate_subset = random.sample(all_steps, n_validate)

    # Re-label with different seed
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.1,  # Slight temperature for diversity
        max_tokens=8,
        stop=["<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>", "\n"],
    )

    prompts = []
    original_labels = []
    for si, step in validate_subset:
        sample = samples[si]
        prompt_text = build_label_prompt(
            question=sample["question"],
            country=sample["country"],
            ground_truth=sample["gt"],
            step_text=step["text"],
        )
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted)
        original_labels.append(step.get("label", 0.5))

    # Batch inference
    matches = 0
    total = 0

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for i, out in enumerate(outputs):
            global_idx = start + i
            raw = out.outputs[0].text.strip()
            new_label = parse_label_response(raw)
            if new_label is not None:
                if new_label == original_labels[global_idx]:
                    matches += 1
                total += 1

    consistency = matches / total if total > 0 else 0.0
    print(f"  Consistency: {matches}/{total} = {consistency:.4f}")
    print(f"  Target: > 0.85")
    return consistency


# ---------------------------------------------------------------------------
# Label distribution statistics
# ---------------------------------------------------------------------------

def print_label_stats(samples: list[dict]) -> None:
    """Print distribution of labels across all steps."""
    label_counts = {0.9: 0, 0.5: 0, 0.1: 0}
    total = 0

    for sample in samples:
        for step in sample["steps"]:
            label = step.get("label", 0.5)
            if label > 0.7:
                label_counts[0.9] += 1
            elif label < 0.3:
                label_counts[0.1] += 1
            else:
                label_counts[0.5] += 1
            total += 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        name = {0.9: "主场确权步", 0.5: "中立讨论步", 0.1: "文化混淆步"}[label]
        print(f"  {label} ({name}): {count} ({pct:.1f}%)")
    print(f"  Total steps: {total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CAMA-D Stage 2b: Open-Book Step Labeling"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Step-split JSONL from split_steps.py")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL with labels")
    parser.add_argument("--model_name", type=str, default="qwen",
                        help="Auditor model: 'llama', 'qwen', or full path")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="vLLM batch size (default: 64)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--validate_consistency", action="store_true",
                        help="Run consistency validation after labeling")
    parser.add_argument("--consistency_sample_rate", type=float, default=0.1,
                        help="Fraction of steps to re-label for consistency check")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)

    # Load step-split data
    samples = [json.loads(l) for l in open(args.input_file, encoding="utf-8")]
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    # Label steps
    samples = label_steps_vllm(
        samples,
        model_path=model_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )

    # Print statistics
    print_label_stats(samples)

    # Consistency validation
    if args.validate_consistency:
        validate_consistency(
            samples,
            model_path=model_path,
            sample_rate=args.consistency_sample_rate,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed + 81,  # Different seed
        )

    # Save output
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nOutput saved to: {args.output_file}")


if __name__ == "__main__":
    main()
