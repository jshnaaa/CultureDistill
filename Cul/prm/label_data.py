"""
Step 1: Build pairwise dataset for PRM training.

Two labeling modes (--label_mode):

  answer_correctness (default, recommended):
    chosen  = paths where answer == gold
    rejected = paths where answer != gold
    No LLM needed. Reliable and fast.

  llm_pairwise:
    For each pair of paths in a sample, ask LLM which is more culturally accurate.
    Outputs A or B directly — avoids the score-saturation problem of absolute scoring.
    Slower but captures cultural quality beyond just answer correctness.

Usage:
    # Mode 1: answer correctness (default, no LLM needed)
    python Cul/prm/label_data.py \
        --input_file  /autodl-fs/data/splits/prm_train.jsonl \
        --output_file /autodl-fs/data/prm/prm_train_pairs.jsonl \
        --label_mode  answer_correctness

    # Mode 2: LLM pairwise comparison
    python Cul/prm/label_data.py \
        --input_file  /autodl-fs/data/splits/prm_train.jsonl \
        --output_file /autodl-fs/data/prm/prm_train_pairs.jsonl \
        --label_mode  llm_pairwise \
        --model_name  qwen \
        --batch_size  32
"""

import re
import json
import argparse
import itertools
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer


MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

# Prompt for pairwise comparison mode
PAIRWISE_PROMPT = (
    "You are evaluating two reasoning paths for a cultural knowledge question.\n"
    "Question (about {country}): {question}\n\n"
    "Path A:\n{reasoning_a}\n\n"
    "Path B:\n{reasoning_b}\n\n"
    "Which path better reflects the actual cultural values and norms of {country}? "
    "Consider factual accuracy, cultural specificity, and relevance.\n"
    "Reply with only the letter A or B."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def split_solutions(response: str) -> list[str]:
    parts = re.split(r"===== Solution \d+ =====", response)
    return [p.strip() for p in parts if p.strip()]


def extract_reasoning(text: str) -> str:
    m = re.search(r"Reasoning\s*:\s*(.*?)(?:\nAnswer|\Z)", text,
                  re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = text.strip().splitlines()
    if lines and re.match(r"Answer\s*:", lines[-1], re.IGNORECASE):
        return "\n".join(lines[:-1]).strip()
    return text.strip()


def extract_answer(text: str) -> str | None:
    m = re.search(r"Answer\s*:\s*([1-4]|yes|no)", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def load_data(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def write_pairs(pairs: list[dict], output_file: str) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Mode 1: Answer Correctness
# ---------------------------------------------------------------------------

def build_pairs_by_correctness(data: list[dict]) -> list[dict]:
    """
    chosen  = agent paths where predicted answer == gold
    rejected = agent paths where predicted answer != gold
    All pairwise combinations within each sample.
    """
    all_pairs = []
    skipped_no_contrast = 0

    for item in data:
        solutions = split_solutions(item["response"])
        agent_sols = solutions[:5]
        gt      = str(item.get("gt", "")).strip().lower()
        question = item["query"]
        country  = item.get("country", "")

        chosen_paths, rejected_paths = [], []
        for sol in agent_sols:
            reasoning = extract_reasoning(sol)
            answer    = extract_answer(sol)
            if not reasoning:
                continue
            if answer == gt:
                chosen_paths.append({"reasoning": reasoning, "answer": answer})
            else:
                rejected_paths.append({"reasoning": reasoning, "answer": answer})

        if not chosen_paths or not rejected_paths:
            skipped_no_contrast += 1
            continue

        for c, r in itertools.product(chosen_paths, rejected_paths):
            all_pairs.append({
                "question": question,
                "country":  country,
                "chosen":   {"reasoning": c["reasoning"], "answer": c["answer"],
                             "score": 1.0},
                "rejected": {"reasoning": r["reasoning"], "answer": r["answer"],
                             "score": 0.0},
            })

    print(f"  Samples with no contrast (all correct/wrong): {skipped_no_contrast}")
    return all_pairs


# ---------------------------------------------------------------------------
# Mode 2: LLM Pairwise Comparison
# ---------------------------------------------------------------------------

def build_pairs_by_llm(data: list[dict], model_path: str,
                       batch_size: int, tensor_parallel_size: int) -> list[dict]:
    """
    For each pair of agent paths in a sample, ask LLM which is better.
    Outputs A or B — no score saturation issue.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        stop=["<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>", "\n"],
    )

    # Collect all path-pairs across all samples
    # meta: (sample_idx, path_a_dict, path_b_dict, question, country)
    all_prompts, meta = [], []

    for si, item in enumerate(data):
        solutions = split_solutions(item["response"])
        agent_sols = solutions[:5]
        question = item["query"]
        country  = item.get("country", "")

        paths = []
        for sol in agent_sols:
            reasoning = extract_reasoning(sol)
            answer    = extract_answer(sol)
            if reasoning:
                paths.append({"reasoning": reasoning, "answer": answer})

        for a, b in itertools.combinations(paths, 2):
            prompt_text = PAIRWISE_PROMPT.format(
                country=country, question=question,
                reasoning_a=a["reasoning"], reasoning_b=b["reasoning"]
            )
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
            meta.append((si, a, b, question, country))

    print(f"Total comparisons to make: {len(all_prompts)}")

    # Batch generate
    results = []
    parse_failures = 0
    for start in range(0, len(all_prompts), batch_size):
        batch = all_prompts[start:start + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for i, out in enumerate(outputs):
            raw = out.outputs[0].text.strip().upper()
            global_idx = start + i
            if global_idx < 5:
                print(f"  [debug] comparison {global_idx}: raw={repr(raw)}")
            if raw.startswith("A"):
                results.append("A")
            elif raw.startswith("B"):
                results.append("B")
            else:
                results.append(None)
                parse_failures += 1
        if (start // batch_size) % 10 == 0:
            print(f"  Compared {min(start+batch_size, len(all_prompts))}/{len(all_prompts)}")

    print(f"  Parse failures: {parse_failures}/{len(all_prompts)}")

    # Build pairs from results
    all_pairs = []
    for (si, a, b, question, country), winner in zip(meta, results):
        if winner is None:
            continue
        chosen  = a if winner == "A" else b
        rejected = b if winner == "A" else a
        all_pairs.append({
            "question": question,
            "country":  country,
            "chosen":   {"reasoning": chosen["reasoning"],
                         "answer": chosen["answer"], "score": 1.0},
            "rejected": {"reasoning": rejected["reasoning"],
                         "answer": rejected["answer"], "score": 0.0},
        })

    return all_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--label_mode",  type=str, default="answer_correctness",
                        choices=["answer_correctness", "llm_pairwise"],
                        help="How to label pairs. 'answer_correctness' is fast and "
                             "reliable. 'llm_pairwise' uses LLM comparison.")
    parser.add_argument("--model_name",  type=str, default="qwen",
                        help="'llama', 'qwen', or full path (only for llm_pairwise)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    data = load_data(args.input_file)
    print(f"Loaded {len(data)} samples from {args.input_file}")

    if args.label_mode == "answer_correctness":
        print("Mode: answer_correctness (no LLM required)")
        pairs = build_pairs_by_correctness(data)

    else:  # llm_pairwise
        model_path = MODEL_ALIASES.get(args.model_name, args.model_name)
        print(f"Mode: llm_pairwise | Judge model: {model_path}")
        pairs = build_pairs_by_llm(
            data, model_path, args.batch_size, args.tensor_parallel_size
        )

    write_pairs(pairs, args.output_file)
    print(f"\nDone. {len(pairs)} pairwise pairs saved to {args.output_file}")

    # Quick stats
    if pairs:
        countries = set(p["country"] for p in pairs)
        print(f"Countries covered: {len(countries)}")
        print(f"Avg pairs per sample: {len(pairs)/len(data):.1f}")


if __name__ == "__main__":
    main()
