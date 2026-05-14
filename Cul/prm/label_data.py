"""
Step 1: Score reasoning paths with LLM-as-a-Judge, then build pairwise dataset.

For each sample in the split file:
  - Extract Solution 1-5 (agent reasoning paths)
  - Score each path on cultural consistency (0.1-0.9) using the judge model
  - Build pairwise (chosen, rejected) pairs where score_diff > 0.2

Usage:
    python Cul/prm/label_data.py \
        --input_file /autodl-fs/data/splits/prm_train.jsonl \
        --output_file /autodl-fs/data/prm/prm_train_pairs.jsonl \
        --model_name llama \
        --batch_size 32

    python Cul/prm/label_data.py \
        --input_file /autodl-fs/data/splits/prm_val.jsonl \
        --output_file /autodl-fs/data/prm/prm_val_pairs.jsonl \
        --model_name llama \
        --batch_size 32
"""

import re
import json
import argparse
import itertools
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

SCORE_THRESHOLD = 0.2   # min score difference to form a pair

JUDGE_PROMPT = (
    "You are evaluating whether the following reasoning reflects "
    "the cultural values and norms of {country}.\n"
    "Question: {question}\n"
    "Reasoning: {reasoning}\n\n"
    "Rate the cultural consistency as a decimal between 0.1 and 0.9, "
    "where 0.9 = perfectly reflects {country}'s cultural values, "
    "0.1 = does not reflect {country}'s cultural values at all.\n"
    "Respond with only a decimal number."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_solutions(response: str) -> list[str]:
    parts = re.split(r"===== Solution \d+ =====", response)
    return [p.strip() for p in parts if p.strip()]


def extract_reasoning(text: str) -> str:
    """Extract the Reasoning: ... part, strip the Answer: line."""
    m = re.search(r"Reasoning\s*:\s*(.*?)(?:\nAnswer|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: return full text minus last line if it starts with Answer
    lines = text.strip().splitlines()
    if lines and re.match(r"Answer\s*:", lines[-1], re.IGNORECASE):
        return "\n".join(lines[:-1]).strip()
    return text.strip()


def parse_score(text: str) -> float | None:
    """
    Parse a score from model output. Handles:
      - Pure decimal: "0.8", "0.75"
      - With label:   "Score: 0.8", "I rate this 0.7"
      - Integer 1-9:  "8" → mapped to 0.8 (assuming 1-10 scale)
    """
    text = text.strip()
    # First try: explicit decimal 0.x or 1.0
    m = re.search(r"\b(0\.\d+|1\.0)\b", text)
    if m:
        v = float(m.group(1))
        return round(max(0.1, min(0.9, v)), 2)
    # Second try: integer 1-9 (model may output "8" for an 0-10 scale)
    m = re.search(r"\b([1-9])\b", text)
    if m:
        v = int(m.group(1)) / 10.0
        return round(max(0.1, min(0.9, v)), 2)
    return None


def build_pairs(scored_paths: list[dict]) -> list[dict]:
    """
    From a list of {reasoning, answer, score} dicts,
    build all pairwise combinations where score_diff > SCORE_THRESHOLD.
    """
    pairs = []
    for a, b in itertools.combinations(scored_paths, 2):
        if a["score"] is None or b["score"] is None:
            continue
        diff = a["score"] - b["score"]
        if abs(diff) <= SCORE_THRESHOLD:
            continue
        chosen  = a if diff > 0 else b
        rejected = b if diff > 0 else a
        pairs.append({"chosen": chosen, "rejected": rejected})
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name",  type=str, default="llama",
                        help="'llama', 'qwen', or full path")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)
    print(f"Judge model: {model_path}")

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    data = [json.loads(l) for l in open(args.input_file, encoding="utf-8")]
    print(f"Loaded {len(data)} samples")

    # Load judge model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Stop tokens: cover both Llama-3 and Qwen-2.5 formats.
    # Do NOT include "\n" — Qwen may emit a leading newline before the number.
    sampling_params = SamplingParams(
        temperature=0.0,    # greedy for consistent scoring
        max_tokens=32,      # allow enough tokens for the model to output a number
        stop=["<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>"],
    )

    # ------------------------------------------------------------------
    # Phase 1: collect all (sample_idx, path_idx, prompt) tuples
    # ------------------------------------------------------------------
    all_prompts = []
    meta = []   # (sample_idx, path_idx, question, country, reasoning)

    for si, item in enumerate(data):
        solutions = split_solutions(item["response"])
        agent_sols = solutions[:5]          # Solution 1-5 only (no Judge)
        question = item["query"]
        country  = item["country"]

        for pi, sol in enumerate(agent_sols):
            reasoning = extract_reasoning(sol)
            if not reasoning:
                continue
            prompt_text = JUDGE_PROMPT.format(
                country=country, question=question, reasoning=reasoning
            )
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
            meta.append((si, pi, question, country, reasoning))

    print(f"Total paths to score: {len(all_prompts)}")

    # ------------------------------------------------------------------
    # Phase 2: batch scoring
    # ------------------------------------------------------------------
    scores_flat = []
    parse_failures = 0
    for start in range(0, len(all_prompts), args.batch_size):
        batch = all_prompts[start:start + args.batch_size]
        outputs = llm.generate(batch, sampling_params)
        for i, out in enumerate(outputs):
            raw = out.outputs[0].text.strip()
            score = parse_score(raw)
            scores_flat.append(score)
            # Debug: print first 5 raw outputs so user can verify format
            global_idx = start + i
            if global_idx < 5:
                print(f"  [debug] path {global_idx}: raw_output={repr(raw)} → score={score}")
            if score is None:
                parse_failures += 1
        if (start // args.batch_size) % 10 == 0:
            print(f"  Scored {min(start + args.batch_size, len(all_prompts))}/{len(all_prompts)}")

    print(f"  Parse failures: {parse_failures}/{len(all_prompts)} "
          f"({parse_failures/max(len(all_prompts),1)*100:.1f}%)")

    # ------------------------------------------------------------------
    # Phase 3: group by sample, build pairwise pairs
    # ------------------------------------------------------------------
    from collections import defaultdict
    sample_paths = defaultdict(list)
    for (si, pi, question, country, reasoning), score in zip(meta, scores_flat):
        sample_paths[si].append({
            "reasoning": reasoning,
            "score": score,
            "question": question,
            "country": country,
        })

    total_pairs = 0
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for si, paths in sorted(sample_paths.items()):
            pairs = build_pairs(paths)
            for pair in pairs:
                record = {
                    "question": pair["chosen"]["question"],
                    "country":  pair["chosen"]["country"],
                    "chosen":  {
                        "reasoning": pair["chosen"]["reasoning"],
                        "score":     pair["chosen"]["score"],
                    },
                    "rejected": {
                        "reasoning": pair["rejected"]["reasoning"],
                        "score":     pair["rejected"]["score"],
                    },
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(f"\nDone. {total_pairs} pairwise pairs saved to {args.output_file}")


if __name__ == "__main__":
    main()
