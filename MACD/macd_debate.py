"""
MACD Baseline: Multi-Agent Cultural Debate

Reproduction of "Mitigating Cultural Bias in LLMs via Multi-Agent Cultural Debate"
(Tan et al., 2026) adapted for cultural reasoning tasks.

Supports two dataset formats (auto-detected):
  - NormAD: cultural acceptability judgment (Yes/No/Neither)
  - CulturalBench: cultural knowledge multiple-choice (4 options)

Framework:
  1. Round 1 - Initial Response: 5 cultural agents (Western, East Asian, African,
     Middle Eastern, South Asian) each answer the question from their cultural persona.
  2. Round 2 - Debate with SCGRD: Each agent views others' Round-1 responses and
     refines its answer using the "Seeking Common Ground while Reserving Differences"
     strategy.
  3. Summary: A summary model synthesizes all Round-2 responses into a final answer.

Prompts follow Appendix A (Meta prompt), B (Cultural Persona), C (SCGRD) from the
original paper, with minimal adaptation for the judgment task format.

Speed optimizations:
  - All 5 agents' prompts in each round are batched into a SINGLE vLLM call
    (5*N prompts at once), leveraging vLLM's continuous batching scheduler.
  - No manual batch loops; vLLM handles scheduling internally for max throughput.
  - Higher gpu_memory_utilization for larger KV cache.

Output naming: {dataset}_MACD_{model}.json + _metrics.json

Usage:
    # NormAD dataset
    python MACD/macd_debate.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --model_name qwen \
        --tensor_parallel_size 2 \
        --max_samples 5

    # CulturalBench dataset (auto-detected)
    python MACD/macd_debate.py \
        --input_file /autodl-fs/data/culturalBench_mas.json \
        --model_name qwen \
        --tensor_parallel_size 2 \
        --max_samples 5
"""

import os
import sys
import json
import time
import argparse
from collections import Counter

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MACD.macd_common import (
    MODEL_ALIASES, CULTURAL_PERSONAS, CULTURAL_VALUES, SCGRD_PROMPT,
    ANSWER_MAP, REVERSE_ANSWER_MAP,
    CB_VALID_ANSWERS, CB_REVERSE_MAP,
    DATASET_NORMAD, DATASET_CULTURALBENCH, detect_dataset_type,
    load_dataset, parse_input, parse_input_culturalbench,
    extract_answer, extract_answer_culturalbench,
    infer_output_path, compute_metrics,
)


# ===================================================================
# System Prompt
# ===================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant with expertise in cross-cultural "
    "social norms and etiquette."
)


# ===================================================================
# Prompt Templates - NormAD (following Appendix A, B, C of MACD paper)
# ===================================================================

# Round 1: Initial Response (Appendix A - Meta prompt for round 1)
PROMPT_ROUND1_NORMAD = (
    "{persona}\n\n"
    "You are currently participating in a debate, and there is round 1 of the debate.\n\n"
    "Question: Is the action in the following scenario socially acceptable in {country}?\n\n"
    "Cultural background of {country}:\n{cultural_bg}\n\n"
    "Scenario: {scenario}\n\n"
    "Based on the cultural background above, judge whether the action is socially "
    "acceptable in {country}.\n"
    "- Answer 'Yes' if the action aligns with the cultural norms described above.\n"
    "- Answer 'No' if the action violates or conflicts with the cultural norms described above.\n"
    "- Answer 'Neither' only if the cultural background provides no relevant guidance "
    "and the acceptability is genuinely indeterminate.\n\n"
    "Be decisive. If the cultural background suggests the action is inappropriate, answer 'No'.\n"
    "Explain your reasoning in one to two sentences.\n"
    "Answer (Yes/No/Neither):"
)

# Round 2: Debate with SCGRD (Appendix A - Meta prompt for round 2+)
PROMPT_ROUND2_NORMAD = (
    "{persona}\n\n"
    "You are currently participating in a debate, and there is round 2 of the debate.\n\n"
    "Question: Is the action in the following scenario socially acceptable in {country}?\n\n"
    "Cultural background of {country}:\n{cultural_bg}\n\n"
    "Scenario: {scenario}\n\n"
    "Previous responses of people from other culture background:\n"
    "{other_responses}\n\n"
    "Based on other perspectives and **{scgrd}** strategy, refine your answer. "
    "Focus on what the cultural background of {country} specifically indicates about "
    "this action. If the action clearly violates cultural norms, answer 'No' even if "
    "some agents disagree.\n"
    "Explain your reasoning in one to two sentences.\n"
    "Answer (Yes/No/Neither):"
)

# Summary prompt - NormAD
PROMPT_SUMMARY_NORMAD = (
    "After a multi-agent cultural debate, the following are the final judgments "
    "from agents representing different cultural perspectives on whether an action "
    "is socially acceptable in {country}:\n\n"
    "{agent_responses}\n\n"
    "Cultural background of {country}:\n{cultural_bg}\n\n"
    "Scenario: {scenario}\n\n"
    "Based on the cultural background and the debate above, provide the final "
    "judgment on whether the action is socially acceptable in {country}.\n"
    "- Answer 'Yes' if the action is acceptable according to the cultural norms.\n"
    "- Answer 'No' if the action violates or conflicts with the cultural norms.\n"
    "- Answer 'Neither' only if there is genuinely no cultural norm that applies.\n\n"
    "Prioritize the cultural background information over agent opinions when they conflict.\n"
    "Answer (Yes/No/Neither):"
)


# ===================================================================
# Prompt Templates - CulturalBench (adapted from MACD paper prompts)
# ===================================================================

# Round 1: Initial Response
PROMPT_ROUND1_CB = (
    "{persona}\n\n"
    "You are currently participating in a debate, and there is round 1 of the debate.\n\n"
    "Answer the following cultural knowledge question about {country}.\n\n"
    "{question}\n\n"
    "Directly answer the question according to your cultural knowledge about {country}. "
    "Select the correct option number and explain your reasoning in less than "
    "three sentences.\n"
    "Answer (option number):"
)

# Round 2: Debate with SCGRD
PROMPT_ROUND2_CB = (
    "{persona}\n\n"
    "You are currently participating in a debate, and there is round 2 of the debate.\n\n"
    "Answer the following cultural knowledge question about {country}.\n\n"
    "{question}\n\n"
    "Previous responses of people from other culture background:\n"
    "{other_responses}\n\n"
    "Based on other perspectives and **{scgrd}** strategy, refine your answer to "
    "the question. You must summarize the common reasoning with other "
    "cultures at the end of your refined answer. Don't over-analyze, such as what "
    "these cultural perspectives indicate or mean. You just discuss the original question.\n\n"
    "Remember: answer specifically based on {country}'s cultural context.\n"
    "Select the correct option number and explain your reasoning in less than "
    "three sentences.\n"
    "Answer (option number):"
)

# Summary prompt - CulturalBench
PROMPT_SUMMARY_CB = (
    "After a multi-agent cultural debate, the following are the final answers "
    "from agents representing different cultural perspectives on a cultural "
    "knowledge question about {country}:\n\n"
    "{agent_responses}\n\n"
    "Question: {question}\n\n"
    "Based on the debate above, determine the correct answer to this cultural "
    "knowledge question about {country}. Consider the majority consensus among "
    "the agents and prioritize perspectives most relevant to {country}.\n\n"
    "Respond with the correct option number only.\n"
    "Answer (option number):"
)


# ===================================================================
# Culture names list
# ===================================================================

CULTURE_NAMES = list(CULTURAL_PERSONAS.keys())
NUM_AGENTS = len(CULTURE_NAMES)


# ===================================================================
# Chat template helper
# ===================================================================

def apply_chat(tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ===================================================================
# Helpers
# ===================================================================

def format_other_responses(responses: dict, exclude_culture: str) -> str:
    parts = []
    for culture, resp in responses.items():
        if culture != exclude_culture:
            parts.append(f"- {culture} perspective: {resp}")
    return "\n".join(parts)


def format_agent_responses_for_summary(
    responses: dict, answers: dict, dataset_type: str
) -> str:
    parts = []
    reverse_map = REVERSE_ANSWER_MAP if dataset_type == DATASET_NORMAD else CB_REVERSE_MAP
    for culture, resp in responses.items():
        ans = answers.get(culture)
        ans_label = reverse_map.get(ans, "Unknown") if ans else "Unknown"
        parts.append(f"[{culture} Agent] (Answer: {ans_label}): {resp}")
    return "\n\n".join(parts)


def majority_vote(answers: dict, dataset_type: str) -> str:
    valid = [a for a in answers.values() if a is not None]
    if not valid:
        return None
    counter = Counter(valid)
    max_count = counter.most_common(1)[0][1]
    candidates = [ans for ans, cnt in counter.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    # Tie-breaking: prefer lower-numbered option
    if dataset_type == DATASET_NORMAD:
        priority = ["1", "2", "3"]
    else:
        priority = ["1", "2", "3", "4"]
    for p in priority:
        if p in candidates:
            return p
    return candidates[0]


# ===================================================================
# Main inference function
# ===================================================================

def run_macd(args):
    t_start = time.time()

    # --- Resolve model ---
    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {model_path}")

    # --- Determine output paths ---
    out_json, out_metrics = infer_output_path(
        args.input_file, args.model_name, args.output_dir
    )
    print(f"Output JSON:    {out_json}")
    print(f"Output Metrics: {out_metrics}")

    # --- Load data ---
    raw_data = load_dataset(args.input_file)
    dataset = raw_data
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
    print(f"Loaded {len(dataset)} samples from {args.input_file}")

    # --- Detect dataset type (by filename) ---
    dataset_type = detect_dataset_type(args.input_file)
    print(f"Dataset type: {dataset_type}")

    # --- Select prompt templates and extract function ---
    if dataset_type == DATASET_NORMAD:
        PROMPT_R1 = PROMPT_ROUND1_NORMAD
        PROMPT_R2 = PROMPT_ROUND2_NORMAD
        PROMPT_SUM = PROMPT_SUMMARY_NORMAD
        extract_fn = extract_answer
    else:
        PROMPT_R1 = PROMPT_ROUND1_CB
        PROMPT_R2 = PROMPT_ROUND2_CB
        PROMPT_SUM = PROMPT_SUMMARY_CB
        extract_fn = extract_answer_culturalbench

    # --- Pre-parse country & scenario/question ---
    parsed = []
    for item in dataset:
        if dataset_type == DATASET_NORMAD:
            country, cultural_bg, scenario = parse_input(item["input"])
            parsed.append({
                **item,
                "country": country,
                "cultural_bg": cultural_bg,
                "scenario": scenario,
            })
        else:
            country, question = parse_input_culturalbench(item)
            parsed.append({
                **item,
                "country": country,
                "question": question,
            })

    n = len(parsed)
    print(f"Number of cultural agents: {NUM_AGENTS}")
    print(f"Cultures: {CULTURE_NAMES}")
    print(f"Debate rounds: {args.num_rounds}")
    print(f"Total prompts per round: {NUM_AGENTS * n}")

    # --- Initialize vLLM (optimized for throughput) ---
    print("Initializing vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        max_num_seqs=256,          # allow more concurrent sequences
        enable_prefix_caching=True, # cache shared prompt prefixes across agents
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "</s>"]
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
        top_p=0.9,
    )
    summary_sampling = SamplingParams(
        temperature=max(0.1, args.temperature - 0.2),
        max_tokens=128,  # summary only needs a short answer
        stop=stop_tokens,
        top_p=0.9,
    )

    # ================================================================
    # Stage 1: Round 1 - ALL 5 agents × N samples in ONE batch
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Stage 1: Round 1 - {NUM_AGENTS}×{n} = {NUM_AGENTS*n} prompts (single batch)")
    print(f"{'='*60}")

    t1 = time.time()

    # Build all Round 1 prompts
    r1_prompts = []
    for culture_name in CULTURE_NAMES:
        persona = CULTURAL_PERSONAS[culture_name]
        for p in parsed:
            if dataset_type == DATASET_NORMAD:
                user_content = PROMPT_R1.format(
                    persona=persona,
                    country=p["country"],
                    cultural_bg=p["cultural_bg"],
                    scenario=p["scenario"],
                )
            else:
                user_content = PROMPT_R1.format(
                    persona=persona,
                    country=p["country"],
                    question=p["question"],
                )
            r1_prompts.append(apply_chat(tokenizer, user_content))

    # Single vLLM call for all Round 1 prompts
    r1_outputs = llm.generate(r1_prompts, sampling)

    # Parse results: index = culture_idx * n + sample_idx
    round1_responses = [{} for _ in range(n)]
    for culture_idx, culture_name in enumerate(CULTURE_NAMES):
        for sample_idx in range(n):
            flat_idx = culture_idx * n + sample_idx
            resp = r1_outputs[flat_idx].outputs[0].text.strip()
            round1_responses[sample_idx][culture_name] = resp

    print(f"  Round 1 done in {time.time()-t1:.1f}s")

    # ================================================================
    # Stage 2: Round 2 - ALL 5 agents × N samples in ONE batch
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Stage 2: Round 2 - {NUM_AGENTS}×{n} = {NUM_AGENTS*n} prompts (single batch)")
    print(f"{'='*60}")

    t2 = time.time()

    r2_prompts = []
    for culture_name in CULTURE_NAMES:
        persona = CULTURAL_PERSONAS[culture_name]
        for sample_idx, p in enumerate(parsed):
            other_resp_text = format_other_responses(
                round1_responses[sample_idx], culture_name
            )
            if dataset_type == DATASET_NORMAD:
                user_content = PROMPT_R2.format(
                    persona=persona,
                    country=p["country"],
                    cultural_bg=p["cultural_bg"],
                    scenario=p["scenario"],
                    other_responses=other_resp_text,
                    scgrd=SCGRD_PROMPT,
                )
            else:
                user_content = PROMPT_R2.format(
                    persona=persona,
                    country=p["country"],
                    question=p["question"],
                    other_responses=other_resp_text,
                    scgrd=SCGRD_PROMPT,
                )
            r2_prompts.append(apply_chat(tokenizer, user_content))

    # Single vLLM call for all Round 2 prompts
    r2_outputs = llm.generate(r2_prompts, sampling)

    round2_responses = [{} for _ in range(n)]
    for culture_idx, culture_name in enumerate(CULTURE_NAMES):
        for sample_idx in range(n):
            flat_idx = culture_idx * n + sample_idx
            resp = r2_outputs[flat_idx].outputs[0].text.strip()
            round2_responses[sample_idx][culture_name] = resp

    print(f"  Round 2 done in {time.time()-t2:.1f}s")

    # ================================================================
    # Stage 3: Summary - N prompts in ONE batch
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Stage 3: Summary - {n} prompts (single batch)")
    print(f"{'='*60}")

    t3 = time.time()

    # Extract Round 2 answers
    all_r2_answers = []
    for idx in range(n):
        r2_answers = {}
        for culture in CULTURE_NAMES:
            r2_answers[culture] = extract_fn(round2_responses[idx][culture])
        all_r2_answers.append(r2_answers)

    # Build summary prompts
    summary_prompts = []
    for idx, p in enumerate(parsed):
        agent_resp_text = format_agent_responses_for_summary(
            round2_responses[idx], all_r2_answers[idx], dataset_type
        )
        if dataset_type == DATASET_NORMAD:
            user_content = PROMPT_SUM.format(
                country=p["country"],
                agent_responses=agent_resp_text,
                cultural_bg=p["cultural_bg"],
                scenario=p["scenario"],
            )
        else:
            user_content = PROMPT_SUM.format(
                country=p["country"],
                agent_responses=agent_resp_text,
                question=p["question"],
            )
        summary_prompts.append(apply_chat(tokenizer, user_content))

    # Single vLLM call for summary
    summary_raw_outputs = llm.generate(summary_prompts, summary_sampling)
    summary_outputs = [o.outputs[0].text.strip() for o in summary_raw_outputs]

    print(f"  Summary done in {time.time()-t3:.1f}s")

    # ================================================================
    # Stage 4: Extract answers and build results
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 4: Extract Answers & Compute Metrics")
    print(f"{'='*60}")

    results = []
    vote_used_count = 0
    summary_used_count = 0

    for idx, p in enumerate(parsed):
        # Round 1 answers
        r1_answers = {}
        for culture in CULTURE_NAMES:
            r1_answers[culture] = extract_fn(round1_responses[idx][culture])

        # Round 2 answers (already extracted)
        r2_answers = all_r2_answers[idx]

        # Majority vote from Round 2
        vote_ans = majority_vote(r2_answers, dataset_type)

        # Summary answer
        summary_resp = summary_outputs[idx]
        summary_ans = extract_fn(summary_resp)

        # Decision logic:
        # - Strong consensus (>=4 agents agree): use vote
        # - Vote is Neither but summary gives Yes/No: prefer summary
        #   (summary has global view of cultural background and can be more decisive)
        # - Otherwise: use vote, fallback to summary
        vote_counts = Counter(v for v in r2_answers.values() if v is not None)
        vote_max_count = vote_counts.most_common(1)[0][1] if vote_counts else 0

        if vote_ans is not None and vote_max_count >= 4:
            # Strong consensus - trust the vote
            final_ans = vote_ans
            vote_used_count += 1
        elif (vote_ans == "3" and summary_ans in ("1", "2")
              and dataset_type == DATASET_NORMAD):
            # Vote says Neither but summary is decisive (Yes/No) -
            # prefer summary as it weighs cultural background more carefully
            final_ans = summary_ans
            summary_used_count += 1
        elif vote_ans is not None:
            final_ans = vote_ans
            vote_used_count += 1
        elif summary_ans is not None:
            final_ans = summary_ans
            summary_used_count += 1
        else:
            final_ans = ""

        gt = str(p.get("output", "")).strip()
        is_correct = (final_ans == gt) if final_ans else False

        record = {
            "instruction": p.get("instruction", ""),
            "input": p.get("input", ""),
            "output": gt,
            "country": p["country"],
            "scenario": p.get("scenario", p.get("question", "")),
            "round1_responses": round1_responses[idx],
            "round1_answers": r1_answers,
            "round2_responses": round2_responses[idx],
            "round2_answers": r2_answers,
            "majority_vote": vote_ans if vote_ans else "",
            "summary_response": summary_resp,
            "summary_answer": summary_ans if summary_ans else "",
            "final_answer": final_ans,
            "answer_source": "summary" if summary_ans else ("vote" if vote_ans else "none"),
            "correct": is_correct,
        }
        results.append(record)

    # ================================================================
    # Write output
    # ================================================================
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nInference results saved to: {out_json}")

    # ================================================================
    # Compute & save metrics
    # ================================================================
    metrics = compute_metrics(results)
    metrics["method"] = "MACD"
    metrics["model"] = args.model_name
    metrics["dataset_type"] = dataset_type
    metrics["num_agents"] = NUM_AGENTS
    metrics["num_rounds"] = args.num_rounds
    metrics["cultures"] = CULTURE_NAMES
    metrics["prompt_source"] = "Appendix A/B/C (MACD paper, Tan et al. 2026)"

    r1_agreement = sum(
        1 for r in results
        if len(set(v for v in r["round1_answers"].values() if v)) == 1
        and any(r["round1_answers"].values())
    )
    r2_agreement = sum(
        1 for r in results
        if len(set(v for v in r["round2_answers"].values() if v)) == 1
        and any(r["round2_answers"].values())
    )
    metrics["round1_full_agreement"] = r1_agreement
    metrics["round2_full_agreement"] = r2_agreement
    metrics["summary_used"] = summary_used_count
    metrics["vote_fallback_used"] = vote_used_count

    vote_correct = sum(1 for r in results if r["majority_vote"] == r["output"])
    metrics["vote_only_accuracy"] = vote_correct / n if n > 0 else 0.0

    total_time = time.time() - t_start
    metrics["total_time_seconds"] = round(total_time, 1)
    metrics["prompts_per_second"] = round((NUM_AGENTS * 2 * n + n) / total_time, 1)

    metrics_dir = os.path.dirname(out_metrics)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {out_metrics}")

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Dataset type: {dataset_type}")
    print(f"Final Accuracy (summary + vote fallback): {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total_samples']})")
    print(f"Vote-only Accuracy: {metrics['vote_only_accuracy']:.4f} ({vote_correct}/{n})")
    print(f"Summary used: {summary_used_count}, Vote fallback: {vote_used_count}")
    print(f"Round 1 full agreement: {r1_agreement}/{n}")
    print(f"Round 2 full agreement: {r2_agreement}/{n}")
    print(f"Total time: {total_time:.1f}s ({metrics['prompts_per_second']} prompts/s)")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MACD Baseline: Multi-Agent Cultural Debate (Tan et al., 2026)"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input dataset (normad_mas.json or culturalBench_mas.json)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama/qwen) or HuggingFace path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: /autodl-fs/data/macd)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="vLLM tensor parallel size (default: 2 for dual GPU)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (lower for more decisive answers)")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Max tokens per generation (agent responses)")
    parser.add_argument("--num_rounds", type=int, default=2,
                        help="Number of debate rounds (paper default: 2)")

    args = parser.parse_args()
    run_macd(args)


if __name__ == "__main__":
    main()
