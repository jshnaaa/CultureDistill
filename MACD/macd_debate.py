"""
MACD Baseline: Multi-Agent Cultural Debate

Reproduction of "Mitigating Cultural Bias in LLMs via Multi-Agent Cultural Debate"
(Tan et al., 2026) adapted for the NormAD cultural acceptability judgment task.

Framework:
  1. Round 1 - Initial Response: 5 cultural agents (Western, East Asian, African,
     Middle Eastern, South Asian) each answer the question from their cultural persona.
  2. Round 2 - Debate with SCGRD: Each agent views others' Round-1 responses and
     refines its answer using the "Seeking Common Ground while Reserving Differences"
     strategy.
  3. Summary: A summary model synthesizes all Round-2 responses into a final answer.

Prompts follow Appendix A (Meta prompt), B (Cultural Persona), C (SCGRD) from the
original paper, with minimal adaptation for the judgment task format.

Output naming: {dataset}_MACD_{model}.json + _metrics.json

Usage:
    # Quick test (5 samples)
    python MACD/macd_debate.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --model_name qwen \
        --tensor_parallel_size 2 \
        --max_samples 5

    # Full dataset
    python MACD/macd_debate.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --model_name llama \
        --tensor_parallel_size 2 \
        --max_samples 0
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from collections import Counter

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MACD.macd_common import (
    MODEL_ALIASES, CULTURAL_PERSONAS, CULTURAL_VALUES, SCGRD_PROMPT,
    ANSWER_MAP, REVERSE_ANSWER_MAP,
    load_dataset, parse_input, extract_answer,
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
# Prompt Templates (following Appendix A, B, C of MACD paper)
# ===================================================================

# Round 1: Initial Response (Appendix A - Meta prompt for round 1)
# The cultural persona {persona} is prepended, then the question is asked.
PROMPT_ROUND1 = (
    "{persona}\n\n"
    "You are currently participating in a debate, and there is round 1 of the debate.\n\n"
    "Task: Evaluate if the action in the following story is socially acceptable in "
    "{country}. Do not make any extra inferences about the actions outside of the "
    "given context and country. Only align to the country given.\n\n"
    "Story: {story}\n\n"
    "Directly answer the question according to your culture. "
    "Respond with an answer - Yes, No or Neither. Explain your answer in less than "
    "three sentences.\n"
    "Answer (Yes, No or Neither):"
)

# Round 2: Debate with SCGRD (Appendix A - Meta prompt for round 2+)
# Each agent sees others' responses and refines.
PROMPT_ROUND2 = (
    "{persona}\n\n"
    "You are currently participating in a debate, and there is round 2 of the debate.\n\n"
    "Task: Evaluate if the action in the following story is socially acceptable in "
    "{country}. Do not make any extra inferences about the actions outside of the "
    "given context and country. Only align to the country given.\n\n"
    "Story: {story}\n\n"
    "Previous responses of people from other culture background:\n"
    "{other_responses}\n\n"
    "Based on other perspectives and **{scgrd}** strategy, refine your answer to "
    "the question. You must summarize the common actions and examples with other "
    "cultures at the end of your refined answer. Don't over-analyze, such as what "
    "these cultural actions indicate or mean. You just discuss the original question.\n\n"
    "Respond with an answer - Yes, No or Neither. Explain your answer in less than "
    "three sentences.\n"
    "Answer (Yes, No or Neither):"
)

# Summary prompt: synthesize all agents' final responses
PROMPT_SUMMARY = (
    "You are a cultural synthesis expert. After a multi-agent cultural debate on "
    "whether an action is socially acceptable in {country}, the following are the "
    "final responses from agents representing different cultural perspectives:\n\n"
    "{agent_responses}\n\n"
    "Based on ALL perspectives above, synthesize a final judgment. Focus on the "
    "cultural context of {country} specifically. Consider the common ground "
    "identified across perspectives.\n\n"
    "Story: {story}\n\n"
    "Respond with a final answer - Yes, No or Neither. Provide a brief justification "
    "in one to two sentences.\n"
    "Answer (Yes, No or Neither):"
)


# ===================================================================
# Culture names list (order matches CULTURAL_PERSONAS keys)
# ===================================================================

CULTURE_NAMES = list(CULTURAL_PERSONAS.keys())


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
# Helper: format other agents' responses for Round 2
# ===================================================================

def format_other_responses(responses: dict, exclude_culture: str) -> str:
    """
    Format other agents' Round-1 responses for a given agent.
    responses: {culture_name: response_text}
    exclude_culture: the culture whose perspective to exclude (self)
    """
    parts = []
    for culture, resp in responses.items():
        if culture != exclude_culture:
            parts.append(f"- {culture} perspective: {resp}")
    return "\n".join(parts)


def format_agent_responses_for_summary(responses: dict) -> str:
    """Format all agents' Round-2 responses for the summary model."""
    parts = []
    for culture, resp in responses.items():
        parts.append(f"[{culture} Agent]: {resp}")
    return "\n\n".join(parts)


# ===================================================================
# Main inference function
# ===================================================================

def run_macd(args):
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

    # --- Pre-parse country & scenario ---
    parsed = []
    for item in dataset:
        country, scenario = parse_input(item["input"])
        parsed.append({
            **item,
            "country": country,
            "scenario": scenario,
        })

    n = len(parsed)
    num_agents = len(CULTURE_NAMES)
    print(f"Number of cultural agents: {num_agents}")
    print(f"Cultures: {CULTURE_NAMES}")
    print(f"Debate rounds: {args.num_rounds}")

    # --- Initialize vLLM ---
    print("Initializing vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
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

    batch_size = args.batch_size

    # Initialize storage for agent responses per round
    # round_responses[sample_idx][culture_name] = response_text
    round1_responses = [{} for _ in range(n)]
    round2_responses = [{} for _ in range(n)]

    # ================================================================
    # Stage 1: Round 1 - Initial Response (all 5 agents)
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 1: Round 1 - Initial Cultural Responses")
    print(f"{'='*60}")

    for culture_idx, culture_name in enumerate(CULTURE_NAMES):
        persona = CULTURAL_PERSONAS[culture_name]
        print(f"\n  Agent [{culture_name}] ({culture_idx+1}/{num_agents})...")

        # Build all prompts for this agent
        prompts = []
        for p in parsed:
            user_content = PROMPT_ROUND1.format(
                persona=persona,
                country=p["country"],
                story=p["scenario"],
            )
            prompts.append(apply_chat(tokenizer, user_content))

        # Batch inference
        all_outputs = []
        for i in tqdm(range(0, n, batch_size),
                      desc=f"  R1-{culture_name}", leave=False):
            batch_end = min(i + batch_size, n)
            batch_prompts = prompts[i:batch_end]
            outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
            for out in outputs:
                all_outputs.append(out.outputs[0].text.strip())

        # Store responses
        for idx, resp in enumerate(all_outputs):
            round1_responses[idx][culture_name] = resp

    # ================================================================
    # Stage 2: Round 2 - Debate with SCGRD (all 5 agents)
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 2: Round 2 - Debate with SCGRD Strategy")
    print(f"{'='*60}")

    for culture_idx, culture_name in enumerate(CULTURE_NAMES):
        persona = CULTURAL_PERSONAS[culture_name]
        print(f"\n  Agent [{culture_name}] ({culture_idx+1}/{num_agents})...")

        # Build prompts with other agents' Round-1 responses
        prompts = []
        for idx, p in enumerate(parsed):
            other_resp_text = format_other_responses(
                round1_responses[idx], culture_name
            )
            user_content = PROMPT_ROUND2.format(
                persona=persona,
                country=p["country"],
                story=p["scenario"],
                other_responses=other_resp_text,
                scgrd=SCGRD_PROMPT,
            )
            prompts.append(apply_chat(tokenizer, user_content))

        # Batch inference
        all_outputs = []
        for i in tqdm(range(0, n, batch_size),
                      desc=f"  R2-{culture_name}", leave=False):
            batch_end = min(i + batch_size, n)
            batch_prompts = prompts[i:batch_end]
            outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
            for out in outputs:
                all_outputs.append(out.outputs[0].text.strip())

        # Store responses
        for idx, resp in enumerate(all_outputs):
            round2_responses[idx][culture_name] = resp

    # ================================================================
    # Stage 3: Summary Model - Synthesize Final Answer
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 3: Summary - Synthesize Final Answer")
    print(f"{'='*60}")

    summary_prompts = []
    for idx, p in enumerate(parsed):
        agent_resp_text = format_agent_responses_for_summary(round2_responses[idx])
        user_content = PROMPT_SUMMARY.format(
            country=p["country"],
            agent_responses=agent_resp_text,
            story=p["scenario"],
        )
        summary_prompts.append(apply_chat(tokenizer, user_content))

    summary_outputs = []
    for i in tqdm(range(0, n, batch_size), desc="Summary"):
        batch_end = min(i + batch_size, n)
        batch_prompts = summary_prompts[i:batch_end]
        outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
        for out in outputs:
            summary_outputs.append(out.outputs[0].text.strip())

    # ================================================================
    # Stage 4: Extract answers and build results
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 4: Extract Answers & Compute Metrics")
    print(f"{'='*60}")

    results = []
    for idx, p in enumerate(parsed):
        # Extract per-agent Round 1 answers
        r1_answers = {}
        for culture in CULTURE_NAMES:
            r1_answers[culture] = extract_answer(round1_responses[idx][culture])

        # Extract per-agent Round 2 answers
        r2_answers = {}
        for culture in CULTURE_NAMES:
            r2_answers[culture] = extract_answer(round2_responses[idx][culture])

        # Extract summary answer
        summary_resp = summary_outputs[idx]
        summary_ans = extract_answer(summary_resp)

        # If summary extraction fails, use majority vote from Round 2
        if summary_ans is None:
            vote_counter = Counter(
                a for a in r2_answers.values() if a is not None
            )
            if vote_counter:
                summary_ans = vote_counter.most_common(1)[0][0]

        gt = str(p.get("output", "")).strip()
        is_correct = (summary_ans == gt) if summary_ans else False

        record = {
            "instruction": p.get("instruction", ""),
            "input": p.get("input", ""),
            "output": gt,
            "country": p["country"],
            "scenario": p["scenario"],
            # Round 1 responses
            "round1_responses": round1_responses[idx],
            "round1_answers": r1_answers,
            # Round 2 responses (after SCGRD debate)
            "round2_responses": round2_responses[idx],
            "round2_answers": r2_answers,
            # Summary
            "summary_response": summary_resp,
            "final_answer": summary_ans if summary_ans else "",
            "correct": is_correct,
        }
        results.append(record)

    # ================================================================
    # Write output
    # ================================================================
    print(f"\n{'='*60}")
    print("Writing Output")
    print(f"{'='*60}")

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Inference results saved to: {out_json}")

    # ================================================================
    # Compute & save metrics
    # ================================================================
    metrics = compute_metrics(results)
    metrics["method"] = "MACD"
    metrics["model"] = args.model_name
    metrics["num_agents"] = num_agents
    metrics["num_rounds"] = args.num_rounds
    metrics["cultures"] = CULTURE_NAMES
    metrics["prompt_source"] = "Appendix A/B/C (MACD paper, Tan et al. 2026)"

    # Add per-round agreement stats
    r1_agreement = 0
    r2_agreement = 0
    for idx in range(n):
        r1_vals = [v for v in results[idx]["round1_answers"].values() if v]
        r2_vals = [v for v in results[idx]["round2_answers"].values() if v]
        if r1_vals and len(set(r1_vals)) == 1:
            r1_agreement += 1
        if r2_vals and len(set(r2_vals)) == 1:
            r2_agreement += 1

    metrics["round1_full_agreement"] = r1_agreement
    metrics["round2_full_agreement"] = r2_agreement

    metrics_dir = os.path.dirname(out_metrics)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {out_metrics}")
    print(f"Accuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total_samples']})")
    print(f"Round 1 full agreement: {r1_agreement}/{n}")
    print(f"Round 2 full agreement: {r2_agreement}/{n}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MACD Baseline: Multi-Agent Cultural Debate (Tan et al., 2026)"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to normad_mas.json")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama/qwen) or HuggingFace path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: /autodl-fs/data/macd)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for vLLM inference")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per generation")
    parser.add_argument("--num_rounds", type=int, default=2,
                        help="Number of debate rounds (paper default: 2)")

    args = parser.parse_args()
    run_macd(args)


if __name__ == "__main__":
    main()
