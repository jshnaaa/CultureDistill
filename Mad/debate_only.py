"""
MAD Baseline: Debate-Only (Appendix A.3)

Two LLM agents debate over a cultural scenario. Stages:
  1. Both agents independently generate initial decisions
  2. Each agent provides feedback on the discussant's decision
  3. Each agent incorporates feedback to make a final decision
  4. If final decisions differ, a Judge LLM resolves the disagreement

No rule-of-thumb information is provided (no-rule variant).

Usage:
    # Quick test
    python MAD/debate_only.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/mad_debate_only.jsonl \
        --model_name qwen \
        --tensor_parallel_size 2 \
        --max_samples 5

    # Full dataset (llama)
    python MAD/debate_only.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/mad_debate_only_llama.jsonl \
        --model_name llama \
        --tensor_parallel_size 2 \
        --max_samples 0
"""

import os
import sys
import re
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAD.mad_common import (
    MODEL_ALIASES, ANSWER_MAP, REVERSE_ANSWER_MAP,
    load_dataset, parse_input, extract_answer,
    write_to_jsonl, reserve_unprocessed_queries,
)


# ===================================================================
# Prompt Templates (Appendix A.3, with Rule lines REMOVED)
# ===================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant with expertise in cross-cultural "
    "social norms and etiquette."
)

# A.3.1: Making Initial Decision (no rule)
PROMPT_A31_INITIAL = (
    "Task: You will be given a short story. Evaluate "
    "if the action in the story is socially acceptable in "
    "{country}. Do not make any extra "
    "inferences about the actions outside of the given context "
    "and country. Only align to the country given. Respond "
    "with an answer - Yes, No or Neither. Explain your "
    "answer in less than three sentences.\n\n"
    "Story: {story}\n"
    "Answer (Yes, No or Neither):"
)

# A.3.2: Generating Feedback (no rule)
PROMPT_A32_FEEDBACK = (
    "Task: You are currently discussing if the action in "
    "the story is socially acceptable in {country} with "
    "the other discussant.\n\n"
    "Story: {story}\n"
    "You: {your_response}\n"
    "Discussant: {other_response}\n\n"
    "Respond to the discussant by providing any relevant "
    "feedback. Respond in less than three sentences.\n"
    "Response:"
)

# A.3.3: Making Final Decision (no rule)
PROMPT_A33_FINAL = (
    "Task: You are currently discussing if the action in "
    "the story is socially acceptable in {country} with "
    "the other discussant.\n\n"
    "Story: {story}\n"
    "You: {your_response}\n"
    "Discussant: {other_response}\n"
    "Your feedback: {your_feedback}\n"
    "Discussant feedback: {other_feedback}\n\n"
    "Based on the above discussion, critically think and make "
    "your final decision. Respond with an answer - Yes, No "
    "or Neither.\n"
    "Answer (Yes, No or Neither):"
)

# A.3.4: Judge LLM Final Decision (no rule)
PROMPT_A34_JUDGE = (
    "Task: You are a judge responsible for making a "
    "final decision based on the debate history between "
    "Model1 and Model2. They have debated whether "
    "the action in the story is socially acceptable in "
    "{country}.\n"
    "Do NOT make any independent "
    "judgments; base your final decision solely on the de-"
    "bate. Respond with a final decision - Yes, No or Neither.\n\n"
    "Story: {story}\n\n"
    "*** Debate starts ***\n"
    "Model1 opinion: {model1_response}\n"
    "Model2 opinion: {model2_response}\n"
    "Model1 feedback: {model1_feedback}\n"
    "Model2 feedback: {model2_feedback}\n"
    "Model1 final decision: {model1_decision}\n"
    "Model2 final decision: {model2_decision}\n"
    "*** Debate ends ***\n\n"
    "Final decision:"
)


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


def build_prompts_batch(tokenizer, template: str, batch_kwargs: list) -> list:
    return [apply_chat(tokenizer, template.format(**kw)) for kw in batch_kwargs]


# ===================================================================
# Main inference function
# ===================================================================

def run_debate_only(args):
    # --- Resolve model ---
    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {model_path}")

    # --- Load data ---
    raw_data = load_dataset(args.input_file)
    dataset = raw_data
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
    print(f"Loaded {len(dataset)} samples from {args.input_file}")

    # --- Resume ---
    dataset = reserve_unprocessed_queries(args.output_file, dataset)
    print(f"After resume filter: {len(dataset)} samples remaining")
    if len(dataset) == 0:
        print("All samples already processed.")
        return

    # --- Pre-parse country & scenario ---
    parsed = []
    for item in dataset:
        country, scenario = parse_input(item["input"])
        parsed.append({
            **item,
            "country": country,
            "scenario": scenario,
        })

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
    results = []

    # -------- Stage 1: Initial decisions (both agents) --------
    print("\n=== Stage 1: Initial Decisions ===")
    prompts_a1 = []
    prompts_a2 = []
    for p in parsed:
        kw = {"country": p["country"], "story": p["scenario"]}
        prompts_a1.append(apply_chat(tokenizer, PROMPT_A31_INITIAL.format(**kw)))
        prompts_a2.append(apply_chat(tokenizer, PROMPT_A31_INITIAL.format(**kw)))

    for i in tqdm(range(0, len(parsed), batch_size), desc="Stage1-Init"):
        batch_end = min(i + batch_size, len(parsed))
        ba1 = prompts_a1[i:batch_end]
        ba2 = prompts_a2[i:batch_end]

        out1 = llm.generate(ba1, sampling, use_tqdm=False)
        out2 = llm.generate(ba2, sampling, use_tqdm=False)

        for j, (p, o1, o2) in enumerate(zip(
            parsed[i:batch_end], out1, out2
        )):
            resp1 = o1.outputs[0].text.strip()
            resp2 = o2.outputs[0].text.strip()
            ans1 = extract_answer(resp1)
            ans2 = extract_answer(resp2)
            parsed[i + j]["resp1_init"] = resp1
            parsed[i + j]["ans1_init"] = ans1
            parsed[i + j]["resp2_init"] = resp2
            parsed[i + j]["ans2_init"] = ans2

    # -------- Stage 2: Generate feedback --------
    print("\n=== Stage 2: Generate Feedback ===")
    prompts_fb1 = []
    prompts_fb2 = []
    for p in parsed:
        kw1 = {
            "country": p["country"], "story": p["scenario"],
            "your_response": p["resp1_init"],
            "other_response": p["resp2_init"],
        }
        kw2 = {
            "country": p["country"], "story": p["scenario"],
            "your_response": p["resp2_init"],
            "other_response": p["resp1_init"],
        }
        prompts_fb1.append(apply_chat(tokenizer, PROMPT_A32_FEEDBACK.format(**kw1)))
        prompts_fb2.append(apply_chat(tokenizer, PROMPT_A32_FEEDBACK.format(**kw2)))

    for i in tqdm(range(0, len(parsed), batch_size), desc="Stage2-Feedback"):
        batch_end = min(i + batch_size, len(parsed))
        fb1 = llm.generate(prompts_fb1[i:batch_end], sampling, use_tqdm=False)
        fb2 = llm.generate(prompts_fb2[i:batch_end], sampling, use_tqdm=False)

        for j in range(batch_end - i):
            parsed[i + j]["fb1"] = fb1[j].outputs[0].text.strip()
            parsed[i + j]["fb2"] = fb2[j].outputs[0].text.strip()

    # -------- Stage 3: Final decisions --------
    print("\n=== Stage 3: Final Decisions ===")
    prompts_final1 = []
    prompts_final2 = []
    for p in parsed:
        kw1 = {
            "country": p["country"], "story": p["scenario"],
            "your_response": p["resp1_init"],
            "other_response": p["resp2_init"],
            "your_feedback": p["fb1"],
            "other_feedback": p["fb2"],
        }
        kw2 = {
            "country": p["country"], "story": p["scenario"],
            "your_response": p["resp2_init"],
            "other_response": p["resp1_init"],
            "your_feedback": p["fb2"],
            "other_feedback": p["fb1"],
        }
        prompts_final1.append(apply_chat(tokenizer, PROMPT_A33_FINAL.format(**kw1)))
        prompts_final2.append(apply_chat(tokenizer, PROMPT_A33_FINAL.format(**kw2)))

    for i in tqdm(range(0, len(parsed), batch_size), desc="Stage3-Final"):
        batch_end = min(i + batch_size, len(parsed))
        f1 = llm.generate(prompts_final1[i:batch_end], sampling, use_tqdm=False)
        f2 = llm.generate(prompts_final2[i:batch_end], sampling, use_tqdm=False)

        for j in range(batch_end - i):
            idx = i + j
            rf1 = f1[j].outputs[0].text.strip()
            rf2 = f2[j].outputs[0].text.strip()
            parsed[idx]["resp1_final"] = rf1
            parsed[idx]["ans1_final"] = extract_answer(rf1)
            parsed[idx]["resp2_final"] = rf2
            parsed[idx]["ans2_final"] = extract_answer(rf2)

    # -------- Stage 4: Judge for disagreements --------
    print("\n=== Stage 4: Judge Resolution ===")
    disagree_indices = []
    for i, p in enumerate(parsed):
        if p["ans1_final"] != p["ans2_final"]:
            disagree_indices.append(i)
        else:
            # Agents agree: no judge needed
            parsed[i]["resp_judge"] = ""
            parsed[i]["ans_judge"] = p["ans1_final"]

    print(f"Agreements: {len(parsed) - len(disagree_indices)}, "
          f"Disagreements: {len(disagree_indices)}")

    if disagree_indices:
        judge_prompts = []
        for idx in disagree_indices:
            p = parsed[idx]
            kw = {
                "country": p["country"],
                "story": p["scenario"],
                "model1_response": p["resp1_init"],
                "model2_response": p["resp2_init"],
                "model1_feedback": p["fb1"],
                "model2_feedback": p["fb2"],
                "model1_decision": p["resp1_final"],
                "model2_decision": p["resp2_final"],
            }
            judge_prompts.append(apply_chat(tokenizer, PROMPT_A34_JUDGE.format(**kw)))

        for i in tqdm(range(0, len(disagree_indices), batch_size), desc="Stage4-Judge"):
            batch_end = min(i + batch_size, len(disagree_indices))
            batch_prompts = judge_prompts[i:batch_end]
            j_out = llm.generate(batch_prompts, sampling, use_tqdm=False)

            for j, (didx, jo) in enumerate(zip(disagree_indices[i:batch_end], j_out)):
                resp = jo.outputs[0].text.strip()
                parsed[didx]["resp_judge"] = resp
                parsed[didx]["ans_judge"] = extract_answer(resp)

    # -------- Write results --------
    write_count = 0
    correct_count = 0
    for p in parsed:
        gt = str(p.get("output", "")).strip()
        final_ans = p.get("ans_judge", "")
        is_correct = (final_ans == gt) if final_ans else False
        if is_correct:
            correct_count += 1

        output = {
            "instruction": p.get("instruction", ""),
            "input": p.get("input", ""),
            "output": gt,
            "country": p.get("country", ""),
            "model1_init": p.get("resp1_init", ""),
            "model1_ans_init": p.get("ans1_init", ""),
            "model2_init": p.get("resp2_init", ""),
            "model2_ans_init": p.get("ans2_init", ""),
            "model1_feedback": p.get("fb1", ""),
            "model2_feedback": p.get("fb2", ""),
            "model1_final": p.get("resp1_final", ""),
            "model1_ans_final": p.get("ans1_final", ""),
            "model2_final": p.get("resp2_final", ""),
            "model2_ans_final": p.get("ans2_final", ""),
            "judge_response": p.get("resp_judge", ""),
            "final_answer": final_ans,
            "correct": is_correct,
            "agree": (p.get("ans1_final", "") == p.get("ans2_final", "")),
        }
        write_to_jsonl(args.output_file, output)
        write_count += 1

    acc = correct_count / write_count if write_count > 0 else 0.0
    print(f"\nDone. {write_count} samples written to {args.output_file}")
    print(f"Accuracy: {correct_count}/{write_count} = {acc:.4f}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MAD Baseline: Debate-Only (A.3, no-rule variant)"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to normad_mas.json")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama/qwen) or HF path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples (0=all)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)

    args = parser.parse_args()
    run_debate_only(args)


if __name__ == "__main__":
    main()
