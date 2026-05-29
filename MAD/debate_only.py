"""
MAD Baseline: Debate-Only (Appendix A.3)

Two LLM agents debate over a cultural scenario. Stages:
  1. Both agents independently generate initial decisions
  2. Each agent provides feedback on the discussant's decision
  3. Each agent incorporates feedback to make a final decision
  4. If final decisions differ, a Judge LLM resolves the disagreement

No rule-of-thumb information is provided (no-rule variant).

Output naming: {dataset}_{MAD}_{debateonly}_{model}.json + _metrics.json

Usage:
    # Quick test (5 samples)
    python MAD/debate_only.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --model_name qwen \
        --tensor_parallel_size 2 \
        --max_samples 5

    # Full dataset
    python MAD/debate_only.py \
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

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAD.mad_common import (
    MODEL_ALIASES, ANSWER_MAP, REVERSE_ANSWER_MAP,
    load_dataset, parse_input, extract_answer,
    infer_output_path, compute_metrics,
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


# ===================================================================
# Main inference function
# ===================================================================

def run_debate_only(args):
    # --- Resolve model ---
    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {model_path}")

    # --- Determine output paths ---
    out_json, out_metrics = infer_output_path(
        args.input_file, "MAD", "debateonly", args.model_name, args.output_dir
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

    # -------- Stage 1: Initial decisions (both agents) --------
    print("\n=== Stage 1: Initial Decisions ===")
    prompts_a1 = []
    prompts_a2 = []
    for p in parsed:
        kw = {"country": p["country"], "story": p["scenario"]}
        prompts_a1.append(apply_chat(tokenizer, PROMPT_A31_INITIAL.format(**kw)))
        prompts_a2.append(apply_chat(tokenizer, PROMPT_A31_INITIAL.format(**kw)))

    for i in tqdm(range(0, n, batch_size), desc="Stage1-Init"):
        batch_end = min(i + batch_size, n)
        ba1 = prompts_a1[i:batch_end]
        ba2 = prompts_a2[i:batch_end]

        out1 = llm.generate(ba1, sampling, use_tqdm=False)
        out2 = llm.generate(ba2, sampling, use_tqdm=False)

        for j in range(batch_end - i):
            idx = i + j
            r1 = out1[j].outputs[0].text.strip()
            r2 = out2[j].outputs[0].text.strip()
            parsed[idx]["model1_initial"] = r1
            parsed[idx]["model1_initial_ans"] = extract_answer(r1)
            parsed[idx]["model2_initial"] = r2
            parsed[idx]["model2_initial_ans"] = extract_answer(r2)

    # -------- Stage 2: Generate feedback --------
    print("\n=== Stage 2: Generate Feedback ===")
    prompts_fb1 = []
    prompts_fb2 = []
    for p in parsed:
        kw1 = {
            "country": p["country"], "story": p["scenario"],
            "your_response": p["model1_initial"],
            "other_response": p["model2_initial"],
        }
        kw2 = {
            "country": p["country"], "story": p["scenario"],
            "your_response": p["model2_initial"],
            "other_response": p["model1_initial"],
        }
        prompts_fb1.append(apply_chat(tokenizer, PROMPT_A32_FEEDBACK.format(**kw1)))
        prompts_fb2.append(apply_chat(tokenizer, PROMPT_A32_FEEDBACK.format(**kw2)))

    for i in tqdm(range(0, n, batch_size), desc="Stage2-Feedback"):
        batch_end = min(i + batch_size, n)
        fb1 = llm.generate(prompts_fb1[i:batch_end], sampling, use_tqdm=False)
        fb2 = llm.generate(prompts_fb2[i:batch_end], sampling, use_tqdm=False)

        for j in range(batch_end - i):
            parsed[i + j]["model1_feedback"] = fb1[j].outputs[0].text.strip()
            parsed[i + j]["model2_feedback"] = fb2[j].outputs[0].text.strip()

    # -------- Stage 3: Final decisions --------
    print("\n=== Stage 3: Final Decisions ===")
    prompts_final1 = []
    prompts_final2 = []
    for p in parsed:
        kw1 = {
            "country": p["country"], "story": p["scenario"],
            "your_response": p["model1_initial"],
            "other_response": p["model2_initial"],
            "your_feedback": p["model1_feedback"],
            "other_feedback": p["model2_feedback"],
        }
        kw2 = {
            "country": p["country"], "story": p["scenario"],
            "your_response": p["model2_initial"],
            "other_response": p["model1_initial"],
            "your_feedback": p["model2_feedback"],
            "other_feedback": p["model1_feedback"],
        }
        prompts_final1.append(apply_chat(tokenizer, PROMPT_A33_FINAL.format(**kw1)))
        prompts_final2.append(apply_chat(tokenizer, PROMPT_A33_FINAL.format(**kw2)))

    for i in tqdm(range(0, n, batch_size), desc="Stage3-Final"):
        batch_end = min(i + batch_size, n)
        f1 = llm.generate(prompts_final1[i:batch_end], sampling, use_tqdm=False)
        f2 = llm.generate(prompts_final2[i:batch_end], sampling, use_tqdm=False)

        for j in range(batch_end - i):
            idx = i + j
            rf1 = f1[j].outputs[0].text.strip()
            rf2 = f2[j].outputs[0].text.strip()
            parsed[idx]["model1_final"] = rf1
            parsed[idx]["model1_final_ans"] = extract_answer(rf1)
            parsed[idx]["model2_final"] = rf2
            parsed[idx]["model2_final_ans"] = extract_answer(rf2)

    # -------- Stage 4: Judge for disagreements --------
    print("\n=== Stage 4: Judge Resolution ===")
    disagree_indices = []
    for i, p in enumerate(parsed):
        m1_ans = p.get("model1_final_ans")
        m2_ans = p.get("model2_final_ans")
        if m1_ans != m2_ans:
            disagree_indices.append(i)
        else:
            # Agents agree: no judge needed, use their consensus answer
            parsed[i]["judge_response"] = ""
            parsed[i]["judge_ans"] = m1_ans if m1_ans is not None else ""

    agree_count = n - len(disagree_indices)
    print(f"Agreements: {agree_count}, Disagreements: {len(disagree_indices)}")

    if disagree_indices:
        judge_prompts = []
        for idx in disagree_indices:
            p = parsed[idx]
            kw = {
                "country": p["country"],
                "story": p["scenario"],
                "model1_response": p["model1_initial"],
                "model2_response": p["model2_initial"],
                "model1_feedback": p["model1_feedback"],
                "model2_feedback": p["model2_feedback"],
                "model1_decision": p["model1_final"],
                "model2_decision": p["model2_final"],
            }
            judge_prompts.append(apply_chat(tokenizer, PROMPT_A34_JUDGE.format(**kw)))

        for i in tqdm(range(0, len(disagree_indices), batch_size), desc="Stage4-Judge"):
            batch_end = min(i + batch_size, len(disagree_indices))
            j_out = llm.generate(judge_prompts[i:batch_end], sampling, use_tqdm=False)

            for j, (didx, jo) in enumerate(zip(disagree_indices[i:batch_end], j_out)):
                resp = jo.outputs[0].text.strip()
                parsed[didx]["judge_response"] = resp
                parsed[didx]["judge_ans"] = extract_answer(resp)

    # -------- Build results & write final output --------
    print("\n=== Writing output ===")

    results = []
    for p in parsed:
        gt = str(p.get("output", "")).strip()
        final_ans = p.get("judge_ans") or ""
        is_correct = (final_ans == gt) if final_ans else False

        record = {
            "instruction": p.get("instruction", ""),
            "input": p.get("input", ""),
            "output": gt,
            "country": p["country"],
            "scenario": p["scenario"],
            "model1_initial": p["model1_initial"],
            "model1_initial_ans": p.get("model1_initial_ans", ""),
            "model2_initial": p["model2_initial"],
            "model2_initial_ans": p.get("model2_initial_ans", ""),
            "model1_feedback": p["model1_feedback"],
            "model2_feedback": p["model2_feedback"],
            "model1_final": p["model1_final"],
            "model1_final_ans": p.get("model1_final_ans", ""),
            "model2_final": p["model2_final"],
            "model2_final_ans": p.get("model2_final_ans", ""),
            "judge_response": p.get("judge_response", ""),
            "final_answer": final_ans,
            "correct": is_correct,
            "agree": (p.get("model1_final_ans", "") == p.get("model2_final_ans", "")),
        }
        results.append(record)

    # Write JSON
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Inference results saved to: {out_json}")

    # -------- Compute & save metrics --------
    metrics = compute_metrics(results)
    metrics["method"] = "MAD"
    metrics["variant"] = "debateonly"
    metrics["model"] = args.model_name
    metrics["prompt_source"] = "Appendix A.3 (no rule-of-thumb)"
    metrics["agree_count"] = agree_count
    metrics["disagree_count"] = n - agree_count

    metrics_dir = os.path.dirname(out_metrics)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {out_metrics}")
    print(f"Accuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total_samples']})")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MAD Baseline: Debate-Only (A.3, no-rule variant)"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to normad_mas.json")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama/qwen) or HF path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input_file)")
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
