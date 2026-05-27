"""
MAD Baseline: Self-Reflect+Debate (Appendix A.4)

Two LLM agents debate with dynamic choice between self-reflection and debate. Stages:
  1. Both agents independently generate initial decisions
  2. Each agent chooses to (A) self-reflect or (B) debate
  3. Each agent generates the chosen response (reflection or feedback)
  4. Each agent incorporates responses to make a final decision
  5. If final decisions differ, Judge LLM resolves the disagreement

No rule-of-thumb information is provided (no-rule variant).

Usage:
    # Quick test
    python MAD/self_reflect_debate.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/mad_self_reflect_debate.jsonl \
        --model_name qwen \
        --tensor_parallel_size 2 \
        --max_samples 5

    # Full dataset (llama)
    python MAD/self_reflect_debate.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/mad_srd_llama.jsonl \
        --model_name llama \
        --tensor_parallel_size 2 \
        --max_samples 0
"""

import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAD.mad_common import (
    MODEL_ALIASES, ANSWER_MAP, REVERSE_ANSWER_MAP,
    load_dataset, parse_input, extract_answer, extract_choice,
    write_to_jsonl, reserve_unprocessed_queries,
)


# ===================================================================
# Prompt Templates (Appendix A.4, with Rule lines REMOVED)
# ===================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant with expertise in cross-cultural "
    "social norms and etiquette."
)

# A.4.1: Making Initial Decision (identical to A.3.1, no rule)
PROMPT_A41_INITIAL = (
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

# A.4.2: Choose to Self-Reflect or Debate (no rule)
PROMPT_A42_CHOOSE = (
    "Task: You are currently discussing if the action in "
    "the story is socially acceptable in {country} with "
    "the other discussant.\n\n"
    "Story: {story}\n"
    "You: {your_response}\n"
    "Discussant: {other_response}\n\n"
    "You can choose to (A) reflect on your response or (B) "
    "respond to the discussant by providing any relevant feed-"
    "back. Respond with your choice - (A) reflect or (B) "
    "respond to the discussant. Only respond your choice as "
    "(A) or (B).\n"
    "Response:"
)

# A.4.3: If Model Chooses to (A) Reflect (no rule)
PROMPT_A43_REFLECT = (
    "Task: You are currently discussing if the action in "
    "the story is socially acceptable in {country} with "
    "the other discussant.\n\n"
    "Story: {story}\n"
    "You: {your_response}\n\n"
    "You chose to reflect on your response. Respond in less "
    "than three sentences.\n"
    "Response:"
)

# A.4.4: If Model Chooses to (B) Debate (no rule)
PROMPT_A44_DEBATE = (
    "Task: You are currently discussing if the action in "
    "the story is socially acceptable in {country} with "
    "the other discussant.\n\n"
    "Story: {story}\n"
    "You: {your_response}\n"
    "Discussant: {other_response}\n\n"
    "You chose to respond to the discussant by providing any "
    "relevant feedback. Respond in less than three sentences.\n"
    "Response:"
)

# A.4.5: Making Final Decision (no rule)
# {your_action} and {other_action} are filled with:
#   "reflect on your response" / "provide feedback to the discussant"
PROMPT_A45_FINAL = (
    "Task: You are currently discussing if the action in "
    "the story is socially acceptable in {country} with "
    "the other discussant. You chose "
    "to {your_action}."
    " Your discussant chose "
    "to {other_action}.\n\n"
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

# A.4.6: Judge LLM Final Decision (same as A.3.4, no rule)
PROMPT_A46_JUDGE = (
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

def run_self_reflect_debate(args):
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

    # -------- Stage 1: Initial decisions (both agents) --------
    print("\n=== Stage 1: Initial Decisions ===")
    prompts_a1 = []
    prompts_a2 = []
    for p in parsed:
        kw = {"country": p["country"], "story": p["scenario"]}
        prompts_a1.append(apply_chat(tokenizer, PROMPT_A41_INITIAL.format(**kw)))
        prompts_a2.append(apply_chat(tokenizer, PROMPT_A41_INITIAL.format(**kw)))

    for i in tqdm(range(0, len(parsed), batch_size), desc="Stage1-Init"):
        batch_end = min(i + batch_size, len(parsed))
        ba1 = prompts_a1[i:batch_end]
        ba2 = prompts_a2[i:batch_end]

        out1 = llm.generate(ba1, sampling, use_tqdm=False)
        out2 = llm.generate(ba2, sampling, use_tqdm=False)

        for j in range(batch_end - i):
            idx = i + j
            parsed[idx]["resp1_init"] = out1[j].outputs[0].text.strip()
            parsed[idx]["ans1_init"] = extract_answer(out1[j].outputs[0].text)
            parsed[idx]["resp2_init"] = out2[j].outputs[0].text.strip()
            parsed[idx]["ans2_init"] = extract_answer(out2[j].outputs[0].text)

    # -------- Stage 2: Choose self-reflect or debate --------
    print("\n=== Stage 2: Choose Self-Reflect or Debate ===")
    prompts_choose1 = []
    prompts_choose2 = []
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
        prompts_choose1.append(apply_chat(tokenizer, PROMPT_A42_CHOOSE.format(**kw1)))
        prompts_choose2.append(apply_chat(tokenizer, PROMPT_A42_CHOOSE.format(**kw2)))

    for i in tqdm(range(0, len(parsed), batch_size), desc="Stage2-Choose"):
        batch_end = min(i + batch_size, len(parsed))
        c1 = llm.generate(prompts_choose1[i:batch_end], sampling, use_tqdm=False)
        c2 = llm.generate(prompts_choose2[i:batch_end], sampling, use_tqdm=False)

        for j in range(batch_end - i):
            idx = i + j
            ch1 = c1[j].outputs[0].text.strip()
            ch2 = c2[j].outputs[0].text.strip()
            parsed[idx]["choice1"] = extract_choice(ch1)
            parsed[idx]["choice2"] = extract_choice(ch2)

    # -------- Stage 3: Execute chosen action (Reflect or Debate) --------
    print("\n=== Stage 3: Execute Chosen Action ===")
    # Build prompts based on choices
    action_prompts1 = []
    action_prompts2 = []
    action_types1 = []  # "reflect" or "debate"
    action_types2 = []

    for p in parsed:
        c1 = p["choice1"]
        c2 = p["choice2"]

        if c1 == "A":
            kw1 = {"country": p["country"], "story": p["scenario"],
                   "your_response": p["resp1_init"]}
            action_prompts1.append(apply_chat(tokenizer, PROMPT_A43_REFLECT.format(**kw1)))
            action_types1.append("reflect")
        else:
            kw1 = {"country": p["country"], "story": p["scenario"],
                   "your_response": p["resp1_init"],
                   "other_response": p["resp2_init"]}
            action_prompts1.append(apply_chat(tokenizer, PROMPT_A44_DEBATE.format(**kw1)))
            action_types1.append("debate")

        if c2 == "A":
            kw2 = {"country": p["country"], "story": p["scenario"],
                   "your_response": p["resp2_init"]}
            action_prompts2.append(apply_chat(tokenizer, PROMPT_A43_REFLECT.format(**kw2)))
            action_types2.append("reflect")
        else:
            kw2 = {"country": p["country"], "story": p["scenario"],
                   "your_response": p["resp2_init"],
                   "other_response": p["resp1_init"]}
            action_prompts2.append(apply_chat(tokenizer, PROMPT_A44_DEBATE.format(**kw2)))
            action_types2.append("debate")

    for i in tqdm(range(0, len(parsed), batch_size), desc="Stage3-Action"):
        batch_end = min(i + batch_size, len(parsed))
        a1 = llm.generate(action_prompts1[i:batch_end], sampling, use_tqdm=False)
        a2 = llm.generate(action_prompts2[i:batch_end], sampling, use_tqdm=False)

        for j in range(batch_end - i):
            idx = i + j
            parsed[idx]["action_response1"] = a1[j].outputs[0].text.strip()
            parsed[idx]["action_type1"] = action_types1[i + j]
            parsed[idx]["action_response2"] = a2[j].outputs[0].text.strip()
            parsed[idx]["action_type2"] = action_types2[i + j]

    # -------- Stage 4: Final decisions --------
    print("\n=== Stage 4: Final Decisions ===")
    prompts_final1 = []
    prompts_final2 = []

    for p in parsed:
        # Your action description
        ya1 = ("reflect on your response" if p["action_type1"] == "reflect"
               else "provide feedback to the discussant")
        oa1 = ("reflect on their response" if p["action_type2"] == "reflect"
               else "provide feedback to you")
        ya2 = ("reflect on your response" if p["action_type2"] == "reflect"
               else "provide feedback to the discussant")
        oa2 = ("reflect on their response" if p["action_type1"] == "reflect"
               else "provide feedback to you")

        kw1 = {
            "country": p["country"], "story": p["scenario"],
            "your_action": ya1, "other_action": oa1,
            "your_response": p["resp1_init"],
            "other_response": p["resp2_init"],
            "your_feedback": p["action_response1"],
            "other_feedback": p["action_response2"],
        }
        kw2 = {
            "country": p["country"], "story": p["scenario"],
            "your_action": ya2, "other_action": oa2,
            "your_response": p["resp2_init"],
            "other_response": p["resp1_init"],
            "your_feedback": p["action_response2"],
            "other_feedback": p["action_response1"],
        }
        prompts_final1.append(apply_chat(tokenizer, PROMPT_A45_FINAL.format(**kw1)))
        prompts_final2.append(apply_chat(tokenizer, PROMPT_A45_FINAL.format(**kw2)))

    for i in tqdm(range(0, len(parsed), batch_size), desc="Stage4-Final"):
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

    # -------- Stage 5: Judge for disagreements --------
    print("\n=== Stage 5: Judge Resolution ===")
    disagree_indices = []
    for i, p in enumerate(parsed):
        if p["ans1_final"] != p["ans2_final"]:
            disagree_indices.append(i)
        else:
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
                "model1_feedback": p["action_response1"],
                "model2_feedback": p["action_response2"],
                "model1_decision": p["resp1_final"],
                "model2_decision": p["resp2_final"],
            }
            judge_prompts.append(apply_chat(tokenizer, PROMPT_A46_JUDGE.format(**kw)))

        for i in tqdm(range(0, len(disagree_indices), batch_size), desc="Stage5-Judge"):
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
    reflect_count = 0
    debate_count = 0

    for p in parsed:
        gt = str(p.get("output", "")).strip()
        final_ans = p.get("ans_judge", "")
        is_correct = (final_ans == gt) if final_ans else False
        if is_correct:
            correct_count += 1

        # Count choices
        if p.get("choice1") == "A":
            reflect_count += 1
        else:
            debate_count += 1
        if p.get("choice2") == "A":
            reflect_count += 1
        else:
            debate_count += 1

        output = {
            "instruction": p.get("instruction", ""),
            "input": p.get("input", ""),
            "output": gt,
            "country": p.get("country", ""),
            "model1_init": p.get("resp1_init", ""),
            "model1_ans_init": p.get("ans1_init", ""),
            "model2_init": p.get("resp2_init", ""),
            "model2_ans_init": p.get("ans2_init", ""),
            "model1_choice": p.get("choice1", ""),
            "model2_choice": p.get("choice2", ""),
            "model1_action_type": p.get("action_type1", ""),
            "model1_action_response": p.get("action_response1", ""),
            "model2_action_type": p.get("action_type2", ""),
            "model2_action_response": p.get("action_response2", ""),
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
    total_choices = reflect_count + debate_count
    print(f"\nDone. {write_count} samples written to {args.output_file}")
    print(f"Accuracy: {correct_count}/{write_count} = {acc:.4f}")
    if total_choices > 0:
        print(f"Choices: reflect={reflect_count} ({reflect_count/total_choices:.1%}), "
              f"debate={debate_count} ({debate_count/total_choices:.1%})")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MAD Baseline: Self-Reflect+Debate (A.4, no-rule variant)"
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
    run_self_reflect_debate(args)


if __name__ == "__main__":
    main()
