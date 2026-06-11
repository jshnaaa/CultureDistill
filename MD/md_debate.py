"""
MD Baseline: Multiagent Debate (Du et al., 2023)

Reproduction of "Improving Factuality and Reasoning in Language Models through
Multiagent Debate". N copies of the same model act as agents. Procedure:

  Round 0 (Starting): each of the N agents independently answers the question.
  Round 1..R (Debate): each agent is given the concatenation of the OTHER
      agents' most recent responses as additional advice, and produces an
      updated answer.
  Final answer: majority vote over the agents' last-round answers
      (ties broken by Agent 1).

Prompt templates follow Appendix Figure 15 (Starting + Debate). The MMLU
template is the closest fit for our multiple-choice cultural tasks; we keep
its "use other agents' reasoning as advice -> updated answer" logic and only
adapt the surface task wording / answer format per dataset.

Supports four datasets (auto-detected from input_file name):
  - NorMAD      (normad):       Yes/No/Neither -> 1/2/3
  - CulturalBench (culturalbench): 4-choice MCQ -> 1/2/3/4
  - BLEND       (blend):        4-choice MCQ -> 1/2/3/4
  - CultureLLM  (culturellm):   World Values Survey, variable options

Output naming: {dataset}_MD_{model}_{timestamp}.json + _metrics.json

Usage:
    # Quick test (5 samples)
    python MD/md_debate.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --model_name qwen --tensor_parallel_size 2 \
        --max_samples 5

    # Full dataset
    python MD/md_debate.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --model_name llama --tensor_parallel_size 2 \
        --num_agents 3 --num_rounds 2 --max_samples 0
"""

import os
import sys
import json
import argparse

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MD.md_common import (
    MODEL_ALIASES,
    DATASET_NORMAD, DATASET_MCQ, DATASET_BLEND, DATASET_CULTURELLM,
    load_dataset, parse_input, detect_dataset_type,
    extract_answer, extract_answer_mcq, extract_answer_culturellm,
    get_culturellm_option_range,
    majority_vote, infer_output_path, compute_metrics,
)


# ===================================================================
# Prompt Templates (Appendix Figure 15: Starting + Debate)
# ===================================================================

# ---------- NorMAD (Yes/No/Neither) ----------

SYSTEM_PROMPT_NORMAD = (
    "You are a helpful assistant with expertise in cross-cultural "
    "social norms and etiquette."
)

# Starting (Figure 15, adapted to acceptability judgement)
PROMPT_NORMAD_START = (
    "You will be given a short story. Evaluate whether the action in the "
    "story is socially acceptable in {country}. Think step by step about the "
    "cultural norms of {country}, then explain your reasoning. Your final "
    "answer should be a single word - Yes, No or Neither - stated at the end "
    "of your response.\n\n"
    "Story: {story}\n"
    "Answer (Yes, No or Neither):"
)

# Debate (Figure 15 MMLU debate prompt, adapted)
PROMPT_NORMAD_DEBATE = (
    "These are the recent/updated responses to the question from other "
    "agents:\n{other_responses}\n\n"
    "Using the reasoning from other agents as additional advice, can you give "
    "an updated answer about whether the action in the story is socially "
    "acceptable in {country}? Examine your solution and that of other agents. "
    "Think step by step, then state your final answer as a single word - Yes, "
    "No or Neither - at the end of your response.\n\n"
    "Story: {story}\n"
    "Answer (Yes, No or Neither):"
)

# ---------- CulturalBench MCQ (option 1/2/3/4) ----------

SYSTEM_PROMPT_MCQ = (
    "You are a helpful assistant with expertise in cross-cultural "
    "knowledge and practices."
)

PROMPT_MCQ_START = (
    "You will be given a cultural knowledge question about {country}. Answer "
    "it as accurately as possible. Think step by step about the cultural "
    "practices of {country}, explain your reasoning, and put your final "
    "answer as a single option number (1, 2, 3, or 4) at the end of your "
    "response.\n\n"
    "Question:\n{story}\n"
    "Answer (1, 2, 3, or 4):"
)

PROMPT_MCQ_DEBATE = (
    "These are the recent/updated responses to the question from other "
    "agents:\n{other_responses}\n\n"
    "Using the reasoning from other agents as additional advice, can you give "
    "an updated answer to the cultural knowledge question about {country}? "
    "Examine your solution and that of other agents. Think step by step, then "
    "put your final answer as a single option number (1, 2, 3, or 4) at the "
    "end of your response.\n\n"
    "Question:\n{story}\n"
    "Answer (1, 2, 3, or 4):"
)

# ---------- BLEND MCQ (option 1/2/3/4, factual cultural knowledge) ----------

SYSTEM_PROMPT_BLEND = (
    "You are a helpful assistant with extensive knowledge of cultures, "
    "traditions, and daily life practices around the world."
)

PROMPT_BLEND_START = (
    "You will be given a factual cultural knowledge question about {country}. "
    "Read the question and all options carefully and select the most widely "
    "known correct answer specific to {country}. Explain your reasoning, then "
    "put your final answer as a single option number (1, 2, 3, or 4) at the "
    "end of your response.\n\n"
    "Question:\n{story}\n"
    "Answer (1, 2, 3, or 4):"
)

PROMPT_BLEND_DEBATE = (
    "These are the recent/updated responses to the question from other "
    "agents:\n{other_responses}\n\n"
    "Using the reasoning from other agents as additional advice, can you give "
    "an updated answer to the factual cultural knowledge question about "
    "{country}? Examine your solution and that of other agents, and decide "
    "which option is factually correct for {country}. Put your final answer "
    "as a single option number (1, 2, 3, or 4) at the end of your response.\n\n"
    "Question:\n{story}\n"
    "Answer (1, 2, 3, or 4):"
)

# ---------- CultureLLM (World Values Survey, variable options) ----------

SYSTEM_PROMPT_CULTURELLM = (
    "You are a helpful assistant with deep expertise in cross-cultural "
    "values, social attitudes, and the World Values Survey."
)

PROMPT_CULTURELLM_START = (
    "You will be given a question from the World Values Survey. Answer it by "
    "selecting the option number that best represents the prevailing cultural "
    "perspective and common attitudes in {country} (not your own values). "
    "Think step by step about the cultural values of {country}, explain your "
    "reasoning, then state your final answer as a single option number at the "
    "end of your response.\n\n"
    "Question:\n{story}\n"
    "Answer ({option_range}):"
)

PROMPT_CULTURELLM_DEBATE = (
    "These are the recent/updated responses to the question from other "
    "agents:\n{other_responses}\n\n"
    "Using the reasoning from other agents as additional advice, can you give "
    "an updated answer about what the prevailing cultural perspective in "
    "{country} would be? Examine your solution and that of other agents. "
    "State your final answer as a single option number at the end of your "
    "response.\n\n"
    "Question:\n{story}\n"
    "Answer ({option_range}):"
)


# Per-dataset template/system-prompt bundle
TEMPLATES = {
    DATASET_NORMAD: {
        "start": PROMPT_NORMAD_START, "debate": PROMPT_NORMAD_DEBATE,
        "system": SYSTEM_PROMPT_NORMAD,
    },
    DATASET_MCQ: {
        "start": PROMPT_MCQ_START, "debate": PROMPT_MCQ_DEBATE,
        "system": SYSTEM_PROMPT_MCQ,
    },
    DATASET_BLEND: {
        "start": PROMPT_BLEND_START, "debate": PROMPT_BLEND_DEBATE,
        "system": SYSTEM_PROMPT_BLEND,
    },
    DATASET_CULTURELLM: {
        "start": PROMPT_CULTURELLM_START, "debate": PROMPT_CULTURELLM_DEBATE,
        "system": SYSTEM_PROMPT_CULTURELLM,
    },
}


# ===================================================================
# Helpers
# ===================================================================

def apply_chat(tokenizer, user_content: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_other_responses(responses, self_idx):
    """Concatenate the OTHER agents' most-recent responses as advice context."""
    parts = []
    n = 1
    for j, resp in enumerate(responses):
        if j == self_idx:
            continue
        parts.append(f"Agent {n} response:\n{resp}")
        n += 1
    return "\n\n".join(parts)


# ===================================================================
# Main inference
# ===================================================================

def run_md(args):
    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {model_path}")

    out_json, out_metrics = infer_output_path(
        args.input_file, args.model_name, args.output_dir
    )
    print(f"Output JSON:    {out_json}")
    print(f"Output Metrics: {out_metrics}")

    # --- Load data ---
    dataset = load_dataset(args.input_file)
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
    print(f"Loaded {len(dataset)} samples from {args.input_file}")

    # --- Detect dataset type ---
    ds_type = detect_dataset_type(args.input_file)
    is_mcq = (ds_type in (DATASET_MCQ, DATASET_BLEND, DATASET_CULTURELLM))
    ds_label = {
        DATASET_NORMAD: "Yes/No/Neither (NorMAD)",
        DATASET_MCQ: "MCQ 4-choice (CulturalBench)",
        DATASET_BLEND: "MCQ 4-choice (BLEND)",
        DATASET_CULTURELLM: "World Values Survey (CultureLLM)",
    }
    print(f"Dataset type: {ds_type} ({ds_label.get(ds_type, ds_type)})")
    print(f"Agents: {args.num_agents}, Debate rounds: {args.num_rounds}")

    tpl = TEMPLATES[ds_type]
    tpl_start, tpl_debate, sys_prompt = tpl["start"], tpl["debate"], tpl["system"]

    # --- Pre-parse country / scenario / option range ---
    parsed = []
    for item in dataset:
        if is_mcq:
            country = item.get("country", "")
            story = item["input"]
        else:
            country, scenario, cultural_context = parse_input(item["input"])
            if cultural_context:
                story = (f"Cultural Background:\n{cultural_context}\n\n"
                         f"Scenario: {scenario}")
            else:
                story = scenario

        if ds_type == DATASET_CULTURELLM:
            min_opt, max_opt = get_culturellm_option_range(item["input"])
            option_range = (f"{min_opt} to {max_opt}" if max_opt <= 5
                            else f"{min_opt}, 2, 3, ... {max_opt}")
        else:
            min_opt, max_opt, option_range = None, None, None

        parsed.append({
            **item,
            "country": country,
            "scenario": story,
            "_max_opt": max_opt,
            "_option_range": option_range,
        })

    n = len(parsed)

    # --- Initialize vLLM ---
    print("Initializing vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        # Many agents/samples share the same system-prompt + question prefix;
        # prefix caching reuses their KV cache across rounds for a big speedup.
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "</s>"]
    # All agents are copies of the same model with the same temperature; diversity
    # arises from sampling randomness (as in the original MD paper).
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
        top_p=0.95,
    )

    def _generate_all(prompts):
        """Submit ALL prompts at once and let vLLM do continuous batching.

        vLLM internally schedules thousands of concurrent sequences far more
        efficiently than feeding it tiny external mini-batches, so we pass the
        whole list in a single call (use_tqdm shows real-time progress).
        """
        outs = llm.generate(prompts, sampling, use_tqdm=True)
        return [o.outputs[0].text.strip() for o in outs]

    def _extract(text, item):
        if ds_type == DATASET_CULTURELLM:
            return extract_answer_culturellm(text, item["_max_opt"])
        if is_mcq:
            return extract_answer_mcq(text)
        return extract_answer(text)

    # agent_responses[a] = list (length n) of latest response text for agent a
    # agent_answers[a]   = list (length n) of latest extracted answer for agent a
    A = args.num_agents
    agent_responses = [["" for _ in range(n)] for _ in range(A)]
    agent_answers = [["" for _ in range(n)] for _ in range(A)]
    # debate_history[i] = per-sample record of every round's responses/answers
    debate_history = [{"rounds": []} for _ in range(n)]

    def _start_kwargs(p):
        kw = {"country": p["country"], "story": p["scenario"]}
        if ds_type == DATASET_CULTURELLM:
            kw["option_range"] = p["_option_range"]
        return kw

    # -------- Round 0: independent generation (Starting prompt) --------
    # All agents share the SAME starting prompt per sample, so we build the
    # per-sample prompt once and replicate it A times into one big batch.
    print(f"\n=== Round 0: Independent Generation ({A} agents x {n} samples) ===")
    start_prompts = [apply_chat(tokenizer, tpl_start.format(**_start_kwargs(p)), sys_prompt)
                     for p in parsed]
    # Flat layout: index = a * n + i  (agent a, sample i)
    flat_prompts = [start_prompts[i] for _ in range(A) for i in range(n)]
    flat_texts = _generate_all(flat_prompts)
    for a in range(A):
        for i in range(n):
            txt = flat_texts[a * n + i]
            agent_responses[a][i] = txt
            agent_answers[a][i] = _extract(txt, parsed[i])

    for i in range(n):
        debate_history[i]["rounds"].append({
            "round": 0,
            "responses": [agent_responses[a][i] for a in range(A)],
            "answers": [agent_answers[a][i] for a in range(A)],
        })

    # -------- Rounds 1..R: debate (Debate prompt) --------
    for r in range(1, args.num_rounds + 1):
        print(f"\n=== Round {r}: Debate ===")
        # Snapshot previous-round responses (all agents update from same snapshot)
        prev_responses = [list(agent_responses[a]) for a in range(A)]

        new_responses = [["" for _ in range(n)] for _ in range(A)]
        new_answers = [["" for _ in range(n)] for _ in range(A)]

        # Build every agent's debate prompt for every sample, then submit the
        # whole (A * n) batch in one call. Flat layout: index = a * n + i.
        flat_prompts = []
        for a in range(A):
            for i, p in enumerate(parsed):
                other = build_other_responses(
                    [prev_responses[aa][i] for aa in range(A)], a
                )
                kw = {"country": p["country"], "story": p["scenario"],
                      "other_responses": other}
                if ds_type == DATASET_CULTURELLM:
                    kw["option_range"] = p["_option_range"]
                flat_prompts.append(apply_chat(tokenizer, tpl_debate.format(**kw), sys_prompt))

        flat_texts = _generate_all(flat_prompts)
        for a in range(A):
            for i in range(n):
                txt = flat_texts[a * n + i]
                new_responses[a][i] = txt
                new_answers[a][i] = _extract(txt, parsed[i])

        agent_responses = new_responses
        agent_answers = new_answers

        for i in range(n):
            debate_history[i]["rounds"].append({
                "round": r,
                "responses": [agent_responses[a][i] for a in range(A)],
                "answers": [agent_answers[a][i] for a in range(A)],
            })

    # -------- Final decision: majority vote over last-round answers --------
    print("\n=== Final Decision: Majority Vote ===")
    results = []
    for i, p in enumerate(parsed):
        last_answers = [agent_answers[a][i] for a in range(A)]
        final_ans = majority_vote(last_answers)
        gt = str(p.get("output", "")).strip()
        is_correct = (final_ans == gt) if final_ans else False

        record = {
            "instruction": p.get("instruction", ""),
            "input": p.get("input", ""),
            "output": gt,
            "country": p["country"],
            "scenario": p["scenario"],
            "debate_rounds": debate_history[i]["rounds"],
            "final_answers": last_answers,
            "final_answer": final_ans,
            "correct": is_correct,
        }
        results.append(record)

    # --- Write outputs ---
    print("\n=== Writing output ===")
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Inference results saved to: {out_json}")

    metrics = compute_metrics(results)
    metrics["method"] = "MD"
    metrics["model"] = args.model_name
    metrics["num_agents"] = args.num_agents
    metrics["num_rounds"] = args.num_rounds
    metrics["temperature"] = args.temperature
    metrics["prompt_source"] = "Du et al. 2023, Figure 15 (Starting + Debate)"

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
        description="MD Baseline: Multiagent Debate (Du et al., 2023)"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to dataset JSON (normad/culturalBench/blend/cultureLLM)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama/qwen) or HF path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: /autodl-fs/data/md)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Deprecated/unused: vLLM does continuous batching "
                             "over the full prompt list internally")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM max context length (prompt + generation)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples (0=all)")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Number of debating agents N (paper default: 3)")
    parser.add_argument("--num_rounds", type=int, default=2,
                        help="Number of debate rounds R after initial generation (paper default: 2)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (shared by all agents; diversity from sampling)")
    parser.add_argument("--max_tokens", type=int, default=512)

    args = parser.parse_args()
    run_md(args)


if __name__ == "__main__":
    main()
