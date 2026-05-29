"""
OG-MAR Baseline: Ontology-Guided Multi-Agent Reasoning

Reproduction of "Toward Culturally Aligned LLMs through Ontology-Guided
Multi-Agent Reasoning" (Seo et al., 2026) adapted for the NormAD cultural
acceptability judgment task.

Framework:
  1. Context Retrieval: For each query, retrieve relevant ontology triples
     and construct demographically grounded cultural profiles.
  2. Persona Agent Simulation (K=5): Instantiate K value-persona agents,
     each conditioned on ontology context, value summaries, and demographics.
     Each agent outputs a culturally grounded answer with reasoning.
  3. Judgment Agent: Synthesize persona outputs via constrained
     meta-adjudication (evidence-first protocol with vote-as-tiebreak).

Prompts follow Appendix E (Tables 8, 9) from the original paper, with
minimal adaptation for NormAD format (country/scenario instead of
WVS demographic/question).

Output naming: {dataset}_OGMAR_{model}.json + _metrics.json

Usage:
    # Quick test (5 samples)
    python OG/og_mar.py \\
        --input_file /autodl-fs/data/normad_mas.json \\
        --model_name qwen \\
        --tensor_parallel_size 2 \\
        --max_samples 5

    # Full dataset
    python OG/og_mar.py \\
        --input_file /autodl-fs/data/normad_mas.json \\
        --model_name qwen \\
        --tensor_parallel_size 2 \\
        --max_samples 0 \\
        --temperature 0.0 \\
        --num_personas 5 \\
        --num_triples 5
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
from OG.og_common import (
    MODEL_ALIASES, ANSWER_MAP, REVERSE_ANSWER_MAP,
    PERSONA_AGENT_PROMPT, JUDGMENT_AGENT_PROMPT,
    load_dataset, extract_country, extract_background, extract_scenario,
    extract_answer, get_relevant_triples, generate_persona_demographics,
    generate_value_summary, format_vote_summary,
    infer_output_path, compute_metrics, NORMAD_AXIS_TO_DOMAINS,
)


# ===================================================================
# System Prompt
# ===================================================================

SYSTEM_PROMPT = (
    "You are a culturally informed reasoning agent. You must strictly follow "
    "the instructions and output format specified in the task. Do not add any "
    "text outside the required JSON output."
)


# ===================================================================
# Response Options (fixed for NormAD task)
# ===================================================================

OPTIONS_TEXT = (
    '1: acceptable (the behavior is socially acceptable in this culture)\n'
    '2: unacceptable (the behavior is socially unacceptable in this culture)\n'
    '3: neutral/indeterminate (cannot determine or culture-neutral)'
)


# ===================================================================
# Chat template helper
# ===================================================================

def apply_chat(tokenizer, user_content: str, system: str = None) -> str:
    """Apply chat template to format prompt for the model."""
    messages = [
        {"role": "system", "content": system or SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ===================================================================
# Infer axis from NormAD input (heuristic)
# ===================================================================

def infer_axis(input_text: str, item: dict) -> str:
    """Infer the cultural axis from the NormAD item."""
    # First check if there's an explicit axis in the item's metadata
    # (normad_mas.json may not have this, so we fall back to heuristics)

    # Check cultural background section for axis keywords
    bg = extract_background(input_text)
    bg_lower = bg.lower() if bg else ""

    # Map keywords to axes
    keyword_map = {
        "etiquette": "Etiquette",
        "greet": "Etiquette",
        "polite": "Etiquette",
        "rude": "Etiquette",
        "moral": "Morality",
        "ethical": "Morality",
        "law": "Law",
        "legal": "Law",
        "illegal": "Law",
        "crime": "Law",
        "religion": "Religion",
        "pray": "Religion",
        "mosque": "Religion",
        "church": "Religion",
        "temple": "Religion",
        "faith": "Religion",
        "family": "Family",
        "marriage": "Family",
        "parent": "Family",
        "work": "Work",
        "employ": "Work",
        "business": "Work",
        "food": "Food",
        "eat": "Food",
        "drink": "Food",
        "education": "Education",
        "school": "Education",
        "visit": "Etiquette",
    }

    # Score each axis
    axis_scores = Counter()
    for keyword, axis in keyword_map.items():
        if keyword in bg_lower:
            axis_scores[axis] += 1

    if axis_scores:
        return axis_scores.most_common(1)[0][0]
    return "default"


# ===================================================================
# Main inference function
# ===================================================================

def run_og_mar(args):
    # --- Resolve model ---
    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {model_path}")
    print(f"Number of persona agents (K): {args.num_personas}")
    print(f"Number of ontology triples (M): {args.num_triples}")

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

    # --- Pre-parse data ---
    parsed = []
    for item in dataset:
        input_text = item.get("input", "")
        country = extract_country(input_text)
        background = extract_background(input_text)
        scenario = extract_scenario(input_text)
        axis = infer_axis(input_text, item)

        parsed.append({
            **item,
            "country": country,
            "background": background,
            "scenario": scenario,
            "axis": axis,
        })

    n = len(parsed)
    K = args.num_personas  # Number of persona agents
    M = args.num_triples   # Number of ontology triples

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
        top_p=0.95 if args.temperature > 0 else 1.0,
    )

    batch_size = args.batch_size

    # ================================================================
    # Stage 1: Ontology & Demographic Retrieval (pre-computed)
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 1: Ontology & Context Retrieval")
    print(f"{'='*60}")

    # Pre-compute ontology triples and demographics for all samples
    all_triples = []       # [sample_idx] -> list of triple strings
    all_demographics = []  # [sample_idx][persona_idx] -> demographics text
    all_value_summaries = []  # [sample_idx][persona_idx] -> value summary

    for idx, p in enumerate(tqdm(parsed, desc="Retrieving context")):
        # Retrieve ontology triples
        triples = get_relevant_triples(p["country"], p["axis"], p["background"], top_n=M)
        all_triples.append(triples)

        # Generate demographics for K personas
        demographics = []
        value_summaries = []
        for k in range(K):
            demo = generate_persona_demographics(p["country"], k)
            demographics.append(demo)
            val_sum = generate_value_summary(
                p["country"], p["axis"], p["background"], k
            )
            value_summaries.append(val_sum)

        all_demographics.append(demographics)
        all_value_summaries.append(value_summaries)

    # ================================================================
    # Stage 2: Multi-Value Persona Agent Simulation
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Stage 2: Persona Agent Simulation (K={K})")
    print(f"{'='*60}")

    # persona_outputs[sample_idx][persona_idx] = raw output text
    persona_outputs = [[None] * K for _ in range(n)]

    for k in range(K):
        print(f"\n  Persona Agent {k+1}/{K}...")

        # Build prompts for this persona across all samples
        prompts = []
        for idx, p in enumerate(parsed):
            hyper_nodes_text = "\n".join(all_triples[idx])
            question = (
                f"In the cultural context of {p['country'].replace('_', ' ').title()}, "
                f"evaluate the following scenario:\n{p['scenario']}\n\n"
                f"Is the described behavior socially acceptable in this culture?"
            )

            user_content = PERSONA_AGENT_PROMPT.format(
                persona_id=k + 1,
                demographics_text=all_demographics[idx][k],
                value_summaries_text=all_value_summaries[idx][k],
                hyper_nodes_text=hyper_nodes_text,
                options_text=OPTIONS_TEXT,
                question=question,
            )
            prompts.append(apply_chat(tokenizer, user_content))

        # Batch inference
        all_outputs = []
        for i in tqdm(range(0, n, batch_size),
                      desc=f"  Persona-{k+1}", leave=False):
            batch_end = min(i + batch_size, n)
            batch_prompts = prompts[i:batch_end]
            outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
            for out in outputs:
                all_outputs.append(out.outputs[0].text.strip())

        # Store outputs
        for idx, resp in enumerate(all_outputs):
            persona_outputs[idx][k] = resp

    # ================================================================
    # Stage 3: Ontology-Guided Final Judgment via Constrained
    #           Meta-Adjudication
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 3: Judgment Agent - Constrained Meta-Adjudication")
    print(f"{'='*60}")

    # First, extract persona answers for vote summary
    persona_answers = [[None] * K for _ in range(n)]
    for idx in range(n):
        for k in range(K):
            persona_answers[idx][k] = extract_answer(persona_outputs[idx][k])

    # Build judgment prompts
    judgment_prompts = []
    for idx, p in enumerate(parsed):
        # Format persona outputs for the judge
        persona_text_parts = []
        for k in range(K):
            persona_text_parts.append(
                f"--- Persona Agent {k+1} ---\n{persona_outputs[idx][k]}"
            )
        persona_outputs_text = "\n\n".join(persona_text_parts)

        # Vote summary
        vote_summary = format_vote_summary(persona_answers[idx])

        # Question
        question_text = (
            f"In the cultural context of {p['country'].replace('_', ' ').title()}, "
            f"evaluate the following scenario:\n{p['scenario']}\n\n"
            f"Is the described behavior socially acceptable in this culture?"
        )

        user_content = JUDGMENT_AGENT_PROMPT.format(
            question_text=question_text,
            options_text=OPTIONS_TEXT,
            vote_summary=vote_summary,
            persona_outputs=persona_outputs_text,
        )
        judgment_prompts.append(apply_chat(tokenizer, user_content))

    # Batch inference for judgment
    judgment_outputs = []
    for i in tqdm(range(0, n, batch_size), desc="Judgment"):
        batch_end = min(i + batch_size, n)
        batch_prompts = judgment_prompts[i:batch_end]
        outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
        for out in outputs:
            judgment_outputs.append(out.outputs[0].text.strip())

    # ================================================================
    # Stage 4: Extract answers and build results
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 4: Extract Answers & Compute Metrics")
    print(f"{'='*60}")

    results = []
    for idx, p in enumerate(parsed):
        # Extract judgment answer
        judge_resp = judgment_outputs[idx]
        judge_ans = extract_answer(judge_resp)

        # If judgment extraction fails, fall back to majority vote
        if judge_ans is None:
            valid_votes = [a for a in persona_answers[idx] if a is not None]
            if valid_votes:
                vote_counter = Counter(valid_votes)
                judge_ans = vote_counter.most_common(1)[0][0]

        gt = str(p.get("output", "")).strip()
        is_correct = (judge_ans == gt) if judge_ans else False

        # Format per-persona answers
        per_persona = {}
        for k in range(K):
            per_persona[f"persona_{k+1}"] = {
                "response": persona_outputs[idx][k],
                "answer": persona_answers[idx][k],
            }

        record = {
            "instruction": p.get("instruction", ""),
            "input": p.get("input", ""),
            "output": gt,
            "country": p["country"],
            "scenario": p["scenario"],
            "axis": p["axis"],
            # Retrieval context
            "ontology_triples": all_triples[idx],
            # Persona agents
            "persona_outputs": per_persona,
            "persona_vote_summary": format_vote_summary(persona_answers[idx]),
            # Judgment
            "judgment_response": judge_resp,
            "final_answer": judge_ans if judge_ans else "",
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
    metrics["method"] = "OG-MAR"
    metrics["model"] = args.model_name
    metrics["num_personas"] = K
    metrics["num_triples"] = M
    metrics["temperature"] = args.temperature
    metrics["prompt_source"] = "Appendix E, Tables 8-9 (OG-MAR paper, Seo et al. 2026)"
    metrics["framework"] = "Ontology-Guided Multi-Agent Reasoning"

    # Add persona agreement stats
    full_agreement = 0
    for idx in range(n):
        valid_votes = [a for a in persona_answers[idx] if a is not None]
        if valid_votes and len(set(valid_votes)) == 1:
            full_agreement += 1
    metrics["persona_full_agreement"] = full_agreement

    metrics_dir = os.path.dirname(out_metrics)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {out_metrics}")
    print(f"\nAccuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total_samples']})")
    print(f"Persona full agreement: {full_agreement}/{n}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OG-MAR Baseline: Ontology-Guided Multi-Agent Reasoning (Seo et al., 2026)"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to normad_mas.json")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama/qwen) or HuggingFace path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: /autodl-fs/data/ogmar)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for vLLM inference")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (paper uses 0 for stable behavior)")
    parser.add_argument("--max_tokens", type=int, default=768,
                        help="Max tokens per generation (longer for structured JSON output)")
    parser.add_argument("--num_personas", type=int, default=5,
                        help="Number of persona agents K (paper default: 5)")
    parser.add_argument("--num_triples", type=int, default=5,
                        help="Number of ontology triples M to retrieve (paper default: 3-9)")

    args = parser.parse_args()
    run_og_mar(args)


if __name__ == "__main__":
    main()
