"""
Generate RECONCILE multi-agent inference data for cultural alignment tasks.

This script generates symmetric multi-agent discussion data using the RECONCILE
framework (5 equal-authority cultural agents + 1 judge), serving as the data
source for the "MAGDi + RECONCILE" baseline.

Unlike HF-CAC which has asymmetric Guardian/Auditor roles, RECONCILE treats
all agents equally — no agent has special authority regardless of the target
country. This is the key difference that CAMAD's HF-CAC mechanism addresses.

Output format: JSONL with the same schema as HF-CAC output, but without
role tags ([GUARDIAN]/[AUDITOR]) in solution headers.

Usage:
    # Generate RECONCILE data for CultureBench
    python generate_reconcile_data.py \
        --input_file /autodl-fs/data/culturalBench_mas.json \
        --output_file /autodl-fs/data/qwen/culturalbench_reconcile_inference.jsonl \
        --config_file ../Cul/configs/reconcile_config.yaml \
        --model_name qwen \
        --use_vllm --tensor_parallel_size 2

    # Generate RECONCILE data for NormAD
    python generate_reconcile_data.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/qwen/normad_reconcile_inference.jsonl \
        --config_file ../Cul/configs/reconcile_config.yaml \
        --model_name qwen \
        --use_vllm --tensor_parallel_size 2
"""

import os
import sys
import re
import json
import yaml
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

# Add parent directory for shared utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen": "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}


def load_config(config_path: str) -> dict:
    """Load RECONCILE config YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def detect_dataset_type(instruction: str) -> str:
    """Auto-detect dataset type from instruction text."""
    if "cultural knowledge question" in instruction.lower():
        return "culturalbench"
    elif "more culturally specific" in instruction.lower():
        return "cultureatlas"
    else:
        return "normad"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_agent_prompt(agent_config: dict, question: str, country: str) -> dict:
    """
    Build prompt for a RECONCILE agent (symmetric, no Guardian/Auditor distinction).
    Returns dict with 'system' and 'user' keys.
    """
    system_prompt = agent_config['system_prompt'].strip()

    user_content = (
        f"Target culture: {country}\n\n"
        f"Question:\n{question}\n\n"
        f"Provide your reasoning briefly, then give your final answer as a single number."
    )

    return {'system': system_prompt, 'user': user_content}


def build_judge_prompt(judge_config: dict, question: str, country: str,
                       agent_responses: List[str]) -> dict:
    """
    Build prompt for the RECONCILE judge (neutral fact-checker).
    Returns dict with 'system' and 'user' keys.
    """
    system_prompt = judge_config['system_prompt'].strip()

    # Format agent responses (truncate overly long ones to save context)
    responses_text = ""
    for i, resp in enumerate(agent_responses, 1):
        # Truncate individual agent response if too long
        truncated = resp[:800] if len(resp) > 800 else resp
        responses_text += f"\n--- Agent {i} ---\n{truncated}\n"

    user_content = (
        f"Target culture: {country}\n\n"
        f"Question:\n{question}\n\n"
        f"Agent responses:{responses_text}\n"
        f"Determine the correct answer. Give brief reasoning, then your "
        f"final answer as a single number."
    )

    return {'system': system_prompt, 'user': user_content}


# ---------------------------------------------------------------------------
# vLLM Engine wrapper (initialized once, reused across all samples)
# ---------------------------------------------------------------------------

class VLLMEngine:
    """Wrapper around vLLM LLM to initialize once and reuse."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams

        print(f"Initializing vLLM engine: {model_path} (tp={tensor_parallel_size})")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=2048,            # Reduced: prompts ~400 tok + output ~512 tok
            gpu_memory_utilization=0.92,   # Use more GPU memory for larger batches
            swap_space=4,                  # 4GB swap for overflow
            enable_prefix_caching=True,    # Cache shared system prompt prefixes
        )
        print("vLLM engine ready.")

    def generate(self, prompts: List[dict], temperature: float,
                 max_tokens: int) -> List[str]:
        """
        Generate responses for a batch of prompts.
        Each prompt is a dict with 'system' and 'user' keys.
        """
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        # Format as chat messages
        formatted_prompts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": p['system']},
                {"role": "user", "content": p['user']},
            ]
            formatted_prompts.append(messages)

        outputs = self.llm.chat(formatted_prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# HuggingFace fallback (also initialized once)
# ---------------------------------------------------------------------------

class HFEngine:
    """Wrapper around HuggingFace transformers for sequential inference."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Initializing HF engine: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto',
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.torch = torch
        print("HF engine ready.")

    def generate(self, prompts: List[dict], temperature: float,
                 max_tokens: int) -> List[str]:
        """Generate responses sequentially."""
        results = []
        for p in tqdm(prompts, desc="HF generating", leave=False):
            messages = [
                {"role": "system", "content": p['system']},
                {"role": "user", "content": p['user']},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=(temperature > 0),
                    top_p=0.9 if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            results.append(response)

        return results


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer_from_response(text: str) -> str:
    """Extract the answer number from a response."""
    # Look for explicit answer patterns
    m = re.search(r"[Aa]nswer\s*(?:is|:)\s*([1-4])", text)
    if m:
        return m.group(1)
    # Look for last standalone digit
    digits = re.findall(r"\b([1-4])\b", text)
    if digits:
        return digits[-1]
    return "1"  # fallback


# ---------------------------------------------------------------------------
# Two-phase batch pipeline
# ---------------------------------------------------------------------------

def run_batch_reconcile(data: List[dict], config: dict, engine,
                        temperature: float, max_tokens: int) -> List[str]:
    """
    Run RECONCILE inference in two batch phases:
      Phase 1: All agents for all samples (batch)
      Phase 2: Judge for all samples (batch, using Phase 1 results)

    This avoids reinitializing the engine per sample and maximizes GPU utilization.
    """
    agents = config['culture_roles']
    judge_config = config['judge']
    num_agents = len(agents)
    num_samples = len(data)

    # ---- Phase 1: Build all agent prompts across all samples ----
    total_agent_prompts = num_agents * num_samples
    print(f"\n[Phase 1] Generating agent responses "
          f"({num_agents} agents × {num_samples} samples = {total_agent_prompts} prompts)")
    all_agent_prompts = []
    for sample in data:
        question = sample.get('input', '')
        country = sample.get('country', 'unknown')
        for agent in agents:
            prompt = build_agent_prompt(agent, question, country)
            all_agent_prompts.append(prompt)

    # Batch generate all agent responses at once
    # Agent responses: short reasoning + answer, 512 tokens is sufficient
    t0 = time.time()
    all_agent_responses = engine.generate(all_agent_prompts, temperature, min(max_tokens, 512))
    t1 = time.time()
    print(f"  Phase 1 done in {t1 - t0:.1f}s "
          f"({total_agent_prompts / (t1 - t0):.1f} prompts/s)")

    # Reshape: [num_samples, num_agents]
    agent_responses_per_sample = []
    for i in range(num_samples):
        start = i * num_agents
        end = start + num_agents
        agent_responses_per_sample.append(all_agent_responses[start:end])

    # ---- Phase 2: Build all judge prompts ----
    print(f"\n[Phase 2] Generating judge responses ({num_samples} prompts)")
    all_judge_prompts = []
    for i, sample in enumerate(data):
        question = sample.get('input', '')
        country = sample.get('country', 'unknown')
        prompt = build_judge_prompt(
            judge_config, question, country, agent_responses_per_sample[i]
        )
        all_judge_prompts.append(prompt)

    # Judge responses: brief synthesis + answer, 256 tokens is sufficient
    t2 = time.time()
    all_judge_responses = engine.generate(all_judge_prompts, 0.3, min(max_tokens, 256))
    t3 = time.time()
    print(f"  Phase 2 done in {t3 - t2:.1f}s "
          f"({num_samples / (t3 - t2):.1f} prompts/s)")

    # ---- Assemble final outputs ----
    print(f"\n[Summary] Total inference time: {t3 - t0:.1f}s")
    results = []
    for i in range(num_samples):
        response_parts = []
        for j, resp in enumerate(agent_responses_per_sample[i], 1):
            answer = extract_answer_from_response(resp)
            response_parts.append(
                f"===== Solution {j} =====\n{answer}\n{resp}"
            )
        # Judge as last solution
        judge_answer = extract_answer_from_response(all_judge_responses[i])
        response_parts.append(
            f"===== Solution {num_agents + 1} =====\n{judge_answer}\n{all_judge_responses[i]}"
        )
        results.append("\n".join(response_parts))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate RECONCILE inference data for cultural alignment"
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help="Input JSON file (MAS format: instruction/input/output/country)")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output JSONL file")
    parser.add_argument('--config_file', type=str,
                        default='../Cul/configs/reconcile_config.yaml',
                        help="RECONCILE config YAML")
    parser.add_argument('--model_name', type=str, default='qwen',
                        help="Model name or alias (llama / qwen)")
    parser.add_argument('--use_vllm', action='store_true',
                        help="Use vLLM for batch inference (recommended)")
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument('--temperature', type=float, default=0.9,
                        help="Agent generation temperature")
    parser.add_argument('--max_tokens', type=int, default=512,
                        help="Max tokens per agent generation (default 512)")

    args = parser.parse_args()

    # Resolve model path
    model_path = MODEL_ALIASES.get(args.model_name, args.model_name)

    # Load config
    print(f"Loading config: {args.config_file}")
    config = load_config(args.config_file)

    # Load data
    print(f"Loading data: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.max_samples > 0:
        data = data[:args.max_samples]
    print(f"  Samples to process: {len(data)}")
    print(f"  Model: {model_path}")
    print(f"  Agent max_tokens: {args.max_tokens}")
    print(f"  Judge max_tokens: {min(args.max_tokens, 256)}")

    # Initialize engine ONCE
    t_start = time.time()
    if args.use_vllm:
        engine = VLLMEngine(model_path, args.tensor_parallel_size)
    else:
        engine = HFEngine(model_path)
    t_init = time.time()
    print(f"  Engine init time: {t_init - t_start:.1f}s")

    # Run batch inference (two-phase)
    results = run_batch_reconcile(
        data, config, engine, args.temperature, args.max_tokens
    )

    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting output to: {args.output_file}")
    with open(output_path, 'w', encoding='utf-8') as fout:
        for i, sample in enumerate(data):
            question = sample.get('input', '')
            country = sample.get('country', 'unknown')
            gt = sample.get('output', '')
            dataset_type = detect_dataset_type(sample.get('instruction', ''))

            output_record = {
                'query': question,
                'gt': gt,
                'country': country,
                'task_type': dataset_type,
                'response': results[i],
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')

    t_end = time.time()
    print(f"\nDone! {len(data)} samples processed.")
    print(f"Total wall time: {t_end - t_start:.1f}s ({(t_end - t_start) / 60:.1f} min)")


if __name__ == '__main__':
    main()
