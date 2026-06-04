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
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

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

def build_agent_prompt(agent_config: dict, question: str, country: str) -> str:
    """
    Build prompt for a RECONCILE agent (symmetric, no Guardian/Auditor distinction).
    """
    system_prompt = agent_config['system_prompt'].strip()
    
    user_content = (
        f"Target culture: {country}\n\n"
        f"Question:\n{question}\n\n"
        f"Provide your reasoning, then give your final answer as a single number."
    )
    
    return system_prompt, user_content


def build_judge_prompt(judge_config: dict, question: str, country: str,
                       agent_responses: List[str]) -> str:
    """
    Build prompt for the RECONCILE judge (neutral fact-checker).
    """
    system_prompt = judge_config['system_prompt'].strip()
    
    # Format agent responses
    responses_text = ""
    for i, resp in enumerate(agent_responses, 1):
        responses_text += f"\n--- Agent {i} ---\n{resp}\n"
    
    user_content = (
        f"Target culture: {country}\n\n"
        f"Question:\n{question}\n\n"
        f"Agent responses:{responses_text}\n"
        f"Based on the agents' perspectives and your knowledge of cultural facts, "
        f"determine the correct answer. Provide brief reasoning, then give your "
        f"final answer as a single number."
    )
    
    return system_prompt, user_content


# ---------------------------------------------------------------------------
# Inference (vLLM or HuggingFace)
# ---------------------------------------------------------------------------

def generate_with_vllm(prompts: List[dict], model_path: str,
                       temperature: float, max_tokens: int,
                       tensor_parallel_size: int = 1) -> List[str]:
    """Generate responses using vLLM for batch inference."""
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
    )
    
    sampling_params = SamplingParams(
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
    
    outputs = llm.chat(formatted_prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


def generate_with_hf(prompts: List[dict], model_path: str,
                     temperature: float, max_tokens: int) -> List[str]:
    """Generate responses using HuggingFace transformers."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='auto',
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    for p in tqdm(prompts, desc="Generating"):
        messages = [
            {"role": "system", "content": p['system']},
            {"role": "user", "content": p['user']},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=(temperature > 0),
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        results.append(response)
    
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_reconcile_inference(sample: dict, config: dict, model_path: str,
                           temperature: float, max_tokens: int,
                           use_vllm: bool, tensor_parallel_size: int) -> Optional[str]:
    """
    Run RECONCILE inference for a single sample.
    Returns formatted response string.
    """
    question = sample.get('input', '')
    country = sample.get('country', 'unknown')
    
    agents = config['culture_roles']
    judge_config = config['judge']
    
    # Phase 1: All agents generate independently
    agent_prompts = []
    for agent in agents:
        sys_prompt, user_content = build_agent_prompt(agent, question, country)
        agent_prompts.append({'system': sys_prompt, 'user': user_content})
    
    # Generate agent responses (batch)
    if use_vllm:
        agent_responses = generate_with_vllm(
            agent_prompts, model_path, temperature, max_tokens, tensor_parallel_size
        )
    else:
        agent_responses = generate_with_hf(
            agent_prompts, model_path, temperature, max_tokens
        )
    
    # Phase 2: Judge synthesizes
    judge_sys, judge_user = build_judge_prompt(
        judge_config, question, country, agent_responses
    )
    judge_prompt = [{'system': judge_sys, 'user': judge_user}]
    
    if use_vllm:
        judge_responses = generate_with_vllm(
            judge_prompt, model_path, 0.3, max_tokens, tensor_parallel_size
        )
    else:
        judge_responses = generate_with_hf(
            judge_prompt, model_path, 0.3, max_tokens
        )
    judge_response = judge_responses[0]
    
    # Format output (RECONCILE format: no role tags)
    response_parts = []
    for i, resp in enumerate(agent_responses, 1):
        # Extract answer number from response
        answer = extract_answer_from_response(resp)
        response_parts.append(
            f"===== Solution {i} =====\n{answer}\n{resp}"
        )
    
    # Judge as last solution
    judge_answer = extract_answer_from_response(judge_response)
    response_parts.append(
        f"===== Solution {len(agents) + 1} =====\n{judge_answer}\n{judge_response}"
    )
    
    return "\n".join(response_parts)


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
                        help="Model name or alias")
    parser.add_argument('--use_vllm', action='store_true',
                        help="Use vLLM for batch inference")
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument('--temperature', type=float, default=0.9,
                        help="Agent generation temperature")
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help="Max tokens per generation")
    
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
    
    # Process each sample
    print(f"Running RECONCILE inference with model: {model_path}")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        for i, sample in enumerate(tqdm(data, desc="Processing")):
            question = sample.get('input', '')
            country = sample.get('country', 'unknown')
            gt = sample.get('output', '')
            dataset_type = detect_dataset_type(sample.get('instruction', ''))
            
            # Run inference
            response = run_reconcile_inference(
                sample, config, model_path,
                args.temperature, args.max_tokens,
                args.use_vllm, args.tensor_parallel_size
            )
            
            # Write output
            output_record = {
                'query': question,
                'gt': gt,
                'country': country,
                'task_type': dataset_type,
                'response': response,
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            fout.flush()
    
    print(f"\nDone! Output saved to: {args.output_file}")


if __name__ == '__main__':
    main()
