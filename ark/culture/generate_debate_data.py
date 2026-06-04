"""
AgentArk Baseline — Stage 0: Multi-Agent Debate Data Generation

Uses homogeneous LLM Debate (same model, same role) to generate
multi-agent reasoning data for cultural alignment tasks.

Key differences from HF-CAC:
  - All agents are identical (no cultural specialization)
  - Symmetric debate (no Guardian/Auditor hierarchy)
  - Standard majority-vote aggregation (no authority weighting)

Supports:
  - NormAD: behavior acceptability judgment (1/2/3)
  - CulturalBench: multiple-choice cultural knowledge QA (1/2/3/4)

Output format: JSONL with ===== Solution N ===== markers (same as HF-CAC)
for downstream pipeline compatibility.

Usage:
    python ark/culture/generate_debate_data.py \
        --input_file /autodl-fs/data/normad_mas.json \
        --output_file /autodl-fs/data/qwen/normad_agentark_debate.jsonl \
        --model_name qwen \
        --num_agents 5 \
        --num_rounds 2 \
        --use_vllm --tensor_parallel_size 2 \
        --max_samples 0

    python ark/culture/generate_debate_data.py \
        --input_file /autodl-fs/data/culturalBench_mas.json \
        --output_file /autodl-fs/data/qwen/culturalbench_agentark_debate.jsonl \
        --model_name qwen \
        --num_agents 5 \
        --num_rounds 2 \
        --use_vllm --tensor_parallel_size 2
"""

import os
import sys
import re
import json
import argparse
import threading
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import write_to_jsonl, reserve_unprocessed_queries


# ---------------------------------------------------------------------------
# Data helpers (shared with generate_hf_cac_data.py)
# ---------------------------------------------------------------------------

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())[0]
    return data


def detect_dataset_type(data: list) -> str:
    """Auto-detect dataset type from data content."""
    if not data:
        return "normad"
    sample = data[:min(10, len(data))]
    for item in sample:
        instruction = item.get("instruction", "").lower()
        if "cultural knowledge question" in instruction:
            return "culturalbench"
        if "correct option number" in instruction:
            return "culturalbench"
        if "acceptable" in instruction or "unacceptable" in instruction:
            return "normad"
        if "determine whether the behavior" in instruction:
            return "normad"
    for item in sample:
        inp = item.get("input", "")
        if "\n1. " in inp and "\n2. " in inp and "\n3. " in inp and "\n4. " in inp:
            return "culturalbench"
    outputs = set(str(item.get("output", "")).strip() for item in data[:100])
    if "4" in outputs:
        return "culturalbench"
    if "3" in outputs and "4" not in outputs:
        return "normad"
    return "normad"


def convert_sample(item):
    """Convert dataset sample to internal format."""
    if "input" in item and item["input"] and "country" in item:
        query = item["input"].strip()
        country = item["country"].strip()
        gt = str(item.get("output", "")).strip()
        return {"query": query, "gt": gt, "country": country}
    instruction = item["instruction"]
    query = instruction.split("### Answer:")[0].strip()
    if "Country" in item and item["Country"]:
        country = item["Country"].strip()
    else:
        m = re.search(r"country or language that is (.+?)\.", instruction)
        country = m.group(1).strip() if m else ""
    gt = str(item.get("output", item.get("label", ""))).strip()
    return {"query": query, "gt": gt, "country": country}


# ---------------------------------------------------------------------------
# AgentArk LLM Debate (Homogeneous Agents)
# ---------------------------------------------------------------------------

class AgentArkDebate:
    """
    Homogeneous multi-agent debate for cultural alignment tasks.

    All agents use the same model and same generic system prompt.
    This is the AgentArk baseline — no cultural specialization.
    """

    def __init__(self, model_name: str, num_agents: int = 5,
                 num_rounds: int = 2, temperature: float = 0.7,
                 max_tokens: int = 512, tensor_parallel_size: int = 1,
                 dataset_type: str = "normad"):
        self.model_name = model_name
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dataset_type = dataset_type
        self.tensor_parallel_size = tensor_parallel_size

        # Determine valid choices based on dataset
        if dataset_type == "culturalbench":
            self.max_choice = 4
        elif dataset_type == "normad":
            self.max_choice = 3
        else:
            self.max_choice = 3

        # System prompt: generic cultural reasoning (no cultural identity)
        self.agent_system_prompt = self._build_system_prompt()
        self.judge_system_prompt = self._build_judge_prompt()

        # vLLM engine (lazy init)
        self._engine = None
        self._tokenizer = None

    def _build_system_prompt(self) -> str:
        if self.dataset_type == "culturalbench":
            return (
                "You are a knowledgeable assistant. Read the cultural knowledge "
                "question carefully and select the correct option number (1-4). "
                "Provide your reasoning first, then your final answer.\n"
                "Format: Reasoning: <your analysis>\nAnswer: <number>"
            )
        else:  # normad
            return (
                "You are a knowledgeable cultural expert. Read the scenario and "
                "determine whether the described behavior is socially acceptable "
                "in the specified country's cultural context.\n"
                "Answer with:\n"
                "  1 = acceptable\n"
                "  2 = unacceptable\n"
                "  3 = neutral/indeterminate\n"
                "Provide your reasoning first, then your final answer.\n"
                "Format: Reasoning: <your analysis>\nAnswer: <number>"
            )

    def _build_judge_prompt(self) -> str:
        return (
            "You are a neutral fact-checker. You will see a question and "
            "responses from multiple agents. Determine the correct answer by "
            "carefully weighing the agents' reasoning and factual evidence.\n"
            "Do NOT simply choose the majority — use critical judgment.\n"
            f"Answer with a single number (1-{self.max_choice}).\n"
            "Format: Reasoning: <your analysis>\nAnswer: <number>"
        )

    def _init_engine(self):
        """Lazy-initialize vLLM engine."""
        if self._engine is not None:
            return
        from vllm import LLM
        from transformers import AutoTokenizer
        self._engine = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _format_prompt(self, messages: list[dict]) -> str:
        """Format messages using chat template."""
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def inference_batch(self, samples: list[dict]) -> list[dict]:
        """
        Batch inference: run multi-agent debate for all samples.

        Args:
            samples: list of {"query": ..., "gt": ..., "country": ...}

        Returns:
            list of {"response": formatted_debate_output}
        """
        from vllm import SamplingParams

        self._init_engine()
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"],
        )

        # Initialize agent contexts for all samples
        # Shape: all_contexts[sample_idx][agent_idx] = list of messages
        all_contexts = []
        for sample in samples:
            query = sample["query"]
            contexts = [
                [
                    {"role": "system", "content": self.agent_system_prompt},
                    {"role": "user", "content": query},
                ]
                for _ in range(self.num_agents)
            ]
            all_contexts.append(contexts)

        # Run debate rounds
        for round_idx in range(self.num_rounds):
            # Collect all prompts for this round
            all_prompts = []
            prompt_meta = []  # (sample_idx, agent_idx)

            for s_idx, contexts in enumerate(all_contexts):
                for a_idx, ctx in enumerate(contexts):
                    if round_idx > 0:
                        # Add other agents' responses as context
                        other_responses = []
                        for other_idx, other_ctx in enumerate(contexts):
                            if other_idx != a_idx and len(other_ctx) >= 3:
                                other_responses.append(
                                    other_ctx[-1]["content"]
                                )
                        debate_msg = self._construct_debate_message(
                            other_responses, samples[s_idx]["query"]
                        )
                        ctx.append({"role": "user", "content": debate_msg})

                    prompt = self._format_prompt(ctx)
                    all_prompts.append(prompt)
                    prompt_meta.append((s_idx, a_idx))

            # Batch generate
            outputs = self._engine.generate(all_prompts, sampling_params)

            # Update contexts
            for output, (s_idx, a_idx) in zip(outputs, prompt_meta):
                response = output.outputs[0].text.strip()
                all_contexts[s_idx][a_idx].append(
                    {"role": "assistant", "content": response}
                )

        # Aggregation: Judge collects all agent responses
        judge_prompts = []
        for s_idx, (sample, contexts) in enumerate(zip(samples, all_contexts)):
            agent_answers = [ctx[-1]["content"] for ctx in contexts]
            judge_msg = self._build_judge_message(
                sample["query"], agent_answers
            )
            judge_prompt = self._format_prompt([
                {"role": "system", "content": self.judge_system_prompt},
                {"role": "user", "content": judge_msg},
            ])
            judge_prompts.append(judge_prompt)

        judge_outputs = self._engine.generate(judge_prompts, sampling_params)

        # Format results (compatible with downstream pipeline)
        results = []
        for s_idx, (sample, contexts, judge_out) in enumerate(
            zip(samples, all_contexts, judge_outputs)
        ):
            formatted = ""
            for a_idx, ctx in enumerate(contexts):
                agent_response = ctx[-1]["content"]
                formatted += (
                    f"===== Solution {a_idx + 1} [AGENT-{a_idx + 1}] =====\n"
                    f"{agent_response}\n"
                )
            judge_response = judge_out.outputs[0].text.strip()
            formatted += (
                f"===== Solution {self.num_agents + 1} [JUDGE] =====\n"
                f"{judge_response}\n"
            )
            results.append({"response": formatted})

        return results

    def _construct_debate_message(self, other_responses: list[str],
                                  question: str) -> str:
        """Build debate context from other agents' responses."""
        if not other_responses:
            return (
                "Please verify your answer is correct. "
                "Reiterate your reasoning and provide your final answer."
            )
        msg = "Here are responses from other agents:\n\n"
        for i, resp in enumerate(other_responses):
            msg += f"--- Agent {i+1} ---\n{resp}\n\n"
        msg += (
            "Consider these perspectives carefully. If you find compelling "
            "evidence to change your answer, do so. Otherwise, defend your "
            "original position with stronger reasoning.\n"
            "Provide your updated answer.\n"
            f"Format: Reasoning: <your analysis>\nAnswer: <number>"
        )
        return msg

    def _build_judge_message(self, question: str,
                             agent_answers: list[str]) -> str:
        """Build judge aggregation prompt."""
        msg = f"Question:\n{question}\n\n"
        msg += "Agent responses:\n\n"
        for i, ans in enumerate(agent_answers):
            msg += f"--- Agent {i+1} ---\n{ans}\n\n"
        msg += (
            "Based on all agent responses, determine the correct answer. "
            "Weigh the evidence and reasoning quality, not just majority vote.\n"
            f"Format: Reasoning: <your analysis>\nAnswer: <number>"
        )
        return msg


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def extract_answer(text: str, max_choice: int = 3) -> str:
    """Extract answer number from response text."""
    pattern = f'[1-{max_choice}]'
    # Try "Answer: X" pattern first
    m = re.search(rf'Answer\s*:\s*({pattern})', text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: last valid digit
    digits = re.findall(rf'\b({pattern})\b', text)
    return digits[-1] if digits else None


def extract_judge_answer(response_text: str, max_choice: int = 3) -> str:
    """Extract Judge's answer from formatted debate output."""
    judge_match = re.search(
        r'===== Solution \d+ \[JUDGE\] =====\n(.*?)$',
        response_text, re.DOTALL
    )
    if not judge_match:
        return None
    return extract_answer(judge_match.group(1), max_choice)


def compute_accuracy(output_file: str, max_choice: int) -> dict:
    """Compute accuracy from inference output."""
    from collections import Counter

    data = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    judge_correct = 0
    judge_total = 0
    country_stats = {}

    for d in data:
        gt = d.get("gt", "").strip()
        if not gt:
            continue
        country = d.get("country", "unknown")
        response = d.get("response", "")

        judge_ans = extract_judge_answer(response, max_choice)
        if judge_ans:
            judge_total += 1
            if judge_ans == gt:
                judge_correct += 1

        if country not in country_stats:
            country_stats[country] = {"total": 0, "correct": 0}
        country_stats[country]["total"] += 1
        if judge_ans == gt:
            country_stats[country]["correct"] += 1

    return {
        "total_samples": len(data),
        "judge_total": judge_total,
        "judge_correct": judge_correct,
        "judge_accuracy": judge_correct / judge_total if judge_total > 0 else 0.0,
        "per_country": {
            c: {**s, "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0.0}
            for c, s in sorted(country_stats.items())
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AgentArk Baseline: Generate LLM Debate data for cultural tasks"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to dataset JSON (normad_mas.json / culturalBench_mas.json)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL path (auto-generated if not specified)")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["llama", "qwen"],
                        help="Base model to use: 'llama' (LLaMA-3.1-8B-Instruct) or 'qwen' (Qwen2.5-7B-Instruct)")
    parser.add_argument("--num_agents", type=int, default=5,
                        help="Number of debate agents (default: 5)")
    parser.add_argument("--num_rounds", type=int, default=2,
                        help="Number of debate rounds (default: 2)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = all samples")
    parser.add_argument("--eval_accuracy", action="store_true", default=True)
    args = parser.parse_args()

    # Model alias resolution (only llama and qwen are supported)
    MODEL_ALIASES = {
        "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
        "qwen": "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
    }
    args.model_name = MODEL_ALIASES[args.model_name]
    print(f"Model: {args.model_name}")

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_file is None:
        stem = Path(args.input_file).stem
        args.output_file = str(
            Path(args.input_file).parent / f"{stem}_agentark_debate_{timestamp}.jsonl"
        )
    else:
        p = Path(args.output_file)
        args.output_file = str(p.parent / f"{p.stem}_{timestamp}{p.suffix}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    print(f"Output: {args.output_file}")

    # Load data
    raw_data = load_dataset(args.input_file)
    dataset = [convert_sample(item) for item in raw_data]
    print(f"Loaded {len(dataset)} samples")

    # Detect dataset type
    dataset_type = detect_dataset_type(raw_data)
    print(f"Dataset type: {dataset_type}")
    max_choice = 4 if dataset_type == "culturalbench" else 3

    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
        print(f"Using first {args.max_samples} samples")

    # Resume
    dataset = reserve_unprocessed_queries(args.output_file, dataset)
    print(f"After resume filter: {len(dataset)} remaining")

    if not dataset:
        print("All samples processed.")
        return

    # Initialize debate engine
    debate = AgentArkDebate(
        model_name=args.model_name,
        num_agents=args.num_agents,
        num_rounds=args.num_rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        dataset_type=dataset_type,
    )

    # Run inference
    lock = threading.Lock()
    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Batches"):
        batch = dataset[start:start + args.batch_size]
        results = debate.inference_batch(batch)
        for sample, result in zip(batch, results):
            output = {**sample, **result, "dataset_type": dataset_type,
                      "num_agents": args.num_agents, "num_rounds": args.num_rounds}
            write_to_jsonl(lock, args.output_file, output)

    print(f"\nDone. Results saved to: {args.output_file}")

    # Accuracy evaluation
    if args.eval_accuracy:
        metrics = compute_accuracy(args.output_file, max_choice)
        metrics_file = str(Path(args.output_file).with_suffix(".metrics.json"))
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n--- Accuracy ---")
        print(f"Judge accuracy: {metrics['judge_accuracy']:.4f} "
              f"({metrics['judge_correct']}/{metrics['judge_total']})")
        print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
