"""
Generate Multi-Agent Interaction Graphs (MAGs) for cultural alignment tasks.

This script converts multi-agent inference data (from either HF-CAC or RECONCILE)
into the MAG format required by MAGDi training. It supports both NormAD and
CultureBench datasets.

Data source modes:
  --data_source hf_cac    : Use HF-CAC inference output (with Guardian/Auditor roles)
  --data_source reconcile : Use RECONCILE inference output (symmetric agents)

The output MAG format follows the original MAGDi paper's structure:
  - Each sample has multiple agent outputs across rounds
  - Each node has: reasoning, answer, correctness label
  - Graph structure encodes which agents saw which outputs

Usage:
    # From HF-CAC data (6 agents, 1 round = 12 nodes max)
    python generate_mag_data.py \
        --data_source hf_cac \
        --input_file ../Cul/data/culturalbench_hf_cac_inference_50_20260531_152401.jsonl \
        --dataset culturalbench \
        --output_file MAG/culturalbench_hf_cac.json

    # From RECONCILE data (5 agents, 1 round = 10 nodes max)
    python generate_mag_data.py \
        --data_source reconcile \
        --input_file ../Cul/data/culturalbench_reconcile_inference.jsonl \
        --dataset culturalbench \
        --output_file MAG/culturalbench_reconcile.json

    # NormAD dataset
    python generate_mag_data.py \
        --data_source hf_cac \
        --input_file ../Cul/data/normad_hf_cac_inference.jsonl \
        --dataset normad \
        --output_file MAG/normad_hf_cac.json
"""

import os
import re
import json
import argparse
import random
from typing import List, Dict, Tuple, Optional

random.seed(42)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HF-CAC: 6 cultural agents + 1 judge = 7 solutions
# RECONCILE: 5 cultural agents + 1 judge = 6 solutions
# In MAGDi's original setup: 3 agents × 4 rounds = 12 nodes max
# For cultural tasks (single-round negotiation):
#   HF-CAC: 6 agents × 1 round (pre-interaction) + judge = treat as 2-round
#   We model: Round 0 = agent independent answers, Round 1 = after seeing Guardian
#   This gives us a natural 2-round structure for HF-CAC

HF_CAC_NUM_AGENTS = 6
RECONCILE_NUM_AGENTS = 5


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_hf_cac_response(response: str) -> List[Dict]:
    """
    Parse HF-CAC response into individual solutions.
    
    Format:
        ===== Solution N [ROLE] =====
        <answer_number>
        <reasoning>
    
    Returns list of dicts with keys: idx, role, answer, reasoning
    """
    pattern = r"===== Solution (\d+) \[(\w+)\] =====\s*\n(.*?)(?====== Solution|\Z)"
    matches = re.findall(pattern, response, re.DOTALL)
    
    solutions = []
    for idx_str, role, content in matches:
        content = content.strip()
        lines = content.split('\n', 1)
        # First line is the answer number
        answer = lines[0].strip() if lines else ""
        reasoning = lines[1].strip() if len(lines) > 1 else ""
        solutions.append({
            'idx': int(idx_str),
            'role': role,  # GUARDIAN, AUDITOR, or JUDGE
            'answer': answer,
            'reasoning': reasoning
        })
    return solutions


def parse_reconcile_response(response: str) -> List[Dict]:
    """
    Parse RECONCILE response into individual solutions.
    
    Format (no role tags):
        ===== Solution N =====
        <answer_number>
        <reasoning>
    
    Returns list of dicts with keys: idx, role, answer, reasoning
    """
    pattern = r"===== Solution (\d+) =====\s*\n(.*?)(?====== Solution|\Z)"
    matches = re.findall(pattern, response, re.DOTALL)
    
    solutions = []
    for idx_str, content in matches:
        content = content.strip()
        lines = content.split('\n', 1)
        answer = lines[0].strip() if lines else ""
        reasoning = lines[1].strip() if len(lines) > 1 else ""
        # In RECONCILE, last solution is judge, others are agents
        solutions.append({
            'idx': int(idx_str),
            'role': 'AGENT',  # Will be updated: last one becomes JUDGE
            'answer': answer,
            'reasoning': reasoning
        })
    
    # Mark last solution as JUDGE
    if solutions:
        solutions[-1]['role'] = 'JUDGE'
    
    return solutions


# ---------------------------------------------------------------------------
# MAG construction
# ---------------------------------------------------------------------------

def build_mag_from_hf_cac(sample: Dict, dataset: str) -> Optional[Dict]:
    """
    Build a MAG from a single HF-CAC inference sample.
    
    HF-CAC structure (6 agents, structured negotiation):
      - Phase 1: Guardian generates independently (Round 0 for Guardian)
      - Phase 2: Auditors generate after seeing Guardian (Round 1 for Auditors)
      - Judge: Final arbitration (Round 1, separate node)
    
    We model this as a 2-round interaction with 6 agents:
      Round 0: All agents' "initial" reasoning (Guardian is authoritative)
      Round 1: Judge's final reasoning (synthesizes all)
    
    Node layout (max 13 nodes, but we use 7 = 6 agents + 1 judge):
      Nodes 0-5: Agent solutions (one is Guardian, rest are Auditors)
      Node 6: Judge solution
    
    Edge structure:
      - Guardian (node at guardian_idx) → all Auditor nodes
      - All agent nodes → Judge node
    """
    response = sample.get('response', '')
    gt = str(sample.get('gt', ''))
    
    solutions = parse_hf_cac_response(response)
    if len(solutions) < 7:  # Need at least 6 agents + 1 judge
        return None
    
    # Separate agent solutions and judge
    agent_solutions = [s for s in solutions if s['role'] != 'JUDGE'][:6]
    judge_solutions = [s for s in solutions if s['role'] == 'JUDGE']
    
    if len(agent_solutions) < 6 or len(judge_solutions) < 1:
        return None
    
    judge_sol = judge_solutions[0]
    
    # Build MAG entry in the format expected by MAGDi
    # We follow the original MAGDi format: agent_output_round fields
    mag_entry = {
        'question': sample.get('query', ''),
        'gold_answer': gt,
        'country': sample.get('country', ''),
        'dataset': dataset,
        'data_source': 'hf_cac',
        'guardian_idx': sample.get('guardian_idx', 0),
    }
    
    # Agent names for HF-CAC (6 agents)
    agent_names = ['agent0', 'agent1', 'agent2', 'agent3', 'agent4', 'agent5']
    
    # Round 0: Each agent's solution
    for i, sol in enumerate(agent_solutions):
        key = f"{agent_names[i]}_output_0"
        mag_entry[key] = {
            'reasoning': sol['reasoning'],
            'answer': sol['answer'],
            'role': sol['role']
        }
    
    # Round 1: Judge's solution (treated as a synthesis round)
    mag_entry['judge_output'] = {
        'reasoning': judge_sol['reasoning'],
        'answer': judge_sol['answer'],
        'role': 'JUDGE'
    }
    
    return mag_entry


def build_mag_from_reconcile(sample: Dict, dataset: str) -> Optional[Dict]:
    """
    Build a MAG from a single RECONCILE inference sample.
    
    RECONCILE structure (5 symmetric agents):
      - All agents generate independently (Round 0)
      - Judge synthesizes (Round 1)
    
    Node layout (6 nodes = 5 agents + 1 judge):
      Nodes 0-4: Agent solutions (symmetric, no Guardian)
      Node 5: Judge solution
    
    Edge structure:
      - All agent nodes are independent (no inter-agent edges in Round 0)
      - All agent nodes → Judge node
    """
    response = sample.get('response', '')
    gt = str(sample.get('gt', ''))
    
    solutions = parse_reconcile_response(response)
    if len(solutions) < 6:  # Need at least 5 agents + 1 judge
        return None
    
    agent_solutions = [s for s in solutions if s['role'] != 'JUDGE'][:5]
    judge_solutions = [s for s in solutions if s['role'] == 'JUDGE']
    
    if len(agent_solutions) < 5 or len(judge_solutions) < 1:
        return None
    
    judge_sol = judge_solutions[0]
    
    mag_entry = {
        'question': sample.get('query', ''),
        'gold_answer': gt,
        'country': sample.get('country', ''),
        'dataset': dataset,
        'data_source': 'reconcile',
    }
    
    agent_names = ['agent0', 'agent1', 'agent2', 'agent3', 'agent4']
    
    # Round 0: Each agent's solution (symmetric)
    for i, sol in enumerate(agent_solutions):
        key = f"{agent_names[i]}_output_0"
        mag_entry[key] = {
            'reasoning': sol['reasoning'],
            'answer': sol['answer'],
            'role': 'AGENT'
        }
    
    # Judge
    mag_entry['judge_output'] = {
        'reasoning': judge_sol['reasoning'],
        'answer': judge_sol['answer'],
        'role': 'JUDGE'
    }
    
    return mag_entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_inference_data(input_file: str) -> List[Dict]:
    """Load JSONL inference data."""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate MAG data for cultural alignment tasks from multi-agent inference."
    )
    parser.add_argument('--data_source', type=str, required=True,
                        choices=['hf_cac', 'reconcile'],
                        help="Source of multi-agent inference data")
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to input JSONL file (multi-agent inference output)")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['normad', 'culturalbench'],
                        help="Dataset type (affects answer parsing)")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to output JSON file (MAG format)")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument('--splits_pkl', type=str, default='',
                        help="Path to splits pkl file (from split_data.py). "
                             "If provided, only processes samples in the specified split.")
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help="Which split to use (only effective with --splits_pkl)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading inference data from: {args.input_file}")
    raw_data = load_inference_data(args.input_file)
    print(f"  Loaded {len(raw_data)} samples")
    
    # Filter by splits_pkl if provided
    if args.splits_pkl:
        import pickle
        print(f"  Filtering by splits_pkl ({args.split} split): {args.splits_pkl}")
        with open(args.splits_pkl, 'rb') as f:
            splits = pickle.load(f)
        split_samples = splits[args.split]
        # Build index set using (query_prefix, country) for matching
        split_keys = set()
        for s in split_samples:
            key = (s.get('query', '')[:200].strip().lower(), s.get('country', '').strip().lower())
            split_keys.add(key)
        # Filter raw_data to only include samples in the target split
        filtered = []
        for sample in raw_data:
            key = (sample.get('query', '')[:200].strip().lower(), sample.get('country', '').strip().lower())
            if key in split_keys:
                filtered.append(sample)
        print(f"  Filtered: {len(raw_data)} -> {len(filtered)} samples (matched {args.split} split)")
        raw_data = filtered
    
    if args.max_samples > 0:
        raw_data = raw_data[:args.max_samples]
        print(f"  Using first {args.max_samples} samples")
    
    # Build MAGs
    print(f"Building MAGs (data_source={args.data_source}, dataset={args.dataset})...")
    mags = []
    skipped = 0
    
    for sample in raw_data:
        if args.data_source == 'hf_cac':
            mag = build_mag_from_hf_cac(sample, args.dataset)
        else:
            mag = build_mag_from_reconcile(sample, args.dataset)
        
        if mag is not None:
            mags.append(mag)
        else:
            skipped += 1
    
    print(f"  Built {len(mags)} MAGs, skipped {skipped} (parse failures)")
    
    # Save
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(mags, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved to: {args.output_file}")
    
    # Print statistics
    if mags:
        correct_counts = []
        for mag in mags:
            gt = mag['gold_answer']
            num_correct = 0
            for key in mag:
                if key.endswith('_output_0') and isinstance(mag[key], dict):
                    if str(mag[key].get('answer', '')).strip() == gt:
                        num_correct += 1
            correct_counts.append(num_correct)
        
        avg_correct = sum(correct_counts) / len(correct_counts)
        num_agents = HF_CAC_NUM_AGENTS if args.data_source == 'hf_cac' else RECONCILE_NUM_AGENTS
        print(f"\n  Statistics:")
        print(f"    Avg correct agents per sample: {avg_correct:.2f} / {num_agents}")
        print(f"    Samples with all correct: {sum(1 for c in correct_counts if c == num_agents)}")
        print(f"    Samples with all wrong: {sum(1 for c in correct_counts if c == 0)}")


if __name__ == '__main__':
    main()
